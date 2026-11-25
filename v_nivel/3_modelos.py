"""Cuatro modelos para predecir qué préstamos impagarán durante el año, probados año a año.

Para cada año de prueba se entrena con todos los años anteriores y se predice el año entero,
que el modelo no ha visto. Los cuatro modelos, de menos a más complejo:

    regla    el baseline: la tasa de impago del año anterior según los meses de retraso del préstamo
    logit    regresión logística, la referencia en riesgo de crédito
    xgboost  gradient boosting sobre árboles, con parámetros fijos
    xgboost_ajustado  el mismo, con los parámetros elegidos cada año por validación

Son los modelos sin nivel. 6_nivel.py repite la regresión logística y el XGBoost dándoles aparte el nivel
de impago del año.

    python v_nivel/3_modelos.py            de 2008 a 2025, con la tabla completa
    python v_nivel/3_modelos.py piloto     2010, 2016 y 2025, con la tabla piloto
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
PILOTO = "piloto" in sys.argv
PANEL = RAIZ / "data" / "sencillo" / ("panel_piloto" if PILOTO else "panel") / "*.parquet"
MODELOS = RAIZ / "data" / "sencillo" / ("modelos_piloto" if PILOTO else "modelos")
RESULTADOS = Path(__file__).resolve().parent / ("resultados_piloto" if PILOTO else "resultados")
ANIOS = [2010, 2016, 2025] if PILOTO else range(2008, 2026)

NUMERICAS = ["fico", "ltv", "dti", "importe", "tipo_inicial", "plazo", "seguro", "unidades", "prestatarios",
             "antiguedad", "saldo", "ltv_actual", "mora_actual", "mora_maxima_12m", "meses_en_mora_12m",
             "modificado"]
CATEGORICAS = ["ocupacion", "finalidad", "primer_comprador", "canal", "inmueble", "estado"]
VARIABLES = NUMERICAS + CATEGORICAS
# El impago es raro. Para entrenar se usan todos los impagos y 1 de cada 10 préstamos que no impagan,
# y cada uno de estos cuenta por 10. La prueba se hace siempre con todos los préstamos del año.
UNO_DE_CADA = 10
# Las doce combinaciones de parámetros que se prueban al ajustar el xgboost.
REJILLA = [{"max_depth": profundidad, "learning_rate": tasa, "min_child_weight": minimo}
           for profundidad in [4, 6, 8] for tasa in [0.05, 0.1] for minimo in [1, 20]]


def preparar(tabla, categorias):
    """Pasa la tabla a pandas, con las variables de texto como categorías con los mismos niveles siempre."""
    datos = tabla.select("anio", "impago", *VARIABLES).to_pandas()
    datos[NUMERICAS] = datos[NUMERICAS].astype("float32")
    for variable in CATEGORICAS:
        datos[variable] = pd.Categorical(datos[variable].fillna("sin dato"), categories=categorias[variable])
    return datos


def regla(entreno, pesos, prueba):
    """A cada préstamo, la tasa de impago que tuvieron el año anterior los de su mismo retraso."""
    ultimo = entreno["anio"] == entreno["anio"].max()
    retraso = entreno["mora_actual"].clip(upper=2).fillna(0)[ultimo]  # 0, 1 o 2 meses
    impagos = (entreno["impago"] * pesos)[ultimo].groupby(retraso).sum()
    prestamos = pesos[ultimo].groupby(retraso).sum()
    return prueba["mora_actual"].clip(upper=2).fillna(0).map(impagos / prestamos).to_numpy()


def logit():
    """Regresión logística: números sin dato a la mediana y a escala común, categorías como indicadores."""
    numeros = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    columnas = ColumnTransformer([("numeros", numeros, NUMERICAS),
                                  ("categorias", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
    return make_pipeline(columnas, LogisticRegression(max_iter=500))


def xgboost():
    """Gradient boosting: 300 árboles de profundidad 6; lee los vacíos y las categorías tal como vienen.

    Se entrena en la tarjeta gráfica, diez veces más rápido que en el procesador; sin tarjeta, device="cpu".
    """
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                         enable_categorical=True, device="cuda", random_state=0)


def ajustar(entreno, y, pesos):
    """Elige los parámetros del xgboost sin mirar el año de prueba y devuelve el modelo y lo elegido.

    Cada combinación se entrena sin el último año disponible y se mide en él, añadiendo árboles hasta
    que deja de mejorar. Con la mejor combinación se vuelve a entrenar usando todos los años.
    """
    ultimo = entreno["anio"] == entreno["anio"].max()
    antes, validacion = entreno[~ultimo][VARIABLES], entreno[ultimo][VARIABLES]
    mejor_error = np.inf
    for parametros in REJILLA:
        modelo = xgboost().set_params(**parametros, n_estimators=1000, early_stopping_rounds=20)
        modelo.fit(antes, y[~ultimo], sample_weight=pesos[~ultimo], eval_set=[(validacion, y[ultimo])],
                   sample_weight_eval_set=[pesos[ultimo]], verbose=False)
        if modelo.best_score < mejor_error:
            mejor_error = modelo.best_score
            elegido = {**parametros, "n_estimators": modelo.best_iteration + 1}
    modelo = xgboost().set_params(**elegido)
    return modelo.fit(entreno[VARIABLES], y, sample_weight=pesos), elegido


def medir(y, p):
    """Cinco medidas de una predicción frente a lo que pasó."""
    # El 10 % de préstamos con más riesgo según el modelo; los empates se deshacen al azar.
    orden = np.lexsort((np.random.default_rng(0).random(len(p)), p))
    mas_riesgo = orden[-len(p) // 10 :]
    return {
        "auc": roc_auc_score(y, p),  # probabilidad de ordenar bien a un préstamo que impaga y a uno que no
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),  # error cuadrático de la probabilidad
        "calibracion": p.mean() / y.mean(),  # 1 si el modelo acierta la tasa media del año
        "captura": y[mas_riesgo].sum() / y.sum(),  # parte de los impagos que cae en ese 10 %
    }


panel = pl.scan_parquet(PANEL)
en_muestra = (pl.col("impago") == 1) | (pl.col("id").hash(seed=0) % UNO_DE_CADA == 0)
muestra = panel.filter(en_muestra, pl.col("anio") < max(ANIOS)).collect()
categorias = {v: sorted(muestra[v].drop_nulls().unique()) + ["sin dato"] for v in CATEGORICAS}
muestra = preparar(muestra, categorias)
print(f"Muestra de entrenamiento: {len(muestra):,} filas de {muestra['anio'].min()} a {muestra['anio'].max()}")

MODELOS.mkdir(parents=True, exist_ok=True)
joblib.dump(categorias, MODELOS / "categorias.joblib")  # 8_otros_modelos.py necesita los mismos niveles
RESULTADOS.mkdir(parents=True, exist_ok=True)
metricas, calibracion, segmentos, importancia, elegidos = [], [], [], [], []
for anio in ANIOS:
    entreno = muestra[muestra["anio"] < anio]
    prueba = preparar(panel.filter(pl.col("anio") == anio).collect(), categorias)
    y, y_prueba = entreno["impago"], prueba["impago"].to_numpy()
    pesos = pd.Series(np.where(y == 1, 1, UNO_DE_CADA), index=entreno.index)
    print(f"{anio}: {len(entreno):,} filas de entrenamiento, {len(prueba):,} de prueba", flush=True)

    predicciones = {"regla": regla(entreno, pesos, prueba)}
    segundos = {"regla": 0}
    for nombre in ["logit", "xgboost", "xgboost_ajustado"]:
        inicio = time.time()
        if nombre == "logit":
            modelo = logit().fit(entreno[VARIABLES], y, logisticregression__sample_weight=pesos)
        elif nombre == "xgboost":
            modelo = xgboost().fit(entreno[VARIABLES], y, sample_weight=pesos)
        else:
            modelo, elegido = ajustar(entreno, y, pesos)
            elegidos.append({"anio": anio, **elegido})
        if nombre != "logit":
            modelo.set_params(device="cpu")  # para predecir basta el procesador, donde ya están los datos
        predicciones[nombre] = modelo.predict_proba(prueba[VARIABLES])[:, 1]
        segundos[nombre] = round(time.time() - inicio)
        joblib.dump(modelo, MODELOS / f"{nombre}_{anio}.joblib")
        if nombre == "xgboost":
            principal = modelo

    al_corriente = (prueba["mora_actual"].fillna(0) == 0).to_numpy()
    for nombre, p in predicciones.items():
        fila = {"anio": anio, "modelo": nombre, "prestamos": len(y_prueba), "impagos": int(y_prueba.sum())}
        metricas.append({**fila, **medir(y_prueba, p), "segundos": segundos[nombre]})
        print(f"  {nombre:16} AUC {metricas[-1]['auc']:.4f}  captura {metricas[-1]['captura']:.3f}  "
              f"calibración {metricas[-1]['calibracion']:.2f}  {segundos[nombre]} s", flush=True)
        # Dónde acierta cada modelo: por separado en los préstamos al corriente y en los que ya van con retraso.
        for segmento, filas in [("al corriente", al_corriente), ("con retraso", ~al_corriente)]:
            segmentos.append({"anio": anio, "modelo": nombre, "segmento": segmento, "prestamos": int(filas.sum()),
                              "tasa": y_prueba[filas].mean(), "auc": roc_auc_score(y_prueba[filas], p[filas])})
        # Calibración: en diez grupos de riesgo creciente, lo predicho frente a lo que pasó.
        for grupo, filas in enumerate(np.array_split(np.argsort(p), 10), 1):
            calibracion.append({"anio": anio, "modelo": nombre, "grupo": grupo, "predicho": p[filas].mean(),
                                "observado": y_prueba[filas].mean()})

    # Qué variables mueven la predicción del xgboost fijo: media del valor absoluto de su aportación (SHAP).
    unos_cuantos = prueba[VARIABLES].sample(min(len(prueba), 100_000), random_state=0)
    aportaciones = principal.get_booster().predict(xgb.DMatrix(unos_cuantos, enable_categorical=True), pred_contribs=True)
    importancia.append(pd.DataFrame({"anio": anio, "variable": VARIABLES,
                                     "importancia": np.abs(aportaciones[:, :-1]).mean(axis=0)}))

pd.DataFrame(metricas).to_csv(RESULTADOS / "metricas.csv", index=False)
pd.DataFrame(segmentos).to_csv(RESULTADOS / "segmentos.csv", index=False)
pd.DataFrame(calibracion).to_csv(RESULTADOS / "calibracion.csv", index=False)
pd.concat(importancia).to_csv(RESULTADOS / "importancia.csv", index=False)
pd.DataFrame(elegidos).to_csv(RESULTADOS / "parametros.csv", index=False)
