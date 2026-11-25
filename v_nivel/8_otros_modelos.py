"""Dos comprobaciones de 3_modelos.py: otra familia de modelos y la lista de revisión a varios tamaños.

1. Un bosque aleatorio, para ver si el resultado depende de XGBoost. Usa las mismas variables, la misma
   muestra de entrenamiento con los mismos pesos y los mismos años de prueba que 3_modelos.py.
2. Umbrales: si el banco revisa el 1, 5, 10 o 20 % de la cartera con más riesgo, qué parte de los impagos
   del año encuentra (sensibilidad) y qué parte de los revisados impaga (precisión), con los cuatro modelos.
   La regresión logística y XGBoost no se reentrenan: se cargan los de data/sencillo/modelos.

Lee data/sencillo/panel y data/sencillo/modelos. Escribe resultados/otros_modelos.csv y resultados/umbrales.csv.
Tarda unos 40 minutos con tarjeta gráfica (de 1,5 minutos el año 2008 a 3 el 2025); el piloto, 3 minutos.

    python v_nivel/8_otros_modelos.py            de 2008 a 2025, con la tabla completa
    python v_nivel/8_otros_modelos.py piloto     2010, 2016 y 2025, con la tabla piloto
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

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
REVISADO = [1, 5, 10, 20]  # % de la cartera que se revisa, empezando por el de más riesgo


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


def bosque(semilla):
    """Bosque aleatorio de 50 árboles independientes de profundidad 12; la predicción es su media.

    Cada árbol ve el 63 % de las filas (las distintas de un bootstrap) y elige cada corte entre una cuarta
    parte de las variables (unas 5 de 22, la raíz cuadrada habitual). Con error cuadrático sobre el 0/1 los
    cortes son los del índice de Gini y cada hoja da la tasa de impago de sus préstamos, así que la media es
    ya una probabilidad. Es la forma que recomienda xgboost 3.4, que da por obsoleto XGBRFClassifier.
    Se entrenan seis de 50 y se promedian, que es un bosque de 300: los 300 de una vez no caben en la tarjeta.
    """
    return xgb.XGBRegressor(n_estimators=1, num_parallel_tree=50, learning_rate=1, max_depth=12,
                            subsample=0.63, colsample_bynode=0.25, reg_lambda=0, min_child_weight=100,
                            tree_method="hist", enable_categorical=True, device="cuda", random_state=semilla)


panel = pl.scan_parquet(PANEL)
en_muestra = (pl.col("impago") == 1) | (pl.col("id").hash(seed=0) % 10 == 0)  # como en 3_modelos.py
categorias = joblib.load(MODELOS / "categorias.joblib")  # los niveles con que se entrenaron los modelos
muestra = preparar(panel.filter(en_muestra, pl.col("anio") < max(ANIOS)).collect(), categorias)
print(f"Muestra de entrenamiento: {len(muestra):,} filas", flush=True)

otros, umbrales = [], []
for anio in ANIOS:
    inicio = time.time()
    entreno = muestra[muestra["anio"] < anio]
    prueba = preparar(panel.filter(pl.col("anio") == anio).collect(), categorias)
    y, y_prueba = entreno["impago"], prueba["impago"].to_numpy()
    pesos = pd.Series(np.where(y == 1, 1, 10), index=entreno.index)

    predicciones = {"regla": regla(entreno, pesos, prueba)}
    for nombre in ["logit", "xgboost"]:
        predicciones[nombre] = joblib.load(MODELOS / f"{nombre}_{anio}.joblib").predict_proba(prueba[VARIABLES])[:, 1]
    reloj = time.time()
    predicciones["bosque"] = np.zeros(len(prueba))
    for semilla in range(6):
        modelo = bosque(semilla).fit(entreno[VARIABLES], y, sample_weight=pesos).set_params(device="cpu")
        predicciones["bosque"] += modelo.predict(prueba[VARIABLES]).clip(0, 1) / 6
    segundos = round(time.time() - reloj)

    # Lista de revisión de cada modelo: de más a menos riesgo; los empates se deshacen al azar, como en 3_modelos.py.
    azar = np.random.default_rng(0).random(len(y_prueba))
    for nombre, p in predicciones.items():
        orden = np.lexsort((azar, p))
        for revisado in REVISADO:
            elegidos = orden[-len(p) * revisado // 100 :]
            umbrales.append({"anio": anio, "modelo": nombre, "revisado": revisado, "prestamos": len(elegidos),
                             "sensibilidad": y_prueba[elegidos].sum() / y_prueba.sum(),
                             "precision": y_prueba[elegidos].mean()})
    p = predicciones["bosque"]
    otros.append({"anio": anio, "modelo": "bosque", "auc": roc_auc_score(y_prueba, p),
                  "captura": y_prueba[np.lexsort((azar, p))[-len(p) // 10 :]].sum() / y_prueba.sum(),
                  "calibracion": p.mean() / y_prueba.mean(),
                  "segundos": segundos})
    print(f"{anio}: bosque AUC {otros[-1]['auc']:.4f}  captura {otros[-1]['captura']:.3f}  calibración "
          f"{otros[-1]['calibracion']:.2f}  {segundos} s de entrenar y predecir; año completo en "
          f"{time.time() - inicio:.0f} s", flush=True)

RESULTADOS.mkdir(parents=True, exist_ok=True)
pd.DataFrame(otros).to_csv(RESULTADOS / "otros_modelos.csv", index=False)
pd.DataFrame(umbrales).to_csv(RESULTADOS / "umbrales.csv", index=False)
