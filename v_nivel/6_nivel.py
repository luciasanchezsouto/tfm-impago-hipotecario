"""Modelo con nivel: el riesgo de cada préstamo como nivel del año más perfil del préstamo.

La idea viene del análisis de series temporales, que separa lo observado en componentes y modela cada una:

    riesgo de un préstamo = nivel del año + perfil del préstamo

    nivel   la tasa de impago del año. Al entrenar se conoce, y se le da al modelo para que no tenga que
            explicarla con las variables del préstamo.
    perfil  lo que aprende el modelo: quién es más arriesgado que quién dentro de cada año.

XGBoost recibe el nivel como punto de partida de cada fila; la regresión logística, como una variable más.
Lo demás es igual que en 3_modelos.py: la misma tabla, la misma muestra, los mismos pesos y los mismos
parámetros. Para cada año de prueba se entrena con todos los años anteriores y se predice el año entero.
XGBoost con nivel es el modelo final del trabajo.

Para predecir el año T hace falta su nivel, que el 1 de enero no se conoce. Se prueban tres previsiones,
todas con las tasas de años ya cerrados: la del último año, un suavizado exponencial simple y la mediana
de los cinco últimos. El orden de los préstamos (AUC, captura y listas de revisión) no depende de esa
previsión; la calibración, sí.

Lee data/sencillo/panel y descriptivo/por_anio.csv. Escribe:

    resultados/nivel.csv            por año y modelo: AUC, captura al 10 %, tasa real y, para cada
                                    previsión del nivel, la tasa prevista y la calibración
    resultados/umbrales_nivel.csv   si se revisa el 1, 5, 10 o 20 % de la cartera con más riesgo, qué parte
                                    de los impagos se encuentra (sensibilidad) y qué parte de los revisados
                                    impaga (precisión)
    resultados/importancia_nivel.csv
                                    por año y variable, cuánto mueve la predicción del XGBoost con nivel:
                                    media del valor absoluto de su aportación (SHAP), como importancia.csv
                                    de 3_modelos.py
    resultados/calibracion_nivel.csv
                                    por año y modelo, en diez grupos de riesgo creciente, el impago predicho
                                    y el real, como calibracion.csv de 3_modelos.py
    resultados/segmentos_nivel.csv  por año y modelo, el AUC por separado en los préstamos al corriente y en
                                    los que van con retraso, como segmentos.csv de 3_modelos.py
                                    En estas dos, la probabilidad es la que da el nivel previsto con la tasa
                                    del último año.
    data/sencillo/modelos_nivel     los dos modelos de cada año (logit_2025.joblib, xgboost_2025.joblib...)
                                    y categorias.joblib; los de 2025 se copian a v_nivel/modelos para la
                                    aplicación

Tarda unos 35 minutos con tarjeta gráfica; un año suelto, de 2 a 4 minutos.

    python v_nivel/6_nivel.py               de 2008 a 2025, con la tabla completa
    python v_nivel/6_nivel.py 2025          solo los años indicados (uno o varios); los resultados van a
                                            resultados/nivel_parcial.csv, umbrales_nivel_parcial.csv,
                                            importancia_nivel_parcial.csv, calibracion_nivel_parcial.csv y
                                            segmentos_nivel_parcial.csv
    python v_nivel/6_nivel.py piloto        2010, 2016 y 2025, con la tabla piloto
"""

import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from scipy.special import expit, logit as logaritmo_razon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
TODOS = range(2008, 2026)
for argumento in sys.argv[1:]:
    if argumento != "piloto" and not (argumento.isdigit() and int(argumento) in TODOS):
        sys.exit(f"Argumento no válido: {argumento}. Se admiten años de 2008 a 2025 y «piloto».")
PILOTO = "piloto" in sys.argv[1:]
SUELTOS = sorted({int(a) for a in sys.argv[1:] if a.isdigit()})
ANIOS = SUELTOS or ([2010, 2016, 2025] if PILOTO else TODOS)
PANEL = RAIZ / "data" / "sencillo" / ("panel_piloto" if PILOTO else "panel") / "*.parquet"
MODELOS = RAIZ / "data" / "sencillo" / ("modelos_nivel_piloto" if PILOTO else "modelos_nivel")
APLICACION = AQUI / "modelos"  # los modelos que usa 5_app.py
RESULTADOS = AQUI / ("resultados_piloto" if PILOTO else "resultados")
SUFIJO = "_parcial" if SUELTOS else ""  # con años sueltos no se pisan los resultados de los 18 años

NUMERICAS = ["fico", "ltv", "dti", "importe", "tipo_inicial", "plazo", "seguro", "unidades", "prestatarios",
             "antiguedad", "saldo", "ltv_actual", "mora_actual", "mora_maxima_12m", "meses_en_mora_12m",
             "modificado"]
CATEGORICAS = ["ocupacion", "finalidad", "primer_comprador", "canal", "inmueble", "estado"]
VARIABLES = NUMERICAS + CATEGORICAS
UNO_DE_CADA = 10
REVISADO = [1, 5, 10, 20]  # % de la cartera que se revisa, empezando por el de más riesgo


def preparar(tabla, categorias):
    """Pasa la tabla a pandas, con las variables de texto como categorías con los mismos niveles siempre."""
    datos = tabla.select("anio", "impago", *VARIABLES).to_pandas()
    datos[NUMERICAS] = datos[NUMERICAS].astype("float32")
    for variable in CATEGORICAS:
        datos[variable] = pd.Categorical(datos[variable].fillna("sin dato"), categories=categorias[variable])
    return datos


def logit():
    """La regresión logística de 3_modelos.py, con el nivel del año como una variable numérica más."""
    numeros = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    columnas = ColumnTransformer([("numeros", numeros, NUMERICAS + ["nivel"]),
                                  ("categorias", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
    return make_pipeline(columnas, LogisticRegression(max_iter=500))


def xgboost():
    """El XGBoost de 3_modelos.py. Se entrena en la tarjeta gráfica; sin tarjeta, device="cpu"."""
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                             enable_categorical=True, device="cuda", random_state=0)


def prever_nivel(tasas):
    """Tres previsiones de la tasa del año siguiente, a partir de las tasas de los años ya cerrados."""
    suavizado = SimpleExpSmoothing(tasas.to_numpy(), initialization_method="estimated").fit().forecast(1)[0]
    return {"ultimo": tasas.iloc[-1], "suavizado": suavizado, "mediana": tasas.iloc[-5:].median()}


def medir(y, puntuacion):
    """AUC y captura (parte de los impagos en el 10 % con más riesgo); los empates se deshacen al azar."""
    orden = np.lexsort((np.random.default_rng(0).random(len(puntuacion)), puntuacion))
    mas_riesgo = orden[-len(puntuacion) // 10 :]
    return {"auc": roc_auc_score(y, puntuacion), "captura": y[mas_riesgo].sum() / y.sum()}


def listas(anio, modelo, y, puntuacion):
    """La lista de revisión a cuatro tamaños, ordenada como en medir() y como en 8_otros_modelos.py."""
    orden = np.lexsort((np.random.default_rng(0).random(len(puntuacion)), puntuacion))
    filas = []
    for revisado in REVISADO:
        elegidos = orden[-len(puntuacion) * revisado // 100 :]
        filas.append({"anio": anio, "modelo": modelo, "revisado": revisado, "prestamos": len(elegidos),
                      "sensibilidad": y[elegidos].sum() / y.sum(), "precision": y[elegidos].mean()})
    return filas


# El nivel de cada año: la tasa de impago de toda la cartera, que calcula 2_descriptivo.py.
tasa = pd.read_csv(AQUI / "descriptivo" / "por_anio.csv").set_index("anio")["tasa"]

# La muestra y los niveles de las categorías salen siempre de los años anteriores a 2025, también con años
# sueltos, para que cada año dé lo mismo que en la ejecución completa.
panel = pl.scan_parquet(PANEL)
en_muestra = (pl.col("impago") == 1) | (pl.col("id").hash(seed=0) % UNO_DE_CADA == 0)
muestra = panel.filter(en_muestra, pl.col("anio") < max(TODOS)).collect()
categorias = {v: sorted(muestra[v].drop_nulls().unique()) + ["sin dato"] for v in CATEGORICAS}
muestra = preparar(muestra, categorias)
muestra["nivel"] = logaritmo_razon(muestra["anio"].map(tasa)).astype("float32")  # el nivel de su año, ya conocido
print(f"Muestra de entrenamiento: {len(muestra):,} filas", flush=True)

# Cómo se usa cada modelo guardado con préstamos nuevos, preparados con preparar() y categorias.joblib.
# z es el nivel previsto para el año, z = log(tasa / (1 - tasa)), por ejemplo con la tasa del último año:
#   logit    (Pipeline, espera las 22 variables y la columna «nivel»):
#            modelo.predict_proba(datos[VARIABLES].assign(nivel=np.float32(z)))[:, 1]
#   xgboost  (se entrenó con base_margin = nivel, así que hay que pasárselo también al predecir; sin él
#            parte de una tasa fija interna de casi el 1 % y la probabilidad sale descalibrada):
#            modelo.predict_proba(datos[VARIABLES], base_margin=np.full(len(datos), z))[:, 1]
#            que es lo mismo que expit(z + perfil), con el perfil de más abajo. Para las aportaciones de
#            cada variable, el mismo base_margin en xgb.DMatrix(..., enable_categorical=True, base_margin=...).
MODELOS.mkdir(parents=True, exist_ok=True)
joblib.dump(categorias, MODELOS / "categorias.joblib")
RESULTADOS.mkdir(exist_ok=True)
filas, umbrales, importancia, calibracion, segmentos = [], [], [], [], []
for anio in ANIOS:
    inicio = time.time()
    entreno = muestra[muestra["anio"] < anio]
    prueba = preparar(panel.filter(pl.col("anio") == anio).collect(), categorias)
    y, y_prueba = entreno["impago"], prueba["impago"].to_numpy()
    pesos = np.where(y == 1, 1, UNO_DE_CADA)
    previstas = prever_nivel(tasa[tasa.index < anio])

    # XGBoost: el nivel es el punto de partida, y los árboles aprenden lo que se suma o se resta.
    modelo = xgboost().fit(entreno[VARIABLES], y, sample_weight=pesos, base_margin=entreno["nivel"]).set_params(device="cpu")
    joblib.dump(modelo, MODELOS / f"xgboost_{anio}.joblib")
    perfil = modelo.predict(prueba[VARIABLES], base_margin=np.zeros(len(prueba)), output_margin=True)
    fila = {"anio": anio, "modelo": "xgboost", **medir(y_prueba, perfil), "tasa_real": y_prueba.mean()}
    for nombre, prevista in previstas.items():
        fila |= {"tasa_" + nombre: prevista,
                 "calibracion_" + nombre: expit(logaritmo_razon(prevista) + perfil).mean() / y_prueba.mean()}
    filas.append(fila)
    umbrales += listas(anio, "xgboost", y_prueba, perfil)
    probabilidades = {"xgboost": expit(logaritmo_razon(previstas["ultimo"]) + perfil)}  # con la tasa del último año

    # Qué variables mueven la predicción: media del valor absoluto de su aportación (SHAP), con los mismos
    # 100.000 préstamos que 3_modelos.py. Se pasa el nivel previsto con la tasa del último año; solo cambia
    # la columna del sesgo, que no se guarda.
    unos_cuantos = prueba[VARIABLES].sample(min(len(prueba), 100_000), random_state=0)
    nivel_previsto = np.full(len(unos_cuantos), logaritmo_razon(previstas["ultimo"]))
    matriz = xgb.DMatrix(unos_cuantos, enable_categorical=True, base_margin=nivel_previsto)
    aportaciones = modelo.get_booster().predict(matriz, pred_contribs=True)
    importancia.append(pd.DataFrame({"anio": anio, "variable": VARIABLES,
                                     "importancia": np.abs(aportaciones[:, :-1]).mean(axis=0)}))

    # Regresión logística: el nivel es una columna más, con el mismo valor para todo el año de prueba.
    modelo = logit().fit(entreno[VARIABLES + ["nivel"]], y, logisticregression__sample_weight=pesos)
    joblib.dump(modelo, MODELOS / f"logit_{anio}.joblib")
    fila = {"anio": anio, "modelo": "logit", "tasa_real": y_prueba.mean()}
    for nombre, prevista in previstas.items():
        p = modelo.predict_proba(prueba[VARIABLES].assign(nivel=np.float32(logaritmo_razon(prevista))))[:, 1]
        fila |= {"tasa_" + nombre: prevista, "calibracion_" + nombre: p.mean() / y_prueba.mean()}
        if nombre == "ultimo":
            probabilidades["logit"] = p
    fila |= medir(y_prueba, p)  # el orden es el mismo con las tres previsiones
    filas.append(fila)
    umbrales += listas(anio, "logit", y_prueba, p)

    # Como en 3_modelos.py, con la probabilidad de cada modelo con el nivel previsto con la tasa del último año.
    al_corriente = (prueba["mora_actual"].fillna(0) == 0).to_numpy()
    for nombre, p in probabilidades.items():
        # Dónde acierta cada modelo: por separado en los préstamos al corriente y en los que ya van con retraso.
        for segmento, cuales in [("al corriente", al_corriente), ("con retraso", ~al_corriente)]:
            segmentos.append({"anio": anio, "modelo": nombre, "segmento": segmento, "prestamos": int(cuales.sum()),
                              "tasa": y_prueba[cuales].mean(), "auc": roc_auc_score(y_prueba[cuales], p[cuales])})
        # Calibración: en diez grupos de riesgo creciente, lo predicho frente a lo que pasó.
        for grupo, cuales in enumerate(np.array_split(np.argsort(p), 10), 1):
            calibracion.append({"anio": anio, "modelo": nombre, "grupo": grupo, "predicho": p[cuales].mean(),
                                "observado": y_prueba[cuales].mean()})

    for fila in filas[-2:]:
        print(f"{anio} {fila['modelo']:8} AUC {fila['auc']:.4f}  captura {fila['captura']:.3f}  calibración: "
              f"último {fila['calibracion_ultimo']:.2f}, suavizado {fila['calibracion_suavizado']:.2f}, "
              f"mediana {fila['calibracion_mediana']:.2f}", flush=True)
    print(f"{anio}: {time.time() - inicio:.0f} s", flush=True)
    pd.DataFrame(filas).to_csv(RESULTADOS / f"nivel{SUFIJO}.csv", index=False)
    pd.DataFrame(umbrales).to_csv(RESULTADOS / f"umbrales_nivel{SUFIJO}.csv", index=False)
    pd.concat(importancia).to_csv(RESULTADOS / f"importancia_nivel{SUFIJO}.csv", index=False)
    pd.DataFrame(calibracion).to_csv(RESULTADOS / f"calibracion_nivel{SUFIJO}.csv", index=False)
    pd.DataFrame(segmentos).to_csv(RESULTADOS / f"segmentos_nivel{SUFIJO}.csv", index=False)

# Los modelos de 2025 pasan a la aplicación, en lugar de los que hubiera.
if not PILOTO and max(TODOS) in ANIOS:
    APLICACION.mkdir(exist_ok=True)
    for fichero in [f"logit_{max(TODOS)}.joblib", f"xgboost_{max(TODOS)}.joblib", "categorias.joblib"]:
        shutil.copy2(MODELOS / fichero, APLICACION / fichero)
    print(f"Modelos de {max(TODOS)} copiados a {APLICACION}", flush=True)
