"""Comprobación: ¿pierde XGBoost su ventaja después de 2020 porque aprende de ese año?

XGBoost sin nivel ordena mejor que la regresión logística en 14 de los 18 años de prueba, pero después de 2020
pierde esa ventaja: de 2021 a 2025 solo gana en dos años y su AUC medio queda por debajo. Este script repite
esos cinco años de prueba con los dos modelos sin nivel de 3_modelos.py, quitando 2020 del entrenamiento.
Si XGBoost recupera la ventaja, es que entrenar con ese año lo perjudicaba. Es solo una comprobación, para
entender qué pasa después de 2020: el modelo final del trabajo sigue siendo XGBoost con nivel, de 6_nivel.py.

Lo demás es igual que en 3_modelos.py: la misma tabla, la misma muestra, los mismos pesos, las mismas
variables y los mismos parámetros, y se predice el año de prueba entero. De 2008 a 2020 no hay nada que
repetir, porque 2020 no entra en el entrenamiento de ninguno de esos años.

Lee data/sencillo/panel y, para el resumen, resultados/metricas.csv (3_modelos.py) y resultados/nivel.csv
(6_nivel.py). Escribe:

    resultados/sin_2020.csv         por año de prueba (2021 a 2025) y modelo: AUC, captura al 10 % y
                                    calibración (impagos previstos entre reales)
    resultados/sin_2020_comparacion.csv
                                    para la regresión logística y XGBoost, tres formas de entrenar: con todos
                                    los años (metricas.csv), sin 2020 (de 2008 a 2020, metricas.csv; de 2021 a
                                    2025, sin_2020.csv) y con nivel (nivel.csv, con la tasa del último año).
                                    AUC y captura medios de 2008 a 2025 y de 2021 a 2025; desvío mediano de la
                                    calibración c de 2009 a 2025 y de 2021 a 2025, mediana de max(c, 1/c) - 1;
                                    años de 2009 a 2025 con c entre 0,8 y 1,25, y años de 2008 a 2025 en que
                                    XGBoost tiene más AUC que la regresión logística entrenada de la misma forma
    data/sencillo/modelos_sin_2020  los dos modelos de cada año y categorias.joblib. No son los de la
                                    aplicación y no se copian a v_nivel/modelos

Tarda unos 10 minutos con tarjeta gráfica; el resumen solo, un segundo.

    python v_nivel/11_sin_2020.py           entrena de 2021 a 2025 y escribe las dos tablas
    python v_nivel/11_sin_2020.py resumen   no entrena: rehace sin_2020_comparacion.csv con las tres tablas de
                                            resultados/, así que no necesita los datos
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
for argumento in sys.argv[1:]:
    if argumento != "resumen":
        sys.exit(f"Argumento no válido: {argumento}. Solo se admite «resumen».")
RESUMEN = "resumen" in sys.argv[1:]
PANEL = RAIZ / "data" / "sencillo" / "panel" / "*.parquet"
MODELOS = RAIZ / "data" / "sencillo" / "modelos_sin_2020"  # nunca v_nivel/modelos, que son los de la aplicación
RESULTADOS = AQUI / "resultados"
TODOS = range(2008, 2026)
ANIOS = range(2021, 2026)  # los años de prueba en cuyo entrenamiento entraría 2020
QUITADO = 2020

NUMERICAS = ["fico", "ltv", "dti", "importe", "tipo_inicial", "plazo", "seguro", "unidades", "prestatarios",
             "antiguedad", "saldo", "ltv_actual", "mora_actual", "mora_maxima_12m", "meses_en_mora_12m",
             "modificado"]
CATEGORICAS = ["ocupacion", "finalidad", "primer_comprador", "canal", "inmueble", "estado"]
VARIABLES = NUMERICAS + CATEGORICAS
UNO_DE_CADA = 10


def preparar(tabla, categorias):
    """Pasa la tabla a pandas, con las variables de texto como categorías con los mismos niveles siempre."""
    datos = tabla.select("anio", "impago", *VARIABLES).to_pandas()
    datos[NUMERICAS] = datos[NUMERICAS].astype("float32")
    for variable in CATEGORICAS:
        datos[variable] = pd.Categorical(datos[variable].fillna("sin dato"), categories=categorias[variable])
    return datos


def logit():
    """La regresión logística de 3_modelos.py."""
    numeros = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    columnas = ColumnTransformer([("numeros", numeros, NUMERICAS),
                                  ("categorias", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
    return make_pipeline(columnas, LogisticRegression(max_iter=500))


def xgboost():
    """El XGBoost de 3_modelos.py. Se entrena en la tarjeta gráfica; sin tarjeta, device="cpu"."""
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                             enable_categorical=True, device="cuda", random_state=0)


def medir(y, p):
    """AUC, captura y calibración, como en 3_modelos.py; los empates se deshacen al azar."""
    orden = np.lexsort((np.random.default_rng(0).random(len(p)), p))
    mas_riesgo = orden[-len(p) // 10 :]  # el 10 % de préstamos con más riesgo
    return {"auc": roc_auc_score(y, p), "captura": y[mas_riesgo].sum() / y.sum(), "calibracion": p.mean() / y.mean()}


def entrenar():
    """Entrena los dos modelos de cada año de prueba sin 2020 y escribe resultados/sin_2020.csv."""
    # La misma muestra que 3_modelos.py: todos los impagos y 1 de cada 10 del resto, de los años antes de 2025.
    panel = pl.scan_parquet(PANEL)
    en_muestra = (pl.col("impago") == 1) | (pl.col("id").hash(seed=0) % UNO_DE_CADA == 0)
    muestra = panel.filter(en_muestra, pl.col("anio") < max(ANIOS)).collect()
    categorias = {v: sorted(muestra[v].drop_nulls().unique()) + ["sin dato"] for v in CATEGORICAS}
    muestra = preparar(muestra, categorias)

    MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(categorias, MODELOS / "categorias.joblib")
    filas = []
    for anio in ANIOS:
        entreno = muestra[(muestra["anio"] < anio) & (muestra["anio"] != QUITADO)]
        prueba = preparar(panel.filter(pl.col("anio") == anio).collect(), categorias)
        y, y_prueba = entreno["impago"], prueba["impago"].to_numpy()
        pesos = np.where(y == 1, 1, UNO_DE_CADA)
        modelos = {"logit": logit().fit(entreno[VARIABLES], y, logisticregression__sample_weight=pesos),
                   "xgboost": xgboost().fit(entreno[VARIABLES], y, sample_weight=pesos).set_params(device="cpu")}
        for nombre, modelo in modelos.items():
            joblib.dump(modelo, MODELOS / f"{nombre}_{anio}.joblib")
            p = modelo.predict_proba(prueba[VARIABLES])[:, 1]
            filas.append({"anio": anio, "modelo": nombre, **medir(y_prueba, p)})
            print(f"{anio} {nombre:8} AUC {filas[-1]['auc']:.4f}  captura {filas[-1]['captura']:.3f}  "
                  f"calibración {filas[-1]['calibracion']:.2f}", flush=True)
    pd.DataFrame(filas).to_csv(RESULTADOS / "sin_2020.csv", index=False)


def resumir():
    """Compara las tres formas de entrenar y escribe resultados/sin_2020_comparacion.csv."""
    todos = pd.read_csv(RESULTADOS / "metricas.csv")
    sin_2020 = pd.read_csv(RESULTADOS / "sin_2020.csv")
    # Hasta 2020, entrenar sin 2020 es lo mismo que entrenar con todos los años anteriores.
    sin_2020 = pd.concat([todos[todos["anio"] <= QUITADO], sin_2020])
    nivel = pd.read_csv(RESULTADOS / "nivel.csv").rename(columns={"calibracion_ultimo": "calibracion"})
    filas = []
    for modelo in ["logit", "xgboost"]:
        for entrenamiento, tabla in [("todos los años", todos), ("sin 2020", sin_2020), ("con nivel", nivel)]:
            anual = {m: tabla[tabla["modelo"] == m].set_index("anio").sort_index() for m in ["logit", "xgboost"]}
            if any(list(a.index) != list(TODOS) for a in anual.values()):
                sys.exit(f"A la tabla «{entrenamiento}» le falta algún año de 2008 a 2025.")
            a = anual[modelo]
            c = a.loc[2009:, "calibracion"]  # la calibración se mide de 2009 a 2025, 17 años
            desvio = np.maximum(c, 1 / c) - 1  # cuánto se aleja de 1 en cualquier sentido
            gana = (anual["xgboost"]["auc"] > anual["logit"]["auc"]).sum()
            filas.append({"modelo": modelo, "entrenamiento": entrenamiento,
                          "auc_2008_2025": a["auc"].mean(), "auc_2021_2025": a.loc[2021:, "auc"].mean(),
                          "captura_2008_2025": a["captura"].mean(), "captura_2021_2025": a.loc[2021:, "captura"].mean(),
                          "desvio_mediano_2009_2025": desvio.median(),
                          "desvio_mediano_2021_2025": desvio.loc[2021:].median(),
                          "anios_bien_calibrados": int(c.between(0.8, 1.25).sum()),
                          "anios_gana_al_logit": gana if modelo == "xgboost" else None})
    tabla = pd.DataFrame(filas).astype({"anios_gana_al_logit": "Int64"})  # vacío en las filas de la logística
    tabla.to_csv(RESULTADOS / "sin_2020_comparacion.csv", index=False)
    print(tabla.to_string(index=False))


if not RESUMEN:
    entrenar()
resumir()
