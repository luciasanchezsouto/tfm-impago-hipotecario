"""¿De qué modelo fiarse cuando la regresión logística y el XGBoost discrepan?

Vuelve a puntuar cada año de prueba, de 2008 a 2025, con los dos modelos con nivel entrenados antes de ese
año, los que 6_nivel.py guarda en data/sencillo/modelos_nivel. El nivel previsto para el año es la tasa de
impago del año anterior (descriptivo/por_anio.csv), como en el resto del trabajo. Separa los préstamos en que
una probabilidad es al menos el triple de la otra y alguna llega al 5 %, y los que tienen algún valor muy poco
habitual, y compara en cada grupo la tasa de impago real con la que preveía cada modelo.

Escribe resultados/discrepancias.csv (una fila por año y grupo) y resultados/rangos.csv (rango habitual de
cada variable, que usa la aplicación para avisar). No entrena nada ni usa la tarjeta gráfica; tarda unos
10 minutos, medio minuto por año.

    python v_nivel/10_discrepancias.py           de 2008 a 2025
    python v_nivel/10_discrepancias.py 2025      solo los años indicados (uno o varios); el resultado va a
                                                 resultados/discrepancias_parcial.csv y no se escribe rangos.csv
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import logit as logaritmo_razon
from sklearn.metrics import roc_auc_score

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
TODOS = range(2008, 2026)
for argumento in sys.argv[1:]:
    if not (argumento.isdigit() and int(argumento) in TODOS):
        sys.exit(f"Argumento no válido: {argumento}. Se admiten años de 2008 a 2025.")
SUELTOS = sorted({int(a) for a in sys.argv[1:]})
ANIOS = SUELTOS or TODOS
PANEL = RAIZ / "data" / "sencillo" / "panel" / "*.parquet"
MODELOS = RAIZ / "data" / "sencillo" / "modelos_nivel"
RESULTADOS = AQUI / "resultados"
SUFIJO = "_parcial" if SUELTOS else ""  # con años sueltos no se pisa el resultado de los 18 años
NUMERICAS = ["fico", "ltv", "dti", "importe", "tipo_inicial", "plazo", "seguro", "unidades", "prestatarios",
             "antiguedad", "saldo", "ltv_actual", "mora_actual", "mora_maxima_12m", "meses_en_mora_12m",
             "modificado"]
CATEGORICAS = ["ocupacion", "finalidad", "primer_comprador", "canal", "inmueble", "estado"]
VARIABLES = NUMERICAS + CATEGORICAS
CONTINUAS = ["fico", "ltv", "dti", "importe", "tipo_inicial", "antiguedad", "saldo", "ltv_actual"]
VECES, MINIMO = 3, 0.05  # discrepan si una probabilidad es el triple de la otra y alguna llega al 5 %

panel = pl.scan_parquet(PANEL)
uno_de_cien = pl.col("id").hash(seed=0) % 100 == 0  # para los rangos basta una muestra de préstamos
tasa = pd.read_csv(AQUI / "descriptivo" / "por_anio.csv").set_index("anio")["tasa"]
categorias = joblib.load(MODELOS / "categorias.joblib")


def rango(hasta):
    """Entre qué valores está el 99,8 % de los préstamos de los años anteriores, variable a variable."""
    antes = panel.filter(pl.col("anio") < hasta, uno_de_cien).select(CONTINUAS).collect().to_pandas()
    return antes.quantile(0.001), antes.quantile(0.999)


filas = []
for anio in ANIOS:
    inicio = time.time()
    datos = panel.filter(pl.col("anio") == anio).select("impago", *VARIABLES).collect().to_pandas()
    datos[NUMERICAS] = datos[NUMERICAS].astype("float32")
    for variable in CATEGORICAS:
        datos[variable] = pd.Categorical(datos[variable].fillna("sin dato"), categories=categorias[variable])
    y = datos["impago"].to_numpy()
    # El nivel previsto: la tasa del año anterior, en la escala del modelo, z = log(tasa / (1 - tasa)).
    z = logaritmo_razon(tasa[tasa.index < anio].iloc[-1])
    logit = joblib.load(MODELOS / f"logit_{anio}.joblib")
    xgboost = joblib.load(MODELOS / f"xgboost_{anio}.joblib").set_params(device="cpu")
    p = {"logit": logit.predict_proba(datos[VARIABLES].assign(nivel=np.float32(z)))[:, 1],
         "xgboost": xgboost.predict_proba(datos[VARIABLES], base_margin=np.full(len(datos), z))[:, 1]}
    logit, xgboost = p["logit"], p["xgboost"]

    minimo, maximo = rango(anio)
    extremo = np.zeros(len(y), bool)
    for variable in CONTINUAS:
        extremo |= ((datos[variable] < minimo[variable]) | (datos[variable] > maximo[variable])).to_numpy()
    alguna = np.maximum(logit, xgboost) >= MINIMO
    xgboost_mas, logit_mas = alguna & (xgboost >= VECES * logit), alguna & (logit >= VECES * xgboost)
    grupos = {"todos": np.ones(len(y), bool), "valor poco habitual": extremo, "xgboost da más riesgo": xgboost_mas,
              "logit da más riesgo": logit_mas, "coinciden": alguna & ~xgboost_mas & ~logit_mas}
    for grupo, filtro in grupos.items():
        fila = {"anio": anio, "grupo": grupo, "prestamos": int(filtro.sum()), "impagos": int(y[filtro].sum()),
                "tasa_real": y[filtro].mean(), "prevista_logit": logit[filtro].mean(),
                "prevista_xgboost": xgboost[filtro].mean()}
        fila |= {f"auc_{m}": roc_auc_score(y[filtro], q[filtro]) for m, q in p.items()}
        filas.append(fila)
    print(f"{anio} hecho en {time.time() - inicio:.0f} s", flush=True)

pd.DataFrame(filas).to_csv(RESULTADOS / f"discrepancias{SUFIJO}.csv", index=False)
if not SUELTOS:
    # Rango habitual con los años que conoce el modelo de la aplicación (hasta 2024).
    minimo, maximo = rango(2025)
    pd.DataFrame({"minimo": minimo, "maximo": maximo}).rename_axis("variable").round(3).to_csv(RESULTADOS / "rangos.csv")
