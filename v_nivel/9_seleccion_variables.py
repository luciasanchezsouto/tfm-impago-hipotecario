"""Selección de variables: ¿cuánto se pierde si los modelos con nivel usan solo las variables de más peso?

Los modelos son los de 6_nivel.py: XGBoost y regresión logística con el nivel de impago del año (su tasa,
como log(tasa / (1 - tasa))). XGBoost recibe el nivel como punto de partida de cada fila; la logística, como
una columna más, «nivel», que entra siempre además de las variables elegidas. El nivel no es una variable
del préstamo: no se cuenta entre ellas ni entra en el orden. La muestra, los pesos y los parámetros son los
de 6_nivel.py.

Para cada año de prueba se entrena el XGBoost con nivel con las 22 variables y se ordenan las variables por
su valor SHAP (cuánto mueve cada una la predicción). Después se vuelve a entrenar el XGBoost con las 10, 5 y
3 primeras, y la logística con las 22, 10 y 5. El orden sale solo de años anteriores al de prueba: del
modelo entrenado con ellos, medido en préstamos del último año de entrenamiento. Con 22 variables los dos
modelos son los de 6_nivel.py y el resultado coincide con nivel.csv, y así se comprueba el script.

AUC y captura (parte de los impagos en el 10 % con más riesgo) se miden como en 6_nivel.py. Para la
calibración (tasa prevista entre tasa real) el nivel del año de prueba se prevé con la tasa del año
anterior, la previsión «último».

Lee data/sencillo/panel y descriptivo/por_anio.csv. Escribe resultados/seleccion.csv (anio, k, modelo, auc,
captura, calibracion, variables), y lo reescribe al terminar cada año: si se corta, quedan los años hechos.
Tarda unos 30 minutos con tarjeta gráfica (de 2 minutos el año 2008 a 4,5 el 2025); el piloto, 3.

    python v_nivel/9_seleccion_variables.py            2008, 2011, 2014, 2017, 2019, 2020, 2022 y 2025
    python v_nivel/9_seleccion_variables.py piloto     2010, 2016 y 2025, con la tabla piloto
    python v_nivel/9_seleccion_variables.py 2008 2019  solo esos años (para probar; pisa seleccion.csv)
"""

import sys
import time
from pathlib import Path

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

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
PILOTO = "piloto" in sys.argv
PANEL = RAIZ / "data" / "sencillo" / ("panel_piloto" if PILOTO else "panel") / "*.parquet"
RESULTADOS = AQUI / ("resultados_piloto" if PILOTO else "resultados")
ELEGIDOS = [int(a) for a in sys.argv[1:] if a.isdigit()]
ANIOS = ELEGIDOS or ([2010, 2016, 2025] if PILOTO else [2008, 2011, 2014, 2017, 2019, 2020, 2022, 2025])
NUMERICAS = ["fico", "ltv", "dti", "importe", "tipo_inicial", "plazo", "seguro", "unidades", "prestatarios",
             "antiguedad", "saldo", "ltv_actual", "mora_actual", "mora_maxima_12m", "meses_en_mora_12m",
             "modificado"]
CATEGORICAS = ["ocupacion", "finalidad", "primer_comprador", "canal", "inmueble", "estado"]
VARIABLES = NUMERICAS + CATEGORICAS
TAMANOS = {"xgboost": [22, 10, 5, 3], "logit": [22, 10, 5]}  # el XGBoost de 22 primero: da el orden


def preparar(tabla, categorias):
    """Pasa la tabla a pandas, con las variables de texto como categorías con los mismos niveles siempre."""
    datos = tabla.select("anio", "impago", *VARIABLES).to_pandas()
    datos[NUMERICAS] = datos[NUMERICAS].astype("float32")
    for variable in CATEGORICAS:
        datos[variable] = pd.Categorical(datos[variable].fillna("sin dato"), categories=categorias[variable])
    return datos


def logit(columnas):
    """La logística de 6_nivel.py con las columnas dadas; el nivel entra siempre, como un número más."""
    numeros = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    partes = [("numeros", numeros, [c for c in columnas if c in NUMERICAS] + ["nivel"]),
              ("categorias", OneHotEncoder(handle_unknown="ignore"), [c for c in columnas if c in CATEGORICAS])]
    return make_pipeline(ColumnTransformer([p for p in partes if p[2]]), LogisticRegression(max_iter=500))


def xgboost():
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                             enable_categorical=True, device="cuda", random_state=0)


# El nivel de cada año: la tasa de impago de toda la cartera, que calcula 2_descriptivo.py.
tasa = pd.read_csv(AQUI / "descriptivo" / "por_anio.csv").set_index("anio")["tasa"]

# La misma muestra que 6_nivel.py (todos los impagos y 1 de cada 10 del resto), y las mismas categorías.
panel = pl.scan_parquet(PANEL)
en_muestra = (pl.col("impago") == 1) | (pl.col("id").hash(seed=0) % 10 == 0)
muestra = panel.filter(en_muestra, pl.col("anio") < 2025).collect()
categorias = {v: sorted(muestra[v].drop_nulls().unique()) + ["sin dato"] for v in CATEGORICAS}
muestra = preparar(muestra, categorias)
muestra["nivel"] = logaritmo_razon(muestra["anio"].map(tasa)).astype("float32")  # el nivel de su año, ya conocido

RESULTADOS.mkdir(parents=True, exist_ok=True)
filas = []
for anio in ANIOS:
    inicio = time.time()
    entreno = muestra[muestra["anio"] < anio]
    prueba = preparar(panel.filter(pl.col("anio") == anio).collect(), categorias)
    y, y_prueba = entreno["impago"], prueba["impago"].to_numpy()
    pesos = np.where(y == 1, 1, 10)
    previsto = logaritmo_razon(tasa[tasa.index < anio].iloc[-1])  # nivel previsto: la tasa del año anterior
    for nombre, tamanos in TAMANOS.items():
        for k in tamanos:
            # Las columnas, en el orden de siempre: con 22 el modelo es idéntico al de 6_nivel.py.
            columnas = [v for v in VARIABLES if k == 22 or v in orden[:k]]
            if nombre == "logit":
                modelo = logit(columnas).fit(entreno[columnas + ["nivel"]], y,
                                             logisticregression__sample_weight=pesos)
                p = modelo.predict_proba(prueba[columnas].assign(nivel=np.float32(previsto)))[:, 1]
                puntuacion, calibracion = p, p.mean() / y_prueba.mean()
            else:
                modelo = xgboost().fit(entreno[columnas], y, sample_weight=pesos,
                                       base_margin=entreno["nivel"]).set_params(device="cpu")
                # Se ordena por el perfil (lo que los árboles suman o restan al nivel), como en 6_nivel.py.
                perfil = modelo.predict(prueba[columnas], base_margin=np.zeros(len(prueba)), output_margin=True)
                puntuacion, calibracion = perfil, expit(previsto + perfil).mean() / y_prueba.mean()
            if nombre == "xgboost" and k == 22:
                # Orden sin mirar el año de prueba: SHAP medio en 100.000 préstamos del último año de
                # entrenamiento, ponderado para deshacer el muestreo de los que no impagan. Cada préstamo lleva
                # el nivel de su año, que solo mueve la columna del sesgo; esa columna no entra en el orden.
                filtro = (entreno["anio"] == anio - 1).to_numpy()
                ultimo = entreno[filtro].assign(peso=pesos[filtro])
                ultimo = ultimo.sample(min(len(ultimo), 100_000), random_state=0)
                matriz = xgb.DMatrix(ultimo[VARIABLES], enable_categorical=True, base_margin=ultimo["nivel"])
                shap = modelo.get_booster().predict(matriz, pred_contribs=True)
                importancia = np.average(np.abs(shap[:, :-1]), axis=0, weights=ultimo["peso"])
                orden = list(pd.Series(importancia, index=VARIABLES).sort_values(ascending=False).index)
            n = len(puntuacion)
            mas_riesgo = np.lexsort((np.random.default_rng(0).random(n), puntuacion))[-n // 10 :]
            filas.append({"anio": anio, "k": k, "modelo": nombre, "auc": roc_auc_score(y_prueba, puntuacion),
                          "captura": y_prueba[mas_riesgo].sum() / y_prueba.sum(),
                          "calibracion": calibracion, "variables": " ".join(orden[:k])})
            print(f"{anio} {nombre:8} k={k:2}  AUC {filas[-1]['auc']:.4f}  captura {filas[-1]['captura']:.3f}  "
                  f"calibración {filas[-1]['calibracion']:.2f}  {round(time.time() - inicio)} s", flush=True)
    pd.DataFrame(filas).to_csv(RESULTADOS / "seleccion.csv", index=False)
