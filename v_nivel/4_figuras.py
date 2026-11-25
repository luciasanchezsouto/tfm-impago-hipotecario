"""Las figuras de resultados, a partir de las tablas que escriben 3_modelos.py y 6_nivel.py.

Escribe en resultados/ fig5_acierto.png, fig6_segmentos.png, fig_calibracion_nivel.png, fig_segmentos_nivel.png,
fig_importancia_nivel.png, fig_nivel.png, recalibracion.csv, intervalos.csv, intervalos_bloques.csv,
calibracion_pendiente.csv, frecuencia_variables.csv y aviso_anomalo.csv. Lee metricas.csv y segmentos.csv de
3_modelos.py; nivel.csv con los 18 años de prueba, importancia_nivel.csv, calibracion_nivel.csv y segmentos_nivel.csv
de 6_nivel.py, y descriptivo/por_anio.csv, con la tasa de impago de cada año, de 2_descriptivo.py. Si ya existe
seleccion.csv, de 9_seleccion_variables.py, la usa también en frecuencia_variables.csv.

    python v_nivel/4_figuras.py
"""

import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter, ScalarFormatter

RESULTADOS = Path(__file__).resolve().parent / "resultados"
COLOR = {"regla": "#8C8C8C", "logit": "#0072BD", "xgboost": "#D95319"}
TRAZO = {"regla": ":", "logit": "-", "xgboost": "--"}  # para distinguirlos también en blanco y negro
NOMBRE = {"regla": "Baseline", "logit": "Regresión logística", "xgboost": "XGBoost"}
NOMBRE_NIVEL = {"regla": "Baseline", "logit": "Regresión logística con nivel", "xgboost": "XGBoost con nivel"}
VARIABLE = {"fico": "FICO", "tipo_inicial": "Tipo de interés", "ltv_actual": "LTV actual",
            "mora_maxima_12m": "Retraso máximo en 12 meses", "antiguedad": "Antigüedad",
            "prestatarios": "Número de prestatarios", "dti": "Deuda sobre ingresos (DTI)", "estado": "Estado",
            "plazo": "Plazo", "finalidad": "Finalidad", "saldo": "Saldo pendiente", "importe": "Importe concedido",
            "ltv": "LTV al conceder", "mora_actual": "Meses de retraso hoy", "canal": "Canal",
            "meses_en_mora_12m": "Meses con retraso en 12 meses", "seguro": "Seguro hipotecario",
            "ocupacion": "Ocupación", "inmueble": "Tipo de inmueble", "unidades": "Viviendas",
            "primer_comprador": "Primera vivienda", "modificado": "Préstamo modificado"}
# Mismo estilo en todas las figuras: Arial como el texto de la memoria y tamaño de figura igual al impreso
# (ancho de texto 16 cm = 6,3 pulgadas), de modo que la letra sale a 9-10 puntos en el papel.
plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "dejavusans", "font.size": 9,
                     "axes.titlesize": 10, "lines.linewidth": 1.5, "lines.markersize": 3.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E0E0E0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.constrained_layout.use": True})


def guardar(figura, fichero):
    """Guarda la figura con coma decimal en los ejes numéricos (no toca las etiquetas puestas a mano)."""
    for eje in [e for ax in figura.axes for e in (ax.xaxis, ax.yaxis)]:
        if isinstance(eje.get_major_formatter(), (ScalarFormatter, LogFormatter)):
            eje.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    figura.savefig(fichero, dpi=300)

metricas = pd.read_csv(RESULTADOS / "metricas.csv")
segmentos = pd.read_csv(RESULTADOS / "segmentos.csv")
nivel = pd.read_csv(RESULTADOS / "nivel.csv")  # los modelos con nivel, de 6_nivel.py
importancia_nivel = pd.read_csv(RESULTADOS / "importancia_nivel.csv")  # la del XGBoost con nivel, de 6_nivel.py
calibracion_nivel = pd.read_csv(RESULTADOS / "calibracion_nivel.csv")  # de 6_nivel.py, con el nivel del último año
segmentos_nivel = pd.read_csv(RESULTADOS / "segmentos_nivel.csv")  # ídem
tasa = pd.read_csv(RESULTADOS.parent / "descriptivo" / "por_anio.csv").set_index("anio")["tasa"]  # de 2_descriptivo.py
if nivel["anio"].nunique() < 18:
    sys.exit("resultados/nivel.csv aún no tiene los 18 años de prueba: falta terminar 6_nivel.py.")
# Para las figuras con los modelos con nivel: el baseline de 3_modelos.py, que no cambia, y la calibración con la
# tasa del último año como nivel previsto.
con_nivel = pd.concat([metricas[metricas["modelo"] == "regla"],
                       nivel.rename(columns={"calibracion_ultimo": "calibracion"})])
segmentos_con_nivel = pd.concat([segmentos[segmentos["modelo"] == "regla"], segmentos_nivel])


def lineas(eje, tabla, columna, nombre=NOMBRE):
    """Una línea por modelo con el valor de la columna en cada año."""
    for modelo, color in COLOR.items():
        del_modelo = tabla[tabla["modelo"] == modelo]
        eje.plot(del_modelo["anio"], del_modelo[columna], color=color, linestyle=TRAZO[modelo], marker="o",
                 label=nombre[modelo])
    eje.set_xticks(range(2008, 2026, 4))


def leyenda(figura, eje):
    """Una sola leyenda para toda la figura, encima de los paneles."""
    figura.legend(*eje.get_legend_handles_labels(), loc="outside upper center", ncol=3)


# %% fig5_acierto.png: cuánto acierta cada modelo sin nivel, año a año
figura, (izquierda, derecha) = plt.subplots(1, 2, figsize=(5.7, 2.6))
lineas(izquierda, metricas, "auc")
izquierda.set(title="AUC", ylim=(0.5, 0.95))
leyenda(figura, izquierda)
metricas["captura_pct"] = metricas["captura"] * 100
lineas(derecha, metricas, "captura_pct")
derecha.set(title="Recall al 10 %", ylabel="% de los impagos", ylim=(0, 80))
guardar(figura, RESULTADOS / "fig5_acierto.png")

# %% fig_calibracion_nivel.png: si la probabilidad de los modelos con nivel es fiable. A la izquierda, impagos
# predichos entre impagos reales de cada año (1 = acierta la tasa); a la derecha, los diez grupos de riesgo del último.
figura, (izquierda, derecha) = plt.subplots(1, 2, figsize=(5.7, 2.7))
lineas(izquierda, con_nivel, "calibracion", NOMBRE_NIVEL)
izquierda.axhline(1, color="k", linewidth=0.8)
izquierda.set(title="Impagos predichos entre reales", yscale="log", yticks=[0.1, 0.25, 0.5, 1, 2, 5])
leyenda(figura, izquierda)
ultimo = calibracion_nivel[calibracion_nivel["anio"] == calibracion_nivel["anio"].max()]
for modelo in ["logit", "xgboost"]:
    del_modelo = ultimo[ultimo["modelo"] == modelo]
    derecha.plot(del_modelo["predicho"] * 100, del_modelo["observado"] * 100, color=COLOR[modelo],
                 linestyle=TRAZO[modelo], marker="o")
derecha.plot([0.01, 10], [0.01, 10], color="k", linewidth=0.8)
derecha.set(title=f"Diez grupos de riesgo en {ultimo['anio'].max()}", xlabel="Impago predicho, %",
            ylabel="Impago real, %", xscale="log", yscale="log", xticks=[0.01, 0.1, 1, 10],
            yticks=[0.01, 0.1, 1, 10])
for eje in figura.axes:
    eje.minorticks_off()
guardar(figura, RESULTADOS / "fig_calibracion_nivel.png")

# %% fig6_segmentos.png: dónde gana el xgboost. AUC por separado en los préstamos al corriente y con retraso
# La segunda vuelta, fig_segmentos_nivel.png, con los modelos con nivel.
for tabla, nombre, fichero in [(segmentos, NOMBRE, "fig6_segmentos.png"),
                               (segmentos_con_nivel, NOMBRE_NIVEL, "fig_segmentos_nivel.png")]:
    figura, ejes = plt.subplots(1, 2, figsize=(5.7, 2.6), sharey=True)
    for eje, segmento in zip(ejes, ["al corriente", "con retraso"]):
        lineas(eje, tabla[tabla["segmento"] == segmento], "auc", nombre)
        eje.set(title=f"Préstamos {segmento}")
    ejes[0].set_ylabel("AUC")
    leyenda(figura, ejes[0])
    guardar(figura, RESULTADOS / fichero)

# %% fig_importancia_nivel.png: las doce variables que más mueven la predicción del modelo final, el XGBoost con
# nivel, en la media de todos los años. Para tres cuartos del ancho de texto.
media = importancia_nivel.groupby("variable")["importancia"].mean().sort_values().tail(12)
figura, eje = plt.subplots(figsize=(5.2, 3.4))
eje.barh(media.index.map(VARIABLE), media, color=COLOR["xgboost"])
eje.set(xlabel="Valor SHAP absoluto medio")
eje.grid(axis="y", visible=False)
eje.grid(axis="x")
guardar(figura, RESULTADOS / "fig_importancia_nivel.png")

# %% fig_nivel.png: ventaja de XGBoost sobre la regresión logística (diferencia de AUC) en cada año de prueba,
# con los modelos de 3_modelos.py (sin nivel) y con los de 6_nivel.py (con nivel).
figura, eje = plt.subplots(figsize=(4.4, 2.35))  # 0,7 del ancho de texto y unos 6 cm de alto
for tabla, nombre, color, trazo, marca in [(metricas, "Sin nivel", "#8C8C8C", "-", "o"),
                                           (nivel, "Con nivel", COLOR["xgboost"], "--", "s")]:
    auc = tabla.pivot(index="anio", columns="modelo", values="auc")
    eje.plot(auc.index, auc["xgboost"] - auc["logit"], color=color, linestyle=trazo, marker=marca, label=nombre)
eje.axhline(0, color="k", linewidth=0.8)
eje.set(ylabel="Ventaja de XGBoost en AUC", xticks=range(2008, 2026, 2))
leyenda(figura, eje)
guardar(figura, RESULTADOS / "fig_nivel.png")

# %% Recalibración: dividir la probabilidad por el desvío que tuvo el modelo el año anterior, ya conocido el
# 1 de enero. La calibración recalibrada de un año es la suya entre la del año anterior.
por_anio = metricas.pivot(index="anio", columns="modelo", values="calibracion")[list(COLOR)]
recalibrada = (por_anio / por_anio.shift(1)).dropna()
recalibrada.join(por_anio, lsuffix="_recalibrada").round(3).to_csv(RESULTADOS / "recalibracion.csv")

# %% Intervalos: con millones de préstamos por año, lo que varía es el año. Se remuestrean los 18 años de prueba
# con reemplazo (los mismos para todos los modelos) y se toma el 95 % central de la media. Se cuentan los años que
# gana cada modelo y la prueba de signos: probabilidad, a cara o cruz, de un reparto así en cualquier sentido.
# Los modelos con nivel llevan el sufijo _nivel (logit_nivel, xgboost_nivel).
sorteo = np.random.default_rng(0).integers(0, 18, size=(10000, 18))
filas, diferencias = [], {}
for medida in ["auc", "captura"]:
    tabla = metricas.pivot(index="anio", columns="modelo", values=medida)
    tabla = tabla.join(nivel.pivot(index="anio", columns="modelo", values=medida).add_suffix("_nivel"))
    for nombre, serie in [*tabla.items(), ("logit - regla", tabla["logit"] - tabla["regla"]),
                          ("xgboost - logit", tabla["xgboost"] - tabla["logit"]),
                          ("xgboost_ajustado - xgboost", tabla["xgboost_ajustado"] - tabla["xgboost"]),
                          ("xgboost_nivel - xgboost", tabla["xgboost_nivel"] - tabla["xgboost"]),
                          ("xgboost_nivel - logit_nivel", tabla["xgboost_nivel"] - tabla["logit_nivel"])]:
        medias = serie.to_numpy()[sorteo].mean(axis=1)
        gana = int((serie > 0).sum()) if " - " in nombre else None
        signos = min(1, 2 * sum(comb(18, k) for k in range(max(gana, 18 - gana), 19)) / 2**18) if gana is not None else None
        filas.append([medida, nombre, serie.mean(), *np.percentile(medias, [2.5, 97.5]), gana, signos])
        if gana is not None:
            diferencias[medida, nombre] = serie  # para los intervalos por bloques, más abajo
pd.DataFrame(filas, columns=["medida", "modelo", "media", "ic_inferior", "ic_superior", "anios_gana",
                             "p_signos"]).astype({"anios_gana": "Int64"}).round(4).to_csv(RESULTADOS / "intervalos.csv", index=False)

# %% Intervalos por bloques: los años seguidos se parecen, y el remuestreo de arriba los trata como independientes.
# Aquí se toman bloques de años consecutivos (bootstrap de bloques circulares: cada bloque empieza en un año al azar
# y, pasado 2025, sigue por 2008) hasta juntar 18. Con bloques de un año es el de intervalos.csv y da lo mismo.
sorteos = {}
for bloque in [1, 2, 3]:
    inicios = np.random.default_rng(0).integers(0, 18, size=(10000, 18 // bloque, 1))  # 18, 9 o 6 bloques
    sorteos[bloque] = ((inicios + np.arange(bloque)) % 18).reshape(10000, 18)
filas = []
for (medida, nombre), serie in diferencias.items():
    for bloque, sorteo in sorteos.items():
        medias = serie.to_numpy()[sorteo].mean(axis=1)
        filas.append([medida, nombre, bloque, serie.mean(), *np.percentile(medias, [2.5, 97.5])])
pd.DataFrame(filas, columns=["medida", "modelo", "bloque", "media", "ic_inferior", "ic_superior"]).round(4).to_csv(
    RESULTADOS / "intervalos_bloques.csv", index=False)

# %% Pendiente de calibración: en cada año de prueba, la recta de log(odds real) sobre log(odds predicha) de los diez
# grupos de calibracion_nivel.csv, con la probabilidad que da el nivel previsto con la tasa del último año. Pendiente
# 1 e intercepto 0 es calibración perfecta; pendiente menor que 1, el modelo separa los grupos más que la realidad.
# El intercepto es el valor en log(odds) 0 (50 %), lejos de los grupos; si la pendiente no es 1, no mide solo el nivel.
filas = []
for (anio, modelo), grupos in calibracion_nivel.groupby(["anio", "modelo"]):
    predicho, observado = (np.log(grupos[c] / (1 - grupos[c])) for c in ["predicho", "observado"])
    pendiente, intercepto = np.polyfit(predicho, observado, 1)
    filas.append([anio, modelo + "_nivel", pendiente, intercepto])
pd.DataFrame(filas, columns=["anio", "modelo", "pendiente", "intercepto"]).round(4).to_csv(
    RESULTADOS / "calibracion_pendiente.csv", index=False)

# %% Frecuencia de las variables: en cuántos años está cada una entre las diez y entre las cinco de más peso, y su
# puesto medio. Con el orden de importancia_nivel.csv, en los 18 años de prueba. Ese orden se mide con el modelo
# entrenado con los años anteriores, pero en 100.000 préstamos del propio año de prueba (con lo que se sabía de ellos
# el 1 de enero; el SHAP no usa si impagaron). No es el orden de 9_seleccion_variables.py, que se mide en el último
# año de entrenamiento. Ese otro orden, si existe seleccion.csv, va en las columnas _seleccion, con sus ocho años.
def contar(puestos):
    """Por variable, en cuántos años está entre las diez y entre las cinco primeras, y su puesto medio."""
    return puestos.groupby("variable")["puesto"].agg(anios_top10=lambda p: (p <= 10).sum(),
                                                     anios_top5=lambda p: (p <= 5).sum(), puesto_medio="mean")


puesto = importancia_nivel.groupby("anio")["importancia"].rank(ascending=False, method="first")
frecuencia = contar(importancia_nivel.assign(puesto=puesto))
if (RESULTADOS / "seleccion.csv").exists():
    orden = pd.read_csv(RESULTADOS / "seleccion.csv").query("modelo == 'xgboost' and k == 22")
    puestos = [(anio, variable, i) for anio, variables in zip(orden["anio"], orden["variables"])
               for i, variable in enumerate(variables.split(), 1)]
    frecuencia = frecuencia.join(contar(pd.DataFrame(puestos, columns=["anio", "variable", "puesto"]))
                                 .add_suffix("_seleccion"))
frecuencia.sort_values("puesto_medio").round(2).to_csv(RESULTADOS / "frecuencia_variables.csv")

# %% Aviso de año anómalo como regla automática. La aplicación avisa cuando la tasa del último año cerrado pasa del
# doble de la mediana de los cinco años anteriores a ese (anio_anomalo() de 5_app.py). Aquí se aplica a cada año de
# prueba con las tasas ya conocidas el 1 de enero: si salta, el nivel previsto pasa de la tasa del último año a la
# mediana de los cinco últimos (calibracion_mediana de nivel.csv). Se imprime el desvío mediano de la calibración c,
# max(c, 1/c) - 1, de 2009 a 2025, con y sin la regla.
mediana = {anio: tasa[tasa.index < anio - 1].iloc[-5:].median() for anio in nivel["anio"].unique()}
aviso = nivel[["anio", "modelo", "tasa_ultimo"]].assign(mediana_cinco_anteriores=nivel["anio"].map(mediana))
aviso["aviso"] = aviso["tasa_ultimo"] > 2 * aviso["mediana_cinco_anteriores"]
aviso["calibracion_ultimo"] = nivel["calibracion_ultimo"]
aviso["calibracion_con_aviso"] = nivel["calibracion_mediana"].where(aviso["aviso"], nivel["calibracion_ultimo"])
aviso.to_csv(RESULTADOS / "aviso_anomalo.csv", index=False)
for modelo, del_modelo in aviso[aviso["anio"] >= 2009].groupby("modelo", sort=False):
    sin_regla, con_regla = ((np.maximum(c, 1 / c) - 1).median()
                            for c in [del_modelo["calibracion_ultimo"], del_modelo["calibracion_con_aviso"]])
    print(f"{NOMBRE_NIVEL[modelo]}, desvío mediano de 2009 a 2025: {sin_regla:.1%} con la tasa del último año, "
          f"{con_regla:.1%} con el aviso como regla")
print(f"Figuras escritas en {RESULTADOS}")
