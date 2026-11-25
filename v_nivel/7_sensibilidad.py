"""Sensibilidad de los modelos de 2025: se mueve una variable del préstamo A y se deja el resto fijo.

Lee el préstamo A (primera fila de ejemplo_cartera.csv, inventado) y los modelos con nivel de 2025, que
6_nivel.py deja en v_nivel/modelos/. El nivel previsto es la tasa de impago del último año cerrado (2025, en
descriptivo/por_anio.csv), como para puntuar un préstamo hoy.
Escribe resultados/sensibilidad.csv, resultados/coeficientes.csv y resultados/fig9_sensibilidad.png.

    python v_nivel/7_sensibilidad.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter, ScalarFormatter
from scipy.special import logit as logaritmo_razon

AQUI = Path(__file__).resolve().parent
MODELOS = AQUI / "modelos"
RESULTADOS = AQUI / "resultados"
COLOR = {"logit": "#0072BD", "xgboost": "#D95319"}
TRAZO = {"logit": "-", "xgboost": "--"}  # para distinguirlos también en blanco y negro
NOMBRE = {"logit": "Regresión logística", "xgboost": "XGBoost"}
# El mismo estilo que en 4_figuras.py: Arial como el texto y figura del tamaño impreso (letra de 9-10 puntos).
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

modelos = {m: joblib.load(MODELOS / f"{m}_2025.joblib") for m in COLOR}
categorias = joblib.load(MODELOS / "categorias.joblib")
columnas = [c for c in modelos["logit"].feature_names_in_ if c != "nivel"]  # las 22 variables del préstamo
# El nivel previsto, en la escala del modelo: z = log(tasa / (1 - tasa)) con la tasa del último año.
tasa = pd.read_csv(AQUI / "descriptivo" / "por_anio.csv").set_index("anio")["tasa"]
z = logaritmo_razon(tasa.iloc[-1])


def probabilidad(modelo, datos):
    """Probabilidad de impago con el nivel previsto: columna «nivel» en la logística, punto de partida en XGBoost."""
    if modelo == "logit":
        return modelos["logit"].predict_proba(datos.assign(nivel=np.float32(z)))[:, 1]
    return modelos["xgboost"].predict_proba(datos, base_margin=np.full(len(datos), z))[:, 1]


prestamo_a = pd.read_csv(AQUI / "ejemplo_cartera.csv").iloc[[0]]
prestamo_a["ltv_actual"] = prestamo_a["saldo"] / prestamo_a["importe"] * prestamo_a["ltv"]
amortizado = (prestamo_a["saldo"] / prestamo_a["importe"]).item()  # 0,97: lo que queda por pagar

# Cada barrido devuelve las columnas que cambian. Si una variable arrastra a otras, se mueven juntas:
# - LTV actual: se cambia el LTV al conceder y se mantiene el saldo pendiente (ltv_actual = 0,97 * ltv).
# - Retraso hoy de k meses: k meses seguidos sin pagar, luego el máximo de 12 meses es k y hubo k meses con retraso.
BARRIDOS = {
    "fico": ("FICO", np.arange(600, 821, 10), lambda v: {"fico": v}),
    "ltv_actual": ("LTV actual, %", np.arange(40, 101, 5), lambda v: {"ltv_actual": v, "ltv": v / amortizado}),
    "tipo_inicial": ("Tipo de interés, %", np.arange(3, 8.01, 0.25), lambda v: {"tipo_inicial": v}),
    "mora_actual": ("Meses de retraso el 1 de enero", np.array([0, 1, 2]),
                    lambda v: {"mora_actual": v, "mora_maxima_12m": v, "meses_en_mora_12m": v}),
}

filas = []
for variable, (_, valores, cambios) in BARRIDOS.items():
    datos = pd.concat([prestamo_a.assign(**cambios(v)) for v in valores], ignore_index=True)[columnas]
    for nombre, niveles in categorias.items():
        datos[nombre] = pd.Categorical(datos[nombre], categories=niveles)
    numericas = [c for c in columnas if c not in categorias]
    datos[numericas] = datos[numericas].astype("float32")
    for modelo in modelos:
        filas += [{"variable": variable, "valor": v, "modelo": modelo, "probabilidad": p}
                  for v, p in zip(valores, probabilidad(modelo, datos))]
tabla = pd.DataFrame(filas)
tabla.round(6).to_csv(RESULTADOS / "sensibilidad.csv", index=False)

# %% Coeficientes de la regresión logística: por cuánto se multiplica la razón de probabilidades de impago
# al subir una desviación típica cada variable numérica (las numéricas entran estandarizadas). El nivel del año
# entra como una numérica más y sale con su nombre, «nivel»; su desviación típica está en la escala del modelo,
# log(tasa / (1 - tasa)), no en %.
numeros = modelos["logit"][0].named_transformers_["numeros"]
nombres = numeros.get_feature_names_out()
coeficientes = pd.DataFrame({"variable": nombres, "desviacion_tipica": numeros[-1].scale_,
                             "multiplica_por": np.exp(modelos["logit"][-1].coef_[0][: len(nombres)])})
orden = np.log(coeficientes["multiplica_por"]).abs().sort_values(ascending=False).index
coeficientes.loc[orden].round(3).to_csv(RESULTADOS / "coeficientes.csv", index=False)
# Por unidad de esa escala: 1 querría decir que el nivel pasa tal cual a cada préstamo, como en XGBoost.
nivel = coeficientes.set_index("variable").loc["nivel"]
print(f"Nivel: coeficiente por unidad de log(tasa / (1 - tasa)) = "
      f"{np.log(nivel['multiplica_por']) / nivel['desviacion_tipica']:.2f}")

# %% fig9_sensibilidad.png: un panel por variable, probabilidad en % y escala logarítmica. Los tres primeros paneles
# comparten escala para comparar cuánto mueve cada variable; el del retraso llega al 100 %.
figura, ejes = plt.subplots(2, 2, figsize=(5.0, 4.0))
for eje, (variable, (titulo, valores, _)) in zip(ejes.flat, BARRIDOS.items()):
    for modelo, color in COLOR.items():
        del_modelo = tabla[(tabla["variable"] == variable) & (tabla["modelo"] == modelo)]
        eje.plot(del_modelo["valor"], del_modelo["probabilidad"] * 100, color=color, linestyle=TRAZO[modelo],
                 label=NOMBRE[modelo], marker="o" if variable == "mora_actual" else None)
    eje.axvline(prestamo_a[variable].item(), color="#8C8C8C", linestyle=":", linewidth=1)
    marcas = [0.1, 1, 10, 100] if variable == "mora_actual" else [0.05, 0.1, 0.2, 0.5, 1, 2]
    eje.set(title=titulo, yscale="log", ylim=(marcas[0] * 0.8, marcas[-1] * 1.2), yticks=marcas,
            yticklabels=[f"{m:g} %".replace(".", ",") for m in marcas])
    eje.minorticks_off()
ejes[1, 1].set_xticks(valores)
for eje in ejes[:, 0]:
    eje.set_ylabel("Probabilidad de impago")
figura.legend(*ejes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
guardar(figura, RESULTADOS / "fig9_sensibilidad.png")
print(tabla.pivot_table(index=["variable", "valor"], columns="modelo", values="probabilidad").round(4))
