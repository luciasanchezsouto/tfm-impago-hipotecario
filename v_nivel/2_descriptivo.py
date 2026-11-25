"""Cómo son los datos: tamaño de la cartera, tasa de impago y perfil de los préstamos.

Lee la tabla anual y escribe en descriptivo/ nueve tablas (entre ellas variables.csv, con la media, los percentiles
1, 25, 50, 75 y 99 y los valores que faltan de cada variable numérica; categorias.csv, con el porcentaje de filas de
cada categoría; calidad.csv, con los valores que faltan y los imposibles; las correlaciones y el VIF de las variables
numéricas del modelo, y saltos_nivel.csv, con cuánto cambia la tasa de impago de un año al siguiente) y cinco
figuras, entre ellas fig_saltos.png con esos saltos.

    python v_nivel/2_descriptivo.py            con la tabla completa
    python v_nivel/2_descriptivo.py piloto     con la tabla piloto
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatter, ScalarFormatter

RAIZ = Path(__file__).resolve().parents[1]
PILOTO = "piloto" in sys.argv
PANEL = RAIZ / "data" / "sencillo" / ("panel_piloto" if PILOTO else "panel") / "*.parquet"
SALIDA = Path(__file__).resolve().parent / ("descriptivo_piloto" if PILOTO else "descriptivo")
SALIDA.mkdir(parents=True, exist_ok=True)
# El mismo estilo que en 4_figuras.py: Arial como el texto y figura del tamaño impreso (letra de 9-10 puntos).
plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "dejavusans", "font.size": 9,
                     "axes.titlesize": 10, "lines.linewidth": 1.5, "lines.markersize": 3.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E0E0E0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.constrained_layout.use": True})
AZUL, NARANJA = "#0072BD", "#D95319"


def guardar(figura, fichero):
    """Guarda la figura con coma decimal en los ejes numéricos (no toca las etiquetas puestas a mano)."""
    for eje in [e for ax in figura.axes for e in (ax.xaxis, ax.yaxis)]:
        if isinstance(eje.get_major_formatter(), (ScalarFormatter, LogFormatter)):
            eje.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    figura.savefig(fichero, dpi=300)

panel = pl.scan_parquet(PANEL)


def tasa_por(grupo):
    """Préstamos, impagos y tasa de impago de cada grupo."""
    tabla = panel.group_by(grupo).agg(prestamos=pl.len(), impagos=pl.col("impago").sum(),
                                      tasa=pl.col("impago").mean())
    return tabla.sort(pl.first()).collect()


def tramos(variable, cortes):
    """Nombre del tramo en que cae cada valor: "<620", "620-659"... y "sin dato" si falta."""
    nombres = [f"<{cortes[0]}"] + [f"{a}-{b - 1}" for a, b in zip(cortes, cortes[1:])] + [f"{cortes[-1]}+"]
    tramo = sum((pl.col(variable) >= corte).cast(pl.Int8) for corte in cortes)  # cuántos cortes alcanza
    return tramo.replace_strict(dict(enumerate(nombres)), return_dtype=pl.String).fill_null("sin dato")


# %% La cartera año a año
por_anio = tasa_por("anio")
por_anio.write_csv(SALIDA / "por_anio.csv")
figura, (izquierda, derecha) = plt.subplots(1, 2, figsize=(5.7, 2.4))
izquierda.bar(por_anio["anio"], por_anio["prestamos"] / 1e6, color=AZUL)
izquierda.set(title="Préstamos vivos el 1 de enero", ylabel="Millones")
derecha.plot(por_anio["anio"], por_anio["tasa"] * 100, color=NARANJA, marker="o")
derecha.set(title="Impagan durante el año", ylabel="% de los préstamos", ylim=0)
guardar(figura, SALIDA / "fig2_cartera.png")

# %% Saltos del nivel: cuánto cambia la tasa de impago de un año al siguiente, en toda la cartera y por grupos
# Sirve para ver qué saltos se salen de lo normal, si duran más de un año y si suben igual en los préstamos que
# el 1 de enero van al corriente que en los que ya van con retraso.
al_corriente = pl.col("mora_actual").fill_null(0) == 0  # como en segmentos.csv de 3_modelos.py
por_retraso = panel.group_by("anio").agg(tasa_al_corriente=pl.col("impago").filter(al_corriente).mean(),
                                         tasa_con_retraso=pl.col("impago").filter(~al_corriente).mean())


def veces(tasa):
    """Por cuánto se multiplican las odds de impago, tasa / (1 - tasa), respecto al año anterior."""
    odds = pl.col(tasa) / (1 - pl.col(tasa))
    return odds / odds.shift()


saltos = por_anio.select("anio", "tasa").join(por_retraso.collect(), on="anio").sort("anio")
saltos = saltos.with_columns(salto=veces("tasa").log(), veces_al_corriente=veces("tasa_al_corriente"),
                             veces_con_retraso=veces("tasa_con_retraso"), veces=veces("tasa"))
# El salto es la diferencia del logit de la tasa. Se tipifica con una escala que no agrandan los saltos grandes:
# 1,4826 por la mediana de la distancia de cada salto a la mediana (en una normal, igual a la desviación típica).
escala = 1.4826 * (saltos["salto"] - saltos["salto"].median()).abs().median()
saltos = saltos.with_columns(salto_tipificado=pl.col("salto") / escala)
saltos.select("anio", "tasa", "salto", "salto_tipificado", "tasa_al_corriente", "tasa_con_retraso",
              "veces_al_corriente", "veces_con_retraso", "veces").write_csv(SALIDA / "saltos_nivel.csv")
# La figura enseña, a la izquierda, el salto tipificado de cada año (en naranja, 2008, 2009, 2020 y 2021) y, a la
# derecha, por cuánto se multiplicaron las odds de cada grupo en 2008, 2009 y 2020: barras desde 1, que es no cambiar.
figura, (izquierda, derecha) = plt.subplots(1, 2, figsize=(5.7, 2.4))
con_salto = saltos.drop_nulls("salto")
izquierda.bar(con_salto["anio"], con_salto["salto_tipificado"],
              color=[NARANJA if anio in (2008, 2009, 2020, 2021) else AZUL for anio in con_salto["anio"]])
izquierda.axhline(0, color="black", linewidth=0.6)
izquierda.set(title="Salto de la tasa de impago", ylabel="Veces la variación habitual", xticks=range(2005, 2026, 5),
              yticks=range(-6, 7, 3))
crisis = saltos.filter(pl.col("anio").is_in([2008, 2009, 2020]))
for desplazamiento, columna, nombre, color in [(-0.2, "veces_al_corriente", "Al corriente", AZUL),
                                               (0.2, "veces_con_retraso", "Con retraso", NARANJA)]:
    barras = derecha.bar(np.arange(len(crisis)) + desplazamiento, crisis[columna] - 1, bottom=1, width=0.38,
                         color=color, label=nombre)
    derecha.bar_label(barras, [f"{v:.1f}".replace(".", ",") for v in crisis[columna]], padding=1.5, fontsize=8)
derecha.set_yscale("log")
derecha.set(title="Cuánto sube el riesgo en cada grupo", ylabel="Riesgo multiplicado por", ylim=(1, 30),
            yticks=[1, 2, 5, 10, 20])
derecha.set_xticks(range(len(crisis)), crisis["anio"].cast(pl.String))
derecha.minorticks_off()
derecha.legend(fontsize=8, loc="upper left")
guardar(figura, SALIDA / "fig_saltos.png")

# %% Quién impaga: tasa por tramos de las variables que un analista miraría primero
GRUPOS = {
    "Meses de retraso el 1 de enero": pl.col("mora_actual").clip(upper_bound=2).cast(pl.String).fill_null("sin dato"),
    "FICO": tramos("fico", [620, 660, 700, 740, 780]),
    "LTV actual": tramos("ltv_actual", [60, 80, 100]),
    "Antigüedad, meses": tramos("antiguedad", [12, 36, 72, 120]),
}
figura, ejes = plt.subplots(2, 2, figsize=(4.5, 3.6))
tablas = []
for eje, (titulo, tramo) in zip(ejes.flat, GRUPOS.items()):
    tabla = tasa_por(tramo.alias("tramo")).filter(pl.col("tramo") != "sin dato")
    # Los tramos se ordenan por su primer número, no por orden alfabético.
    tabla = tabla.sort(pl.col("tramo").str.extract(r"(\d+)").cast(pl.Int32) - pl.col("tramo").str.starts_with("<"))
    eje.barh(tabla["tramo"], tabla["tasa"] * 100, color=AZUL)  # barras tumbadas: los tramos caben sin girarlos
    eje.set(title=titulo, xlabel="Impago, %")
    eje.invert_yaxis()
    eje.grid(axis="y", visible=False)
    eje.grid(axis="x")
    tablas.append(tabla.with_columns(variable=pl.lit(titulo)))
pl.concat(tablas).select("variable", "tramo", "prestamos", "impagos", "tasa").write_csv(SALIDA / "por_tramo.csv")
guardar(figura, SALIDA / "fig3_quien_impaga.png")

# %% El perfil de los préstamos nuevos ha cambiado: FICO y LTV medios por año de concesión
concesion = (pl.col("anio") - (pl.col("antiguedad") + 10) // 12).alias("concesion")
perfil = panel.filter(pl.col("antiguedad") <= 12).group_by(concesion).agg(pl.col("fico", "ltv", "dti").mean())
perfil = perfil.sort("concesion").collect()
figura, (izquierda, derecha) = plt.subplots(1, 2, figsize=(5.7, 2.4))
izquierda.plot(perfil["concesion"], perfil["fico"], color=AZUL, marker="o")
izquierda.set(title="FICO medio")
derecha.plot(perfil["concesion"], perfil["ltv"], color=AZUL, marker="o", label="LTV")
derecha.plot(perfil["concesion"], perfil["dti"], color=NARANJA, linestyle="--", marker="o", label="DTI")
derecha.set(title="LTV y DTI medios", ylabel="Porcentaje")
derecha.legend()
guardar(figura, SALIDA / "fig4_prestamos_nuevos.png")

# %% Resumen de cada variable numérica: media, desviación típica, mínimo, percentiles 1, 25, 50, 75 y 99, máximo y
# cuántos valores faltan. Los percentiles 1 y 99 dan su rango habitual.
numericas = panel.select(pl.selectors.numeric().exclude("anio")).collect()
resumen = numericas.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
resumen = resumen.transpose(include_header=True, header_name="variable", column_names="statistic")
resumen.write_csv(SALIDA / "variables.csv")

# %% Calidad del dato: valores que faltan en las 22 variables del modelo y valores imposibles
DEL_MODELO = ["fico", "ltv", "dti", "importe", "tipo_inicial", "plazo", "seguro", "unidades", "prestatarios",
              "antiguedad", "saldo", "ltv_actual", "mora_actual", "mora_maxima_12m", "meses_en_mora_12m",
              "modificado", "ocupacion", "finalidad", "primer_comprador", "canal", "inmueble", "estado"]
IMPOSIBLES = {  # comprobación: (variable, condición)
    "FICO fuera de 300-850": ("fico", ~pl.col("fico").is_between(300, 850)),
    "LTV mayor de 200": ("ltv", pl.col("ltv") > 200),
    "LTV actual mayor de 200": ("ltv_actual", pl.col("ltv_actual") > 200),
    "DTI fuera de 0-100": ("dti", ~pl.col("dti").is_between(0, 100)),
    "Tipo de interés 0 o menos": ("tipo_inicial", pl.col("tipo_inicial") <= 0),
    "Plazo 0 meses o menos": ("plazo", pl.col("plazo") <= 0),
    "Plazo más de 480 meses": ("plazo", pl.col("plazo") > 480),
    "Antigüedad negativa": ("antiguedad", pl.col("antiguedad") < 0),
    "Saldo 0 o menos": ("saldo", pl.col("saldo") <= 0),
}
cuentas = panel.select(pl.len().alias("total"), *(pl.col(v).null_count().alias(v) for v in DEL_MODELO),
                       *(condicion.sum().alias(nombre) for nombre, (_, condicion) in IMPOSIBLES.items()))
cuentas = cuentas.collect().row(0, named=True)
calidad = pl.DataFrame({
    "variable": DEL_MODELO + [variable for variable, _ in IMPOSIBLES.values()],
    "comprobacion": ["sin dato"] * len(DEL_MODELO) + list(IMPOSIBLES),
    "filas": [cuentas[nombre] for nombre in DEL_MODELO + list(IMPOSIBLES)],
}).with_columns(porcentaje=(pl.col("filas") / cuentas["total"] * 100).round(4))
calidad.write_csv(SALIDA / "calidad.csv")

# %% Porcentaje de filas de cada categoría, en las seis variables de texto del modelo y en los meses de retraso hoy
# (0, 1 o 2). El vacío cuenta como «sin dato», como en los modelos.
categorias = pl.concat([panel.group_by(categoria=pl.col(v).cast(pl.String).fill_null("sin dato")).agg(filas=pl.len())
                        .sort(["filas", "categoria"], descending=[True, False])
                        .select(pl.lit(v).alias("variable"), "categoria", "filas")
                        for v in DEL_MODELO[16:] + ["mora_actual"]]).collect()
categorias = categorias.with_columns(porcentaje=(pl.col("filas") / cuentas["total"] * 100).round(4))
categorias.write_csv(SALIDA / "categorias.csv")

# %% Correlaciones de Spearman (con el impago y entre ellas) y VIF de las 16 variables numéricas del modelo
NUMERICAS = DEL_MODELO[:16]  # las seis últimas son de texto
# Filas: 1 de cada 100 préstamos, elegidos por hash del identificador, con todos sus años (unos 2,2 millones).
muestra = panel.filter(pl.col("id").hash(seed=0) % 100 == 0).select("impago", *NUMERICAS).collect()
# Ausentes: cada correlación usa solo las filas en que las dos variables tienen dato (el que más falta es DTI, en
# el 10 % de las filas); con Spearman, los puestos se calculan dentro de esas filas. El VIF sale de esas correlaciones.


def correlaciones(metodo):
    """Correlación entre cada par de columnas de la muestra, en una tabla cuadrada."""
    columnas = muestra.columns
    pares = muestra.select(pl.corr(a, b, method=metodo).alias(f"{a}/{b}") for a in columnas for b in columnas)
    return pl.DataFrame(np.reshape(pares.row(0), (len(columnas), len(columnas))), schema=columnas)


spearman = correlaciones("spearman")
con_impago = pl.DataFrame({"variable": NUMERICAS, "spearman": spearman["impago"][1:]})
con_impago.sort(pl.col("spearman").abs(), descending=True).write_csv(SALIDA / "correlacion_impago.csv")
entre_ellas = spearman[1:].drop("impago")
pl.DataFrame({"variable": NUMERICAS}).hstack(entre_ellas).write_csv(SALIDA / "correlacion_variables.csv")
# VIF = 1 / (1 - R²) de la regresión de cada variable sobre las otras 15: la diagonal de la inversa de sus
# correlaciones de Pearson. Mide cuánto se agranda la varianza de su coeficiente por parecerse a las demás.
vif = np.diag(np.linalg.inv(correlaciones("pearson")[1:].drop("impago").to_numpy()))
pl.DataFrame({"variable": NUMERICAS, "vif": vif}).sort("vif", descending=True).write_csv(SALIDA / "vif.csv")

# La figura muestra la mitad de abajo de la matriz (la otra repite los valores), sin la diagonal de unos.
NOMBRES = {"fico": "FICO", "ltv": "LTV al conceder", "dti": "Deuda sobre ingresos (DTI)",
           "importe": "Importe concedido", "tipo_inicial": "Tipo de interés", "plazo": "Plazo",
           "seguro": "Seguro hipotecario", "unidades": "Viviendas", "prestatarios": "Número de prestatarios",
           "antiguedad": "Antigüedad", "saldo": "Saldo pendiente", "ltv_actual": "LTV actual",
           "mora_actual": "Meses de retraso hoy", "mora_maxima_12m": "Retraso máximo en 12 meses",
           "meses_en_mora_12m": "Meses con retraso en 12 meses", "modificado": "Préstamo modificado"}
matriz = entre_ellas.to_numpy(writable=True)
matriz[np.triu_indices(len(NUMERICAS))] = np.nan
matriz = matriz[1:, :-1]  # la primera fila y la última columna quedan vacías
figura, eje = plt.subplots(figsize=(5.7, 4.9))
imagen = eje.imshow(matriz, cmap=LinearSegmentedColormap.from_list("", [AZUL, "#F2F2F2", NARANJA]), vmin=-1, vmax=1)
for (fila, columna), valor in np.ndenumerate(matriz):
    if not np.isnan(valor):  # + 0 evita que salga "-0,00"
        eje.text(columna, fila, f"{round(valor, 2) + 0:.2f}".replace(".", ","), ha="center", va="center",
                 fontsize=5, color="white" if abs(valor) > 0.6 else "black")
eje.set_xticks(range(len(matriz)), [NOMBRES[v] for v in NUMERICAS[:-1]], rotation=45, ha="right",
               rotation_mode="anchor")
eje.set_yticks(range(len(matriz)), [NOMBRES[v] for v in NUMERICAS[1:]])
eje.tick_params(length=0)  # sin las rayitas de los ejes, que junto a las casillas parecen signos menos
eje.grid(False)
eje.spines[:].set_visible(False)
figura.colorbar(imagen, shrink=0.6, ticks=[-1, -0.5, 0, 0.5, 1], label="Correlación de Spearman")
guardar(figura, SALIDA / "fig_correlaciones.png")
print(f"Escrito en {SALIDA}")
