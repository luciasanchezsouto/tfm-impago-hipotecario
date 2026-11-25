"""Aplicación: cada enero, qué hipotecas revisar con la capacidad del equipo.

Dos pantallas. «Evaluación de cartera» puntúa una cartera, prevé sus impagos del año y da los préstamos que el equipo
puede revisar, de más a menos probabilidad de impago, en dos bloques (con cuotas impagadas y al corriente con alerta)
y con el motivo principal de cada uno; el coste y el ahorro van aparte, en una simulación opcional. «Evaluación de
préstamo» explica la probabilidad de un préstamo y qué pasa si cambia un dato. Las dos usan los modelos con nivel de
2025, que 6_nivel.py deja en modelos/, junto a este fichero, y la tasa de impago prevista para el año: uno de tres
escenarios calculados con descriptivo/por_anio.csv u otra escrita a mano. De resultados/ leen cuatro tablas agregadas:
nivel.csv (cuánto se desviaron los impagos previstos en los años pasados, para la horquilla), umbrales_nivel.csv (qué
parte de los impagos se encontró con una lista de cada tamaño), y discrepancias.csv y rangos.csv (los avisos). Los
textos de pantalla no usan términos técnicos y son cortos: las explicaciones van en la ayuda de cada cifra, campo o
columna. Nada necesita los datos de Freddie Mac, y todos los préstamos que se ven son inventados.

    streamlit run v_nivel/5_app.py
"""

import io
import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

AQUI = Path(__file__).resolve().parent
# Los modelos con nivel de 2025, junto al código; si no están, los de 6_nivel.py o los de su piloto (mismo formato).
# Los modelos sin nivel de 3_modelos.py no sirven: no admiten el nivel del año.
MODELOS = next((c for c in [AQUI / "modelos", *(AQUI.parent / "data" / "sencillo" / n
                                                for n in ["modelos_nivel", "modelos_nivel_piloto"])]
                if c.exists()), AQUI / "modelos")
EJEMPLO = AQUI / "ejemplo_cartera.csv"

# Nombres que ve el usuario, y su ayuda. Los internos (columnas y ficheros: xgboost, logit) no cambian.
MODELO = {"xgboost": "Modelo principal", "logit": "Modelo de contraste"}
AYUDA_MODELO = {"xgboost": "La cifra de referencia: la misma que ordena la lista.",
                "logit": "Una segunda opinión, de un modelo más sencillo. Si se separa mucho de la del modelo "
                         "principal, sale un aviso."}

NOMBRES = {
    "fico": "Puntuación FICO", "ltv": "LTV al conceder, %", "dti": "Deuda sobre ingresos (DTI), %",
    "importe": "Importe concedido, $", "tipo_inicial": "Tipo de interés, %", "plazo": "Plazo, meses",
    "seguro": "Seguro hipotecario, %", "unidades": "Viviendas", "prestatarios": "Prestatarios",
    "antiguedad": "Antigüedad, meses", "saldo": "Saldo pendiente, $", "ltv_actual": "LTV actual, %",
    "mora_actual": "Cuotas impagadas hoy", "mora_maxima_12m": "Máximo de cuotas impagadas en 12 meses",
    "meses_en_mora_12m": "Meses con cuotas impagadas, de 12", "modificado": "Préstamo modificado",
    "ocupacion": "Ocupación", "finalidad": "Finalidad", "primer_comprador": "Primera vivienda",
    "canal": "Canal", "inmueble": "Tipo de inmueble", "estado": "Estado",
}
COLUMNAS = [v for v in NOMBRES if v != "ltv_actual"]  # las del CSV; el LTV actual se calcula
# Texto de cada código de Freddie Mac en los desplegables; el modelo sigue recibiendo el código.
ETIQUETAS = {
    "ocupacion": {"P": "Vivienda habitual", "S": "Segunda residencia", "I": "Inversión"},
    "finalidad": {"P": "Compra", "C": "Refinanciación con disposición", "N": "Refinanciación sin disposición"},
    "primer_comprador": {"Y": "Sí", "N": "No"},
    "canal": {"R": "Directo", "B": "Bróker", "C": "Corresponsal", "T": "Intermediario sin especificar"},
    "inmueble": {"SF": "Unifamiliar", "PU": "Urbanización (PUD)", "CO": "Piso en condominio", "CP": "Cooperativa",
                 "MH": "Vivienda prefabricada"},
    "modificado": {0: "No", 1: "Sí"},
}
# Préstamos inventados del formulario: los cinco de ejemplo_cartera.csv y dos más.
EJEMPLOS = dict(zip(["A: buen FICO, al corriente", "B: primera vivienda, una cuota impagada",
                     "C: refinanciación en California", "D: modificado, dos cuotas impagadas",
                     "E: segunda residencia en Nueva York"], pd.read_csv(EJEMPLO).to_dict("records")))
EJEMPLOS["F: inversión, dos viviendas en Illinois"] = dict(
    EJEMPLOS["A: buen FICO, al corriente"], fico=720, ltv=70, dti=36, importe=350000, tipo_inicial=7.0, unidades=2,
    prestatarios=1, antiguedad=12, saldo=346000, ocupacion="I", estado="IL")
# G enseña en qué se diferencian los modelos: un cliente de buen FICO (el de A) que paga tarde casi todos los meses.
EJEMPLOS["G: buen FICO, paga tarde casi todos los meses"] = dict(
    EJEMPLOS["A: buen FICO, al corriente"], mora_actual=1, mora_maxima_12m=1, meses_en_mora_12m=9)
ES = {"number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["", " $"]}}  # cifras de los gráficos
BLOQUES = ["Con cuotas impagadas", "Al corriente con alerta"]  # una o dos cuotas impagadas hoy, y ninguna
TRAMOS = {"Muy alto": 20, "Alto": 5, "Medio": 1}  # veces la tasa prevista para el año; por debajo, «Bajo»
# Mandos de «¿Y si…?»: mínimo, máximo y paso. El primero va a la vista; los otros, en un desplegable.
MANDOS = {"mora_actual": (0, 2, 1), "fico": (300, 850, 5), "ltv": (10, 150, 1), "tipo_inicial": (2.0, 10.0, 0.125)}
GRIS = {"delta_color": "off", "delta_arrow": "off"}  # la línea de debajo de una cifra, como nota y sin flecha


@st.cache_resource
def cargar():
    """Los dos modelos del último año guardado y los niveles de las variables de texto; None si no hay modelos."""
    guardados = [int(f.stem.split("_")[1]) for f in MODELOS.glob("logit_*.joblib")]
    if not guardados:
        return None
    return (joblib.load(MODELOS / f"logit_{max(guardados)}.joblib"),
            joblib.load(MODELOS / f"xgboost_{max(guardados)}.joblib"), joblib.load(MODELOS / "categorias.joblib"))


@st.cache_data
def cartera_ejemplo(n=500, semilla=2026):
    """Cartera inventada: cada variable se sortea, con semilla fija, en un rango redondo escrito a mano.
    Ninguna fila sale de los datos de Freddie Mac ni del panel."""
    azar = np.random.default_rng(semilla)
    sorteo = lambda opciones, pesos: azar.choice(opciones, n, p=pesos)  # un valor de la lista con esos pesos
    c = pd.DataFrame({
        "fico": azar.integers(640, 821, n), "ltv": azar.integers(40, 98, n), "dti": azar.integers(15, 51, n),
        "importe": azar.integers(60, 801, n) * 1000, "tipo_inicial": azar.choice(np.arange(2.5, 8.01, 0.125), n),
        "plazo": sorteo([180, 240, 360], [0.15, 0.05, 0.8]), "unidades": sorteo([1, 2, 3, 4], [0.96, 0.02, 0.01, 0.01]),
        "prestatarios": sorteo([1, 2], [0.5, 0.5]), "antiguedad": azar.integers(1, 121, n),
        "mora_actual": sorteo([0, 1, 2], [0.98, 0.015, 0.005]), "modificado": sorteo([0, 1], [0.99, 0.01]),
        "ocupacion": sorteo(["P", "S", "I"], [0.88, 0.04, 0.08]), "finalidad": sorteo(["P", "N", "C"], [0.5, 0.3, 0.2]),
        "primer_comprador": sorteo(["N", "Y"], [0.7, 0.3]), "canal": sorteo(["R", "B", "C"], [0.5, 0.15, 0.35]),
        "inmueble": sorteo(["SF", "PU", "CO"], [0.65, 0.25, 0.1]),
        "estado": azar.choice(["CA", "TX", "FL", "NY", "IL", "OH", "GA", "WA", "AZ", "PA"], n)})
    c["seguro"] = np.where(c["ltv"] > 80, 25, 0)  # seguro hipotecario solo por encima del 80 % de LTV
    c["saldo"] = (c["importe"] * (1 - c["antiguedad"] / c["plazo"] / 2)).round(-3).astype(int)  # se amortiza despacio
    # La peor situación del último año es al menos la de hoy, y cada cuota impagada es un mes con cuotas impagadas.
    c["mora_maxima_12m"] = np.maximum(c["mora_actual"], sorteo([0, 1, 2], [0.95, 0.04, 0.01]))
    c["meses_en_mora_12m"] = np.where(c["mora_maxima_12m"] > 0, c["mora_maxima_12m"] + azar.integers(0, 4, n), 0)
    c.loc[c["finalidad"] != "P", "primer_comprador"] = "N"  # solo se estrena vivienda al comprar
    return c[COLUMNAS]


def tabla(ruta):
    """Una tabla agregada de resultados/ o descriptivo/; None si todavía no se ha calculado."""
    return st.cache_data(pd.read_csv)(AQUI / ruta) if (AQUI / ruta).exists() else None


def coma(texto):
    """Coma decimal y espacio antes del signo de porcentaje, como se escribe en español."""
    return re.sub(r"(?<=[0-9])[.](?=[0-9])", ",", texto).replace("%", " %")


def numero(valor, decimales=0):
    """Un número con punto en los miles y coma decimal: 1.234.567,8."""
    valor = round(valor, decimales) + 0  # sin «-0» cuando un negativo se redondea a cero
    return f"{valor:,.{decimales}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def falta(valor):
    """Si el dato falta: None o vacío."""
    return valor is None or (not isinstance(valor, str) and pd.isna(valor))


def legible(variable, valor):
    """El valor de una variable como se enseña: el texto de su código, o el número con punto de miles y coma decimal."""
    if falta(valor):
        return "sin dato"
    if isinstance(valor, str) or valor in ETIQUETAS.get(variable, {}):
        return ETIQUETAS.get(variable, {}).get(valor, valor)
    texto = numero(float(valor), 3 if variable == "tipo_inicial" else 1)  # sin ceros al final: 6,5 y no 6,500
    return texto.rstrip("0").rstrip(",") if "," in texto else texto


def corto(variable):
    """El nombre de una variable sin la unidad: «Importe concedido»."""
    return NOMBRES[variable].split(",")[0]


def dato(variable, valor):
    """Nombre, valor y unidad de un dato: «Antigüedad: 24 meses», «Meses con cuotas impagadas: 4 de 12»."""
    unidad = NOMBRES[variable].partition(", ")[2]
    return f"{corto(variable)}: {legible(variable, valor)}" + (f" {unidad}" if unidad and not falta(valor) else "")


cuotas = lambda n: f"{n:.0f} cuota impagada" if n == 1 else f"{n:.0f} cuotas impagadas"  # en singular o plural
# El dato que más sube el riesgo, en lenguaje llano. Si la frase no encaja con el valor, va la genérica de motivo().
# Que un dato suba el riesgo frente a la media no dice que sea alto o bajo: esas frases solo nombran el dato.
PLANTILLAS = {
    "mora_actual": lambda x: x >= 1 and f"Tiene {cuotas(x)}",
    "mora_maxima_12m": lambda x: x >= 1 and f"Llegó a {cuotas(x)} en los últimos 12 meses",
    "meses_en_mora_12m": lambda x: x >= 1 and f"Pagó tarde {x:.0f} de los últimos 12 meses",
    "fico": lambda x: f"Su puntuación FICO ({x:.0f})",
    "ltv_actual": lambda x: coma(f"Lo que debe frente al valor de la vivienda (LTV actual {x:.0f}%)"),
    "dti": lambda x: coma(f"Su deuda sobre ingresos (DTI {x:.0f}%)"),
    "tipo_inicial": lambda x: f"Su tipo de interés ({legible('tipo_inicial', x)} %)",
    "modificado": lambda x: x == 1 and "Préstamo modificado",
    "prestatarios": lambda x: x == 1 and "Un solo prestatario",
}


def motivo(variable, valor):
    """El dato que más sube el riesgo de un préstamo, dicho en lenguaje llano; nunca se oculta cuál es."""
    frase = variable in PLANTILLAS and not falta(valor) and PLANTILLAS[variable](valor)
    return frase or f"Otras características del préstamo: {dato(variable, valor)}"


def escenarios(por_anio):
    """Tasa de impago prevista para el año, en % con dos decimales, en cada escenario, a partir de la tasa de cada
    año de toda la cartera de Freddie Mac (descriptivo/por_anio.csv)."""
    if por_anio is None:  # sin la tabla, un valor fijo: la tasa de 2025, último año cerrado al entregar
        return {"Base: la de 2025, último año cerrado al entregar (0,55 %)": 0.55}
    tasas = (100 * por_anio.set_index("anio")["tasa"].sort_index()).round(2)
    ultimo, mayor = tasas.index[-1], tasas.tail(5).idxmax()
    opciones = {coma(f"Base: la del último año cerrado ({tasas[ultimo]:.2f}%, {ultimo})"): float(tasas[ultimo]),
                coma(f"Prudente: la mayor de los cinco últimos años ({tasas[mayor]:.2f}%, {mayor})"): float(tasas[mayor])}
    if 2009 in tasas.index:
        opciones[coma(f"Severa: la de 2009, el peor año de la crisis financiera ({tasas[2009]:.2f}%)")] = float(tasas[2009])
    return opciones


def anio_anomalo(por_anio):
    """Año, tasa y mediana de los cinco años anteriores si la tasa del último año cerrado pasa del doble de esa
    mediana: un año anómalo, cuya tasa conviene no usar como previsión. None si no lo es."""
    tasas = por_anio.set_index("anio")["tasa"].sort_index()
    mediana = tasas.iloc[-6:-1].median()
    return (int(tasas.index[-1]), tasas.iloc[-1], mediana) if tasas.iloc[-1] > 2 * mediana else None


def tasa_elegida():
    """La tasa de impago prevista para el año, en %, y el escenario elegido en «Evaluación de cartera» (o el base)."""
    opciones = escenarios(tabla("descriptivo/por_anio.csv"))
    escenario = st.session_state.get("escenario", next(iter(opciones)))
    if escenario == "Otra":
        return st.session_state.get("tasa_otra", next(iter(opciones.values()))), escenario
    return opciones[escenario], escenario


def elegir_tasa(columna):
    """Elige la tasa de impago prevista. Los dos campos llevan clave y no valor inicial: lo elegido no se pierde al
    escribir dos valores seguidos y se conserva al cambiar de pantalla (véase el final del fichero)."""
    por_anio = tabla("descriptivo/por_anio.csv")
    opciones = escenarios(por_anio)
    escenario = columna.selectbox("Tasa de impago prevista para el año", [*opciones, "Otra"], key="escenario", help=(
        "La fija el área de Riesgos. Es la tasa de referencia de toda la cartera de Freddie Mac, no la de la cartera "
        "subida. Cambiarla no cambia el orden de la lista, sí el número de impagos previstos."))
    if escenario == "Otra":
        st.session_state.setdefault("tasa_otra", next(iter(opciones.values())))
        columna.number_input("Otra tasa de impago prevista, %", 0.01, 20.0, step=0.05, format="%.2f", key="tasa_otra")
    anomalo = None if por_anio is None else anio_anomalo(por_anio)
    if anomalo:
        st.warning(coma(f"El último año cerrado, {anomalo[0]}, fue anómalo: su tasa de impago ({anomalo[1]:.2%}) pasa "
                        f"del doble de la mediana de los cinco años anteriores ({anomalo[2]:.2%}). Elija «Otra» y "
                        "escriba una tasa más parecida a la de esos años."))
    return tasa_elegida()


def dibujar(grafico):
    """Muestra un gráfico de Altair con las cifras en español."""
    st.altair_chart(grafico.configure(locale=ES))


def puntuar(prestamos, tasa):
    """Añade a cada préstamo la probabilidad de impago de cada modelo y devuelve también las aportaciones.
    Los dos modelos parten del nivel del año, la tasa de impago prevista (en tanto por uno), y le suman el perfil."""
    logit, xgboost, categorias = cargar()
    numericas = [v for v in COLUMNAS if v not in categorias]
    datos = prestamos[COLUMNAS].copy()
    datos[numericas] = datos[numericas].apply(pd.to_numeric, errors="coerce")  # un dato que falta queda vacío
    # LTV actual: saldo sobre el valor de la vivienda al conceder. Con un importe de cero no se puede calcular.
    datos["ltv_actual"] = (datos["saldo"] / datos["importe"] * datos["ltv"]).replace([np.inf, -np.inf], np.nan)
    datos = datos[list(NOMBRES)]
    for variable, niveles in categorias.items():
        datos[variable] = pd.Categorical(datos[variable].fillna("sin dato"), categories=niveles)
    datos[numericas + ["ltv_actual"]] = datos[numericas + ["ltv_actual"]].astype("float32")
    # El nivel en la escala de los modelos, log(tasa / (1 - tasa)): la logística lo recibe como la última columna,
    # «nivel», y XGBoost como punto de partida de cada fila (sin él, las probabilidades salen descalibradas).
    z = np.log(tasa / (1 - tasa))
    margen = np.full(len(datos), z)
    resultado = prestamos.assign(logit=logit.predict_proba(datos.assign(nivel=np.float32(z)))[:, 1],
                                 xgboost=xgboost.predict_proba(datos, base_margin=margen)[:, 1],
                                 ltv_actual=datos["ltv_actual"])
    # Aportación de cada variable a la predicción del modelo principal (valores SHAP); la última columna es la base.
    aportaciones = xgboost.get_booster().predict(xgb.DMatrix(datos, enable_categorical=True, base_margin=margen),
                                                 pred_contribs=True)
    return resultado, pd.DataFrame(aportaciones[:, :-1], columns=list(NOMBRES), index=prestamos.index)


@st.cache_data(show_spinner=False)
def horquilla_impagos(probabilidades, cocientes, sorteos=2_000, semilla=2026, exacto=2_000):
    """Entre cuántos y cuántos impagos quedaría la cartera en el año en 8 de cada 10 casos: la horquilla «entre…».

    Junta las dos razones por las que los impagos reales no son los previstos, sorteando el año muchas veces:
    - Lo que se desvió el modelo en los años pasados. Cada cociente c es lo que previó el modelo principal entre lo
      que pasó en uno de los años de prueba, con la tasa del último año cerrado (resultados/nivel.csv; c = 2 es que
      previó el doble). Para cada c, la probabilidad p de cada préstamo se corrige a la de un año así dividiendo su
      odds, p / (1 - p), por c: p' = p / (p + c (1 - p)). Es mover el punto de partida del modelo (el nivel, que va en
      la escala de las odds) lo que se desvió ese año; con probabilidades pequeñas equivale a dividirlas por c, y p'
      nunca pasa de 1.
    - El azar de una cartera de este tamaño: aunque las p' fueran exactas, el número de impagos varía de un año a
      otro, y mucho en una cartera pequeña. Cada sorteo decide si impaga cada préstamo con su p' (un Bernoulli
      independiente) y cuenta los impagos.
    Con el mismo número de sorteos por cociente, cada año pesa igual (17 × 2.000 = 34.000 sorteos), y la horquilla va
    de los percentiles 10 a 90 de todos ellos, redondeados a enteros. La semilla fija hace que salga siempre lo mismo.

    Con más de `exacto` préstamos, para que tarde poco con cualquier tamaño de cartera, se sortean juntos los
    préstamos cuya odds difiere menos de un 5 % (los del mismo tramo de 0,05 en el logaritmo de la odds; corregir por c
    resta lo mismo a todos y no cambia los tramos): cuántos impagan en cada tramo sale de una binomial con su p' media.
    La media del total es la misma, y su varianza apenas cambia: la binomial le añade la suma de los (p'_i - p' media)²
    del tramo, menos del 0,3 % de su varianza con odds tan parecidas. En carteras así, además, el azar pesa poco frente
    a lo que se desvió el modelo."""
    p = np.clip(np.asarray(probabilidades, dtype=float), 1e-12, 1 - 1e-12)
    azar, grande, impagos = np.random.default_rng(semilla), len(p) > exacto, []
    if grande:
        _, tramo, tamano = np.unique(np.floor((np.log(p) - np.log1p(-p)) / 0.05), return_inverse=True,
                                     return_counts=True)
    for c in cocientes:
        corregida = p / (p + c * (1 - p))
        if grande:
            media = np.bincount(tramo, weights=corregida) / tamano
            impagos.append(azar.binomial(tamano, media, (sorteos, len(tamano))).sum(axis=1))
        else:  # préstamo a préstamo, en precisión simple, que basta y va el doble de rápido
            uniformes = azar.random((sorteos, len(p)), dtype=np.float32)
            impagos.append((uniformes < corregida.astype(np.float32)).sum(axis=1))
    bajo, alto = np.rint(np.quantile(np.concatenate(impagos), [0.1, 0.9]))
    return int(bajo), int(alto)


def discrepancia(logit, xgboost):
    """Qué modelo da más riesgo cuando una probabilidad es al menos el triple de la otra y alguna llega al 5 %."""
    alguna = np.maximum(logit, xgboost) >= 0.05
    return np.select([alguna & (xgboost >= 3 * logit), alguna & (logit >= 3 * xgboost)],
                     ["xgboost da más riesgo", "logit da más riesgo"], "")


def que_hacer(logit, xgboost):
    """Si los dos modelos discrepan, avisa y dice qué hacer, con la tasa de impago que tuvieron en los años de prueba
    los préstamos en que discreparon igual (resultados/discrepancias.csv, de 10_discrepancias.py)."""
    caso = str(discrepancia(logit, xgboost))
    if not caso:
        return
    historia, pasado = tabla("resultados/discrepancias.csv"), ""
    if historia is not None:
        grupo = historia[historia["grupo"] == caso]
        pasado = coma(f" (en el pasado impagó el {grupo['impagos'].sum() / grupo['prestamos'].sum():.0%} de los "
                      "préstamos así)")
    mas, menos = ("principal", "de contraste") if caso == "xgboost da más riesgo" else ("de contraste", "principal")
    (st.warning if mas == "principal" else st.info)(
        f"Los dos modelos discrepan: el modelo {mas} ve mucho más riesgo que el {menos}{pasado}. **Qué hacer:** que un "
        "analista compruebe los datos de este préstamo.")


def fuera_de_rango(puntuados):
    """Qué datos de cada préstamo puntuado quedan fuera del rango en que está el 99,8 % de los préstamos
    (resultados/rangos.csv, de 10_discrepancias.py): verdadero o falso por préstamo y variable."""
    fuera, rangos = pd.DataFrame(index=puntuados.index), tabla("resultados/rangos.csv")
    for r in [] if rangos is None else rangos.itertuples():
        valor = pd.to_numeric(puntuados[r.variable], errors="coerce")
        fuera[r.variable] = valor.notna() & ~valor.between(r.minimo, r.maximo)
    return fuera


def poco_habitual(puntuado):
    """Avisa de los datos de un préstamo puntuado que quedan fuera del rango habitual."""
    rangos = tabla("resultados/rangos.csv")
    if rangos is None:
        return
    fuera = fuera_de_rango(puntuado.to_frame().T).iloc[0]
    textos = [f"{corto(r.variable)} ({legible(r.variable, puntuado[r.variable])}; lo habitual va de "
              f"{legible(r.variable, r.minimo)} a {legible(r.variable, r.maximo)})"
              for r in rangos.itertuples() if fuera[r.variable]]
    if textos:
        st.info("Dato poco habitual: " + "; ".join(textos) + ". Compruébelo: con datos así, la probabilidad es "
                "orientativa.")


def sin_modelos():
    """Avisa si faltan los modelos de la carpeta modelos/."""
    if cargar() is None:
        st.warning("Esta pantalla necesita los modelos de la carpeta modelos/, que no se encuentra. Se crean con "
                   "1_datos.py, 2_descriptivo.py y 6_nivel.py.")
    return cargar() is None


def ficha_del_modelo():
    """Barra lateral: qué predice la herramienta y con qué."""
    st.sidebar.subheader("Ficha del modelo")
    st.sidebar.markdown(
        "**Qué predice.** La probabilidad de que una hipoteca impague durante el año. Impagar es llegar a tres cuotas "
        "impagadas o cerrar el préstamo con pérdida; con una o dos cuotas impagadas, todavía no ha impagado.\n\n"
        "**Con qué.** Dos modelos, el principal y el de contraste, entrenados con hipotecas de Freddie Mac de 2000 a "
        "2024.")


def leer_cartera(fichero):
    """La cartera del usuario o la de ejemplo, con un identificador por préstamo; None si faltan columnas."""
    cartera = pd.read_csv(fichero) if fichero else cartera_ejemplo()
    faltan = [c for c in COLUMNAS if c not in cartera]
    if faltan:
        st.error(f"Al fichero le faltan estas columnas: {', '.join(faltan)}. Debe tener las mismas columnas que el "
                 "ejemplo, con los mismos nombres.")
        return None
    if "id" not in cartera:
        cartera.insert(0, "id", range(1, len(cartera) + 1))
    # Los códigos con que Freddie Mac marca el dato que falta pasan a vacío, como en el entrenamiento.
    for variable, tope in {"fico": 850, "dti": 65, "ltv": 998}.items():
        cartera[variable] = cartera[variable].where(cartera[variable] <= tope)
    en_impago = cartera["mora_actual"] >= 3
    if en_impago.any():
        st.warning(f"{numero(en_impago.sum())} préstamos tienen tres o más cuotas impagadas: ya están en impago y no "
                   "se puntúan.")
    return cartera[~en_impago].reset_index(drop=True)


def abrir_prestamo(prestamo):
    """Lleva un préstamo de la lista a la pantalla «Evaluación de préstamo» (lo llama abrir_elegido)."""
    st.session_state["de_la_lista"] = prestamo
    st.session_state["pantalla"] = "Evaluación de préstamo"


def abrir_elegido(clave, prestamos):
    """Lleva a «Evaluación de préstamo» el préstamo marcado en la lista (se llama al marcar su fila)."""
    filas = st.session_state[clave].selection.rows
    if filas:
        fila = prestamos.iloc[filas[0]]
        st.session_state["aperturas"] = st.session_state.get("aperturas", 0) + 1  # la lista vuelve sin fila marcada
        abrir_prestamo({"puesto": int(fila["puesto"]), **{v: None if pd.isna(fila[v]) else fila[v] for v in COLUMNAS}})


def guardar_cartera():
    """Guarda en la sesión la cartera subida (o la quita, si se retira el fichero): al volver de «Evaluación de
    préstamo» el campo del fichero sale vacío, pero la cartera sigue siendo la misma."""
    fichero = st.session_state["fichero_cartera"]
    st.session_state["cartera_subida"] = (fichero.name, fichero.getvalue()) if fichero else None


def pagina_cartera():
    st.title("Evaluación de cartera")
    st.write("Qué préstamos revisar primero este año, con la capacidad de su equipo.")
    if sin_modelos():
        return
    # Lo que casi nunca se toca va plegado: la cartera (por defecto, la de ejemplo) y la tasa prevista (la base).
    subida = st.session_state.get("cartera_subida")  # nombre y contenido del fichero subido
    with st.expander(f"Cartera: {subida[0]}" if subida else "Cartera: ejemplo de 500 préstamos inventados"):
        st.file_uploader("Cartera", type="csv", key="fichero_cartera", on_change=guardar_cartera, help=(
            "Un CSV con una fila por préstamo y las columnas del ejemplo. Sin fichero, se usa una cartera de ejemplo de "
            "500 préstamos inventados, con más cuotas impagadas de lo habitual para que la lista tenga casos."))
        st.download_button("Descargar el ejemplo", cartera_ejemplo().to_csv(index=False), "ejemplo_cartera_500.csv")
    cartera = leer_cartera(io.BytesIO(subida[1]) if subida else None)
    if cartera is None or cartera.empty:
        return
    por_anio = tabla("descriptivo/por_anio.csv")
    anomalo = por_anio is not None and anio_anomalo(por_anio) is not None  # entonces va abierto: que se vea el aviso
    with st.expander("Tasa de impago prevista para el año", expanded=anomalo):
        tasa, escenario = elegir_tasa(st.columns(2)[0])
    resultado, aportaciones = puntuar(cartera, tasa / 100)
    orden = resultado["xgboost"].sort_values(ascending=False).index  # el orden es siempre el de la probabilidad
    todos, aportaciones = resultado.loc[orden].reset_index(drop=True), aportaciones.loc[orden].reset_index(drop=True)
    probabilidad = todos["xgboost"]

    revisar = st.columns(3)[0].number_input(
        "Préstamos que puede revisar el equipo", 1, len(cartera), max(1, round(len(cartera) / 10)),
        help="Por defecto, el 10 % de la cartera. Escriba la capacidad real del equipo.")
    esperados, en_lista = probabilidad.sum(), probabilidad.head(revisar).sum()
    horquilla = None
    ayuda = (f"De los {numero(len(todos))} préstamos de la cartera, sin los que ya están en impago. Es la suma de sus "
             "probabilidades de impago, con la tasa prevista elegida.")
    nivel = tabla("resultados/nivel.csv")
    if nivel is not None and not escenario.startswith("Base"):  # lo que se desvió el modelo se midió con la base
        ayuda += (" Con esta tasa no se da horquilla, porque el acierto del modelo en los años pasados se midió con la "
                  "tasa del último año cerrado.")
    elif nivel is not None:  # impagos previstos entre reales del modelo principal en los años de prueba, desde 2009
        cociente = nivel[(nivel["modelo"] == "xgboost") & (nivel["anio"] >= 2009)].set_index("anio")["calibracion_ultimo"]
        bajo, alto = horquilla_impagos(probabilidad.to_numpy(), tuple(cociente))
        horquilla = f"entre {numero(bajo)} y {numero(alto)}"
        ayuda += (" **Entre…:** el rango en que quedarían los impagos en 8 de cada 10 casos, contando lo que el modelo "
                  f"se desvió en los años pasados (de {cociente.index.min()} a {cociente.index.max()}) y el azar de "
                  "una cartera de este tamaño. No es una garantía.")
    medido = "Los impagos previstos entre los préstamos de la lista."
    umbrales = tabla("resultados/umbrales_nivel.csv")
    if umbrales is not None:  # lo medido en los años de prueba con la lista de tamaño más parecido
        umbrales = umbrales[umbrales["modelo"] == "xgboost"]
        parte = min(umbrales["revisado"].unique(), key=lambda r: abs(r - 100 * revisar / len(todos)))
        anios = umbrales[umbrales["revisado"] == parte]
        recall = anios["sensibilidad"]
        medido += coma(f" De {anios['anio'].min()} a {anios['anio'].max()}, una lista del {parte}% de la cartera de "
                       f"Freddie Mac encontró entre el {recall.min():.0%} y el {recall.max():.0%} de los impagos del "
                       f"año ({recall.mean():.0%} de media).")
    a, b = st.columns(2)
    a.metric("Impagos previstos en el año", f"unos {numero(esperados)}", horquilla, help=ayuda, **GRIS)
    b.metric("De ellos, en la lista", f"unos {numero(en_lista)}", coma(f"{en_lista / esperados:.0%} del total"),
             help=medido, **GRIS)

    st.subheader("Lista de revisión")
    lista, aportan = todos.head(revisar).copy(), aportaciones.head(revisar)
    lista["motivo"] = [motivo(v, lista.at[i, v]) if aportan.at[i, v] > 0 else "" for i, v in aportan.idxmax(axis=1).items()]
    raros = [", ".join(corto(v) for v in fila.index[fila]) for _, fila in fuera_de_rango(lista).iterrows()]
    lista["verificar"] = ["; ".join(["Los dos modelos discrepan"] * bool(d) + [f"Dato poco habitual: {r}"] * bool(r))
                          for d, r in zip(discrepancia(lista["logit"], lista["xgboost"]), raros)]
    lista["tramo"] = np.select([lista["xgboost"] >= veces * tasa / 100 for veces in TRAMOS.values()], list(TRAMOS), "Bajo")
    lista["ponderado"] = (lista["xgboost"] * pd.to_numeric(lista["saldo"], errors="coerce")).round().astype("Int64")
    lista["bloque"] = np.where(lista["mora_actual"] >= 1, *BLOQUES)
    lista["probabilidad"] = [coma(f"{100 * p:5.1f}%") for p in lista["xgboost"]]  # texto alineado: ordena bien
    lista["ltv_actual"] = lista["ltv_actual"].astype(float).round(1)
    lista.insert(0, "puesto", range(1, len(lista) + 1))
    # En pantalla, lo justo para decidir; el resto de datos va en el fichero que se descarga.
    visibles = ["puesto", "id", "probabilidad", "motivo", "verificar"]
    formato = st.column_config  # el nombre de cada columna y su ayuda, que se ve al pasar el cursor
    columnas = {"puesto": formato.Column("Puesto", help="El lugar en la lista, de más a menos probabilidad de impago."),
                "id": formato.Column("Préstamo", help="El identificador del préstamo en el fichero."),
                "probabilidad": formato.Column("Probabilidad de impago",
                                               help="La del modelo principal, para los próximos doce meses."),
                "motivo": formato.TextColumn("Motivo principal", width="large",
                                             help="El dato que más sube el riesgo del préstamo según el modelo principal."),
                "verificar": formato.TextColumn(
                    "Verificar datos", width="medium",
                    help="Los dos modelos discrepan mucho o algún dato es poco habitual: conviene comprobar esos datos "
                         "antes de actuar.")}
    for pestana, bloque in zip(st.tabs([f"{b} ({numero((lista['bloque'] == b).sum())})" for b in BLOQUES]), BLOQUES):
        parte = lista[lista["bloque"] == bloque]
        if parte.empty:
            pestana.caption("Ningún préstamo de la lista está en este bloque.")
        else:
            clave = f"lista_{bloque}_{st.session_state.get('aperturas', 0)}"
            pestana.dataframe(parte[visibles], hide_index=True, column_config=columnas, key=clave,
                              selection_mode="single-row", on_select=lambda c=clave, d=parte: abrir_elegido(c, d))
    st.caption("Marque la casilla de una fila para abrir ese préstamo en «Evaluación de préstamo».")
    resto = [c for c in lista if c not in visibles + ["bloque", "xgboost", "logit", "tramo", "ponderado", "mora_actual",
                                                       "saldo"]]  # también las columnas propias del fichero
    descarga = lista[["puesto", "id", "bloque", "xgboost", "logit", "tramo", "ponderado", "verificar", "motivo",
                      "mora_actual", "saldo", *resto]].rename(columns={
        "xgboost": "probabilidad_principal", "logit": "probabilidad_contraste", "ponderado": "saldo_ponderado",
        "verificar": "verificar_datos", "motivo": "motivo_principal"})
    st.download_button("Descargar la lista completa", descarga.to_csv(index=False), "lista_de_revision.csv",
                       help="Con todos los datos de cada préstamo, el tramo de riesgo y la cifra del modelo de contraste.")

    with st.expander("Simulación de coste y ahorro (opcional)"):
        a, b = st.columns(2)
        coste = a.number_input("Coste de revisar un préstamo, $", 1, 10_000, 100, 10, help=(
            "Un supuesto suyo: lo que cuesta revisar un préstamo. La simulación es una cuenta con sus supuestos, no un "
            "resultado medido."))
        ahorro = b.number_input("Pérdida evitada por impago detectado, $", 1, 500_000, 2_000, 500, help=(
            "Un supuesto suyo: lo que se evita al actuar a tiempo sobre un préstamo que iba a impagar. Detectar un "
            "impago no es evitarlo."))
        corte = coste / ahorro  # revisar un préstamo compensa si su probabilidad pasa del coste entre la pérdida evitada
        beneficio = (probabilidad * ahorro - coste).cumsum()
        neto = beneficio.iloc[revisar - 1]
        # Cuánto puede moverse el ahorro por puro azar si las probabilidades fueran exactas (dos desviaciones típicas).
        azar = 2 * ahorro * np.sqrt((probabilidad * (1 - probabilidad)).head(revisar).sum())
        st.write(f"Con estos supuestos compensaría revisar {numero((probabilidad >= corte).sum())} préstamos, los de "
                 f"probabilidad de impago de al menos el {coma(f'{corte:.1%}')}. Con los {numero(revisar)} que puede "
                 f"revisar el equipo, el ahorro estimado sería de unos {numero(round(neto, -2))} dólares; solo por "
                 f"azar, podría quedar entre {numero(round(neto - azar, -2))} y {numero(round(neto + azar, -2))}.")
        curva = pd.DataFrame({"revisados": range(1, len(todos) + 1), "ahorro": beneficio})
        dibujar(alt.layer(
            alt.Chart(curva).mark_line(color="#D95319").encode(
                x=alt.X("revisados", title="Préstamos revisados, de más a menos riesgo"),
                y=alt.Y("ahorro", title="Ahorro neto estimado, $"),
                tooltip=[alt.Tooltip("revisados", title="Préstamos revisados"),
                         alt.Tooltip("ahorro", title="Ahorro neto estimado, $", format=",.0f")]),
            alt.Chart(pd.DataFrame({"revisados": [revisar]})).mark_rule(strokeDash=[4, 4], color="#808080").encode(
                x="revisados"),
        ).properties(height=230))
        st.caption("La línea de puntos marca los préstamos que puede revisar el equipo.")


def pagina_prestamo():
    st.title("Evaluación de préstamo")
    st.write("Probabilidad de que un préstamo impague en los próximos doce meses, y qué la explica.")
    if sin_modelos():
        return
    categorias, valores, ejemplos = cargar()[2], {}, dict(EJEMPLOS)
    if "de_la_lista" in st.session_state:  # el que se abrió desde «Evaluación de cartera» va el primero
        traido = dict(st.session_state["de_la_lista"])
        ejemplos = {f"De la lista: puesto {traido.pop('puesto')}": traido, **ejemplos}
    elegido = st.selectbox("Préstamo", list(ejemplos),
                           help="Préstamos inventados o el abierto desde la lista. Se puede cambiar cualquier dato.")
    with st.expander("Datos del préstamo", expanded=True):
        columnas = st.columns(3)
        for i, variable in enumerate(COLUMNAS):
            # La clave lleva el préstamo: al elegir otro, los campos se rellenan con sus valores.
            columna, valor, clave = columnas[i % 3], ejemplos[elegido][variable], f"{variable}_{elegido}"
            opciones = categorias.get(variable) or list(ETIQUETAS.get(variable, {}))
            if opciones:
                valor = valor if valor in opciones else "sin dato" if "sin dato" in opciones else opciones[0]
                textos = {codigo: ETIQUETAS.get(variable, {}).get(codigo, codigo) for codigo in opciones}
                valores[variable] = columna.selectbox(NOMBRES[variable], opciones, opciones.index(valor), key=clave,
                                                      format_func=textos.get)
            elif variable == "tipo_inicial":  # un dato que falta se deja vacío, como en la lista
                valores[variable] = columna.number_input(NOMBRES[variable], 0.0, 20.0, None if falta(valor) else
                                                         float(valor), 0.125, "%.3f", key=clave, placeholder="sin dato")
            else:  # el resto son números enteros
                valores[variable] = columna.number_input(NOMBRES[variable], 0, None, None if falta(valor) else int(valor),
                                                         key=clave, placeholder="sin dato")

    st.subheader("¿Y si…?")
    st.caption("Mueva un dato y compare con el préstamo de arriba.")
    escenario = dict(valores)
    sitios = [st.columns(3)[0], *st.expander("Otros datos, para entender el modelo").columns(3)]
    for sitio, (variable, (minimo, maximo, paso)) in zip(sitios, MANDOS.items()):
        if valores[variable] is None:
            sitio.caption(f"{NOMBRES[variable]}: sin dato")
            continue
        # El deslizador empieza en el valor del préstamo; si cae fuera de su rango, el rango se amplía.
        base = type(minimo)(valores[variable])
        escenario[variable] = sitio.slider(NOMBRES[variable], min(minimo, base), max(maximo, base), base, paso,
                                           key=f"{variable}_{base}_{elegido}", help=(
                                               "Enseña cómo cambia la cifra del modelo, no qué pasaría en la realidad: "
                                               "en una cartera real estos datos cambian juntos."))
    # Un préstamo con k cuotas impagadas hoy ha tenido al menos k en el último año.
    if escenario["mora_actual"] != valores["mora_actual"]:
        for variable in ["mora_maxima_12m", "meses_en_mora_12m"]:
            escenario[variable] = max(valores[variable] or 0, escenario["mora_actual"])

    tasa, de_donde = tasa_elegida()  # la misma tasa prevista que se eligió en «Evaluación de cartera»
    resultado, aportaciones = puntuar(pd.DataFrame([valores, escenario]), tasa / 100)
    cambia = escenario != valores
    for columna, modelo in zip(st.columns(2), MODELO):
        antes, despues = resultado[modelo]
        columna.metric(MODELO[modelo], coma(f"{despues:.1%}"), delta_color="inverse", help=AYUDA_MODELO[modelo],
                       delta=coma(f"{(despues - antes) * 100:+.1f} puntos frente a {antes:.1%}") if cambia else None)
    st.caption(coma(f"Calculado con una tasa de impago prevista para el año del {tasa:.2f}%")
               + (" (escrita a mano)" if de_donde == "Otra" else f" (escenario «{de_donde.split(':')[0]}»)")
               + ", que se cambia en «Evaluación de cartera».")
    que_hacer(resultado["logit"].iloc[-1], resultado["xgboost"].iloc[-1])
    poco_habitual(resultado.iloc[-1])

    st.subheader("Qué sube y qué baja el riesgo de este préstamo")
    st.caption("Los diez datos que más pesan y por cuánto multiplica cada uno el riesgo frente a la media: 2 lo "
               "duplica; 0,5 lo reduce a la mitad.")
    aportacion = aportaciones.iloc[-1]
    aportacion = aportacion[aportacion.abs().sort_values(ascending=False).index[:10]]
    valor = dict(escenario, ltv_actual=resultado["ltv_actual"].iloc[-1])
    datos = pd.DataFrame({"variable": [dato(v, valor[v]) for v in aportacion.index],
                          "multiplica": np.exp(aportacion.to_numpy()),
                          "efecto": np.where(aportacion > 0, "Sube el riesgo", "Baja el riesgo")})
    exponente = np.log2(datos["multiplica"])  # marcas en potencias de 2, de 0,5 o menos a 2 o más
    marcas = [2.0 ** k for k in range(min(-1, int(np.floor(exponente.min()))), max(2, int(np.ceil(exponente.max())) + 1))]
    barras = alt.Chart(datos).mark_bar().encode(
        x=alt.X("multiplica", title="Multiplica el riesgo por", axis=alt.Axis(values=marcas, format=".3~r"),
                scale=alt.Scale(type="log", domain=[marcas[0], marcas[-1]])),
        x2=alt.datum(1), y=alt.Y("variable", sort=None, title=None, axis=alt.Axis(labelLimit=0)),
        color=alt.Color("efecto", title=None, legend=alt.Legend(orient="bottom", labelLimit=0),
                        scale=alt.Scale(domain=["Sube el riesgo", "Baja el riesgo"], range=["#D95319", "#0072BD"])),
        tooltip=["variable", alt.Tooltip("multiplica", title="Multiplica el riesgo por", format=".2f")])
    dibujar(alt.layer(barras, alt.Chart(datos).mark_rule(color="#404040").encode(x=alt.datum(1))))


PAGINAS = {"Evaluación de cartera": pagina_cartera, "Evaluación de préstamo": pagina_prestamo}
if __name__ == "__main__":  # Streamlit ejecuta el fichero como __main__; al importarlo, como probar_app.py, no dibuja
    if st.get_option("client.toolbarMode") != "minimal":  # sin el botón «Deploy» ni el menú de Streamlit
        st.set_option("client.toolbarMode", "minimal")
        st.rerun()  # para que se aplique ya en la primera visita
    st.set_page_config(page_title="Riesgo de impago hipotecario", layout="wide")
    # Streamlit borra el valor de un campo que no se dibuja; reescribirlo al empezar conserva la tasa elegida en
    # «Evaluación de cartera» mientras se ve la otra pantalla.
    for clave in ["escenario", "tasa_otra"]:
        if clave in st.session_state:
            st.session_state[clave] = st.session_state[clave]
    st.sidebar.title("Riesgo de impago hipotecario")
    pantalla = st.sidebar.radio("Pantalla", list(PAGINAS), label_visibility="collapsed", key="pantalla")
    ficha_del_modelo()
    PAGINAS[pantalla]()
