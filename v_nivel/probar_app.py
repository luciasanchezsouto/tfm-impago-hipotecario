"""Pruebas de la aplicación: abre las dos pantallas sin navegador y comprueba que no enseñan términos técnicos y que
sus textos son cortos, la lista y sus dos bloques, la horquilla de los impagos previstos, los escenarios de tasa, los
préstamos inventados y los avisos.

Necesita los modelos de modelos/ y las tablas de descriptivo/ y resultados/. Al final abre una copia de la aplicación
cuya tabla de tasas acaba en 2020, un año anómalo, y la misma copia sin la carpeta modelos/. La prueba de lo medido con
una lista de ese tamaño, que necesita resultados/umbrales_nivel.csv con los 18 años, dice si queda pendiente.

AppTest no sabe marcar la casilla de una fila de una tabla, que es como se abre un préstamo de la lista: marcar() le
envía lo que mandaría el navegador (véase su explicación).

    python -X utf8 v_nivel/probar_app.py
"""

import importlib.util
import io
import json
import logging
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from google.protobuf.message import Message
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

AQUI = Path(__file__).resolve().parent
APP = str(AQUI / "5_app.py")
EJEMPLO = AQUI / "ejemplo_cartera.csv"
PANTALLAS = ["Evaluación de cartera", "Evaluación de préstamo"]
# Textos que solo están en la barra lateral, en la ayuda de una cifra, de un campo o de una columna, en un
# desplegable cerrado o en un gráfico: si los encuentra, la búsqueda de términos técnicos mira también ahí.
ESCONDIDOS = {"Evaluación de cartera": ["qué predice", "más cuotas impagadas de lo habitual", "la fija el área de riesgos",
                                        "no es una garantía", "antes de actuar", "la cifra del modelo de contraste",
                                        "ahorro neto estimado"],
              "Evaluación de préstamo": ["qué predice", "segunda opinión", "no qué pasaría en la realidad",
                                         "multiplica el riesgo por"]}
# Lo que no debe verse en ningún sitio de la aplicación, tampoco en la barra lateral ni en las ayudas: nombres de
# modelos y de métodos, jerga estadística y explicaciones de cómo se validó el modelo.
TECNICOS = ["auc", "xgboost", "logit", "logístic", "logistic", "shap", "baseline", "regla", "random forest",
            "bosque aleatorio", "percentil", "desviaci", "típica", "razón", "intervalo de confianza", "validaci",
            "validó", "años de prueba", "con nivel", "sin nivel", "calibraci", "efecto causal", "binomial",
            "bernoulli", "odds", "monte carlo"]
COMPROBAR = "que un analista compruebe los datos de este préstamo"
CAPACIDAD, TASA, OTRA = "Préstamos que puede revisar el equipo", "Tasa de impago prevista para el año", \
    "Otra tasa de impago prevista, %"
CARTERA, SIMULACION = "Cartera: ejemplo de 500 préstamos inventados", "Simulación de coste y ahorro (opcional)"
PREVISTOS, EN_LISTA = "Impagos previstos en el año", "De ellos, en la lista"
SIN_HORQUILLA = ("Con esta tasa no se da horquilla, porque el acierto del modelo en los años pasados se midió con la "
                 "tasa del último año cerrado.")
VISIBLES = ["Puesto", "Préstamo", "Probabilidad de impago", "Motivo principal", "Verificar datos"]
TASAS = pd.read_csv(AQUI / "descriptivo" / "por_anio.csv").sort_values("anio")
DEFECTO = round(100 * TASAS["tasa"].iloc[-1], 2)  # el escenario base: la tasa del último año cerrado, en %
NIVEL_CSV, UMBRALES_CSV = AQUI / "resultados" / "nivel.csv", AQUI / "resultados" / "umbrales_nivel.csv"
DISCREPANCIAS = pd.read_csv(AQUI / "resultados" / "discrepancias.csv")
assert (AQUI / "modelos").exists(), "Faltan los modelos de modelos/: se crean con python v_nivel/6_nivel.py 2025."
pendiente = []  # lo que no se puede comprobar si umbrales_nivel.csv no tiene los 18 años
for ruido in ["streamlit.runtime.scriptrunner_utils.script_run_context", "streamlit.runtime.caching.cache_data_api"]:
    logging.getLogger(ruido).addFilter(lambda registro: False)  # avisos de Streamlit sin navegador, que no importan

# Las funciones de la aplicación, para probarlas aparte: al importarla no es __main__ y no dibuja nada.
especificacion = importlib.util.spec_from_file_location("aplicacion", APP)
funciones = importlib.util.module_from_spec(especificacion)
especificacion.loader.exec_module(funciones)
BLOQUES = dict(zip([True, False], funciones.BLOQUES))  # con alguna cuota impagada hoy, o ninguna
# Las columnas de la lista descargable, en su orden: las de la lista, las que no se ven y los datos del préstamo.
DESCARGA = ["puesto", "id", "bloque", "probabilidad_principal", "probabilidad_contraste", "tramo", "saldo_ponderado",
            "verificar_datos", "motivo_principal", "mora_actual", "saldo",
            *[c for c in funciones.COLUMNAS if c not in ["mora_actual", "saldo"]], "ltv_actual"]

# Lo que la aplicación da a descargar, por nombre de fichero.
descargas, guardar = {}, MemoryMediaFileStorage.load_and_get_id


def capturar(self, datos, mimetype, kind, filename=None):
    descargas[filename] = datos
    return guardar(self, datos, mimetype, kind, filename)


MemoryMediaFileStorage.load_and_get_id = capturar


def revisar(prueba, donde):
    """Sin errores de Python ni avisos de deprecación de Streamlit, que salen como avisos en la pantalla."""
    assert not prueba.exception, (donde, prueba.exception)
    assert not [w.value for w in prueba.warning if "deprecat" in w.value.lower()], donde


def abrir(pantalla, app=APP):
    """Ejecuta una pantalla de la aplicación y comprueba que no da ningún error."""
    prueba = AppTest.from_file(app, default_timeout=120)
    prueba.run()
    prueba.sidebar.radio[0].set_value(pantalla).run()
    revisar(prueba, pantalla)
    return prueba


def cadenas(mensaje):
    """Todos los textos de un elemento de Streamlit: etiquetas, ayudas, opciones, avisos, nombres y ayudas de las
    columnas y especificación de los gráficos. Sin los identificadores internos ni los datos de las tablas, que van
    en binario."""
    for campo, valor in mensaje.ListFields():
        if campo.name == "id" or campo.name.endswith("_id"):
            continue
        for v in [valor] if isinstance(valor, (str, bytes, int, float, Message)) else valor:  # uno o una lista
            if isinstance(v, str):
                yield v
            elif isinstance(v, Message):
                yield from cadenas(v)


def sin_escapes(texto):
    """Las columnas y los gráficos llegan en JSON, con las tildes escritas como \\u00e9: se leen como texto normal."""
    try:
        return json.dumps(json.loads(texto), ensure_ascii=False)
    except ValueError:
        return texto


def todo_el_texto(prueba):
    """Todo el texto de la pantalla y de la barra lateral, también lo que solo se ve en las ayudas, en minúsculas."""
    return " ".join(sin_escapes(t) for zona in [prueba.main, prueba.sidebar] for nodo in zona
                    if isinstance(nodo.proto, Message) for t in cadenas(nodo.proto)).lower()


def frases(texto):
    """Cuántas frases tiene un texto."""
    return len(re.findall(r"[.!?…](?=\s|$)", texto.strip()))


def graficos(prueba):
    """La especificación de los gráficos de Altair de la pantalla, como texto."""
    especificaciones = [json.dumps(json.loads(e.proto.spec), ensure_ascii=False) for e in prueba.get("vega_lite_chart")]
    assert especificaciones, "la pantalla no tiene gráficos"
    return " ".join(especificaciones)


def porcentaje(texto):
    return float(texto.rstrip("% ").replace(",", "."))


def entero(texto):
    """El último número de un texto como «unos 1.234»."""
    return int(texto.split()[-1].replace(".", ""))


def widget(lista, etiqueta):
    return next(w for w in lista if w.label == etiqueta)


def mostrada(prueba):
    """La lista que se ve, con sus dos bloques juntos y en el orden del puesto; None si no hay lista."""
    partes = [d.value.assign(bloque=t.label.split(" (")[0]) for t in prueba.tabs for d in t.dataframe]
    return pd.concat(partes).sort_values("puesto").reset_index(drop=True) if partes else None


def columnas(tabla):
    """Nombre en pantalla y ayuda de cada columna que se ve de una tabla de la lista, en su orden."""
    formato = json.loads(tabla.proto.columns)
    return {formato[c]["label"]: formato[c].get("help") for c in tabla.value.columns}


def subir(prueba, nombre, cartera, capacidad=None):
    """Sube una cartera a «Evaluación de cartera», fija si se pide la capacidad del equipo y devuelve la lista."""
    prueba.file_uploader[0].set_value((nombre, cartera.to_csv(index=False).encode(), "text/csv")).run()
    if capacidad and not prueba.error:
        widget(prueba.number_input, CAPACIDAD).set_value(capacidad).run()
    revisar(prueba, nombre)
    return mostrada(prueba)


def descargada():
    """La última lista descargable, como tabla."""
    return pd.read_csv(io.BytesIO(descargas["lista_de_revision.csv"]))


def marcar(prueba, bloque, fila):
    """Marca la casilla de una fila de la pestaña de ese bloque y devuelve la fila tal como se veía.

    AppTest no tiene cómo marcar una fila de una tabla. Se le envía lo mismo que mandaría el navegador al marcarla: el
    estado de la tabla con esa fila elegida, junto con el de los demás campos de la pantalla, mediante AppTest._run,
    una función interna de Streamlit 1.58 (la versión de requirements.txt). Así corre lo mismo que en el navegador: la
    tabla llama a abrir_elegido, que lleva el préstamo a abrir_prestamo, y la aplicación vuelve a ejecutarse en
    «Evaluación de préstamo». Que la casilla se vea y se pueda pulsar solo se comprueba en un navegador."""
    tabla = next(t for t in prueba.tabs if t.label.startswith(bloque)).dataframe[0]
    elegida = tabla.value.iloc[fila]
    estados = prueba._tree.get_widget_states()
    estados.widgets.add(id=tabla.proto.id).string_value = json.dumps(
        {"selection": {"rows": [fila], "columns": [], "cells": []}})
    prueba._run(estados)
    revisar(prueba, f"{bloque}, fila {fila}")
    return elegida


def esperados(tasa):
    """Probabilidades de impago del modelo principal en la cartera de ejemplo, de mayor a menor, con una tasa en %."""
    cartera = funciones.cartera_ejemplo().assign(id=range(1, 501))
    return funciones.puntuar(cartera, tasa / 100)[0]["xgboost"].sort_values(ascending=False).reset_index(drop=True)


def horquilla_exacta(probabilidades, cocientes, copias=1):
    """La horquilla de los impagos calculada aparte y sin sorteos: con cada cociente c, la distribución exacta del
    número de impagos con las probabilidades corregidas, p / (p + c (1 - p)), sumando préstamo a préstamo; después, la
    media de las de todos los cocientes, a partes iguales, y el primer número de impagos cuya probabilidad acumulada
    llega al 10 % y al 90 %. Con copias, la de esa cartera repetida tantas veces: su distribución se eleva a esa
    potencia con la transformada de Fourier."""
    total = len(probabilidades) * copias
    mezcla = np.zeros(total + 1)
    for c in cocientes:
        q = probabilidades / (probabilidades + c * (1 - probabilidades))
        distribucion = np.array([1.0])
        for qi in q:  # el préstamo impaga (un impago más) o no
            distribucion = np.append(distribucion * (1 - qi), 0) + np.append(0, distribucion * qi)
        if copias > 1:
            largo = 2 ** int(np.ceil(np.log2(total + 1)))
            distribucion = np.fft.irfft(np.fft.rfft(distribucion, largo) ** copias, largo)[:total + 1]
        mezcla += distribucion / len(cocientes)
    acumulada = np.cumsum(mezcla)
    return int(np.searchsorted(acumulada, 0.1)), int(np.searchsorted(acumulada, 0.9))


# Las dos pantallas: ningún término técnico, ni en las ayudas, ni en los desplegables cerrados, ni en la barra
# lateral; una frase de introducción y como mucho una frase por pie; la ficha lateral, breve.
for pantalla in PANTALLAS:
    prueba = abrir(pantalla)
    texto = todo_el_texto(prueba)
    tecnicos = [n for n in TECNICOS if re.search(rf"\b{re.escape(n)}", texto)]
    assert not tecnicos and "Ficha del modelo" in [s.value for s in prueba.sidebar.subheader], (pantalla, tecnicos)
    assert all(t in texto for t in ESCONDIDOS[pantalla]), [t for t in ESCONDIDOS[pantalla] if t not in texto]
    assert prueba.title[0].value == pantalla and prueba.sidebar.radio[0].options == PANTALLAS
    introduccion = prueba.main.markdown[0].value
    assert frases(introduccion) == 1 and len(introduccion) < 100, introduccion
    assert all(frases(c.value) <= 1 for c in prueba.main.caption), [c.value for c in prueba.main.caption]
    ficha = " ".join(m.value for m in prueba.sidebar.markdown)
    assert "tres cuotas impagadas o cerrar el préstamo con pérdida" in ficha, ficha
    assert "Dos modelos, el principal y el de contraste, entrenados con hipotecas de Freddie Mac de 2000 a 2024" in ficha
    assert "límites de uso" not in texto and "cómo se comprobó" not in texto and not prueba.sidebar.expander
    assert not prueba.sidebar.info and len(ficha) < 400, ficha
    graficos(prueba)
    print(f"abre: {pantalla}; ni un término técnico; introducción: «{introduccion}»")
print("ficha:", ficha)

# Funciones puras: escenarios de tasa y año anómalo.
assert list(funciones.escenarios(TASAS).values()) == [DEFECTO, 0.69, 2.54], funciones.escenarios(TASAS)
assert funciones.anio_anomalo(TASAS) is None
anio, tasa, mediana = funciones.anio_anomalo(TASAS[TASAS["anio"] <= 2020])  # la serie cortada en 2020
assert anio == 2020 and round(tasa, 4) == 0.0365 and round(mediana, 4) == 0.0047, (anio, tasa, mediana)
print(f"año anómalo: no con la serie entera; con la serie cortada en 2020, sí ({tasa:.2%} frente a una mediana de "
      f"{mediana:.2%})".replace(".", ","))

lista = abrir("Evaluación de cartera")
nivel = pd.read_csv(NIVEL_CSV) if NIVEL_CSV.exists() else None
# La pantalla, sin bloques numerados ni botones: arriba, la cartera y la tasa prevista en dos desplegables
# cerrados; un campo, dos cifras y la lista de revisión; al final, la simulación, en otro desplegable cerrado.
assert [(e.label, e.proto.expanded) for e in lista.expander] == [(CARTERA, False), (TASA, False),
                                                                 (SIMULACION, False)], lista.expander
assert [s.value for s in lista.main.subheader] == ["Lista de revisión"] and not lista.button
cartera = widget(lista.expander, CARTERA)
assert [f.label for f in cartera.file_uploader] == ["Cartera"]
assert [d.proto.label for d in cartera.get("download_button")] == ["Descargar el ejemplo"]
assert [d.proto.label for d in lista.get("download_button")] == ["Descargar el ejemplo", "Descargar la lista completa"]
assert [s.label for s in widget(lista.expander, TASA).selectbox] == [TASA]
assert [n.label for n in lista.number_input] == [CAPACIDAD, *[n.label for n in widget(lista.expander, SIMULACION)
                                                              .number_input]], [n.label for n in lista.number_input]
assert [m.label for m in lista.metric] == [PREVISTOS, EN_LISTA], [m.label for m in lista.metric]
assert "Marque la casilla de una fila para abrir ese préstamo en «Evaluación de préstamo»." in \
    [c.value for c in lista.main.caption]
# El ejemplo que se descarga: los 500 préstamos inventados, con las columnas que debe tener una cartera.
ejemplo_500 = pd.read_csv(io.BytesIO(descargas["ejemplo_cartera_500.csv"]))
assert ejemplo_500.shape == (500, len(funciones.COLUMNAS)) and list(ejemplo_500) == funciones.COLUMNAS

# Cartera de ejemplo: 500 préstamos inventados; la lista son los 50 primeros (el 10 %), de más a menos riesgo,
# con la probabilidad del modelo principal.
tabla, capacidad, todos = mostrada(lista), widget(lista.number_input, CAPACIDAD), esperados(DEFECTO)
assert capacidad.value == 50 and len(tabla) == 50 and list(tabla["puesto"]) == list(range(1, 51)), capacidad.value
assert list(tabla["probabilidad"].str.strip()) == [funciones.coma(f"{100 * p:.1f}%") for p in todos.head(50)]
# Columnas visibles, en las dos pestañas: las cinco en su orden, cada una con su ayuda. El tramo, las cuotas
# impagadas, el saldo y la probabilidad del modelo de contraste no se ven: van en la lista descargable.
for pestana in lista.tabs:
    ayudas = columnas(pestana.dataframe[0])
    assert list(ayudas) == VISIBLES and all(ayudas.values()), ayudas
assert "conviene comprobar esos datos antes de actuar" in ayudas["Verificar datos"], ayudas
assert tabla["motivo"].str.len().min() > 0
assert tabla["verificar"].str.fullmatch(r"(Los dos modelos discrepan)?(; )?(Dato poco habitual: .+)?").all()
# La lista descargable: todas las columnas en su orden, los mismos préstamos y bloques que en pantalla, el tramo,
# el saldo ponderado y la probabilidad del modelo de contraste.
csv = descargada()
p, c = csv["probabilidad_principal"], csv["probabilidad_contraste"]
assert list(csv.columns) == DESCARGA, list(csv.columns)
assert len(csv) == 50 and list(csv["id"]) == list(tabla["id"]) and list(csv["bloque"]) == list(tabla["bloque"])
assert (csv["bloque"] == (csv["mora_actual"] >= 1).map(BLOQUES)).all() and ((p - todos.head(50)).abs() < 1e-6).all()
pestanas = [t.label for t in lista.tabs]
assert pestanas == [f"{b} ({(csv['bloque'] == b).sum()})" for b in funciones.BLOQUES], pestanas
tramos = pd.cut(p / (DEFECTO / 100), [0, 1, 5, 20, float("inf")], right=False,
                labels=["Bajo", "Medio", "Alto", "Muy alto"])
assert (csv["tramo"] == tramos.astype(str)).all() and ((csv["saldo_ponderado"] - p * csv["saldo"]).abs() <= 1).all()
discrepan = (pd.concat([p, c], axis=1).max(axis=1) >= 0.05) & ((p >= 3 * c) | (c >= 3 * p))
assert (csv["verificar_datos"].fillna("").str.contains("discrepan") == discrepan).all()
assert (csv["motivo_principal"].fillna("") == tabla["motivo"]).all()
motivos = tabla["motivo"].str.split(" (", regex=False).str[0].value_counts()
print(f"lista: 50 préstamos ordenados, en {pestanas[0]} y {pestanas[1]}; verificar datos en "
      f"{(tabla['verificar'] != '').sum()}; motivos: {'; '.join(f'{m}: {n}' for m, n in motivos.items())}")
print("   columnas en pantalla:", " | ".join(VISIBLES))
print(f"   lista descargable: {len(DESCARGA)} columnas; tramos {csv['tramo'].value_counts().to_dict()}")
lista_ejemplo = tabla

# Las dos cifras: impagos del año (suma de probabilidades), con su horquilla debajo, y cuántos de ellos están en la
# lista, con su parte del total; lo que lo explica va en la ayuda de cada cifra, en lenguaje llano.
previstos, en_lista = widget(lista.metric, PREVISTOS), widget(lista.metric, EN_LISTA)
assert entero(previstos.value) == round(todos.sum()) and entero(en_lista.value) == round(todos.head(50).sum())
assert en_lista.delta == f"{todos.head(50).sum() / todos.sum():.0%} del total".replace("%", " %"), en_lista.delta
assert "en 8 de cada 10 casos" in previstos.help and "el azar de una cartera de este tamaño" in previstos.help
assert "No es una garantía" in previstos.help and not re.search(r"simul|sorte|percentil", previstos.help.lower())
if nivel is not None:
    # La horquilla junta lo que se desvió el modelo en los 17 años de prueba desde 2009 y el azar de una cartera de
    # este tamaño. Se recalcula aparte, sin sorteos, y debe salir igual o a uno de distancia (los sorteos son 34.000);
    # también con los préstamos sorteados por tramos, como en las carteras grandes. Contiene la cifra central, es más
    # ancha que la que solo contaba lo que se desvió el modelo y sale siempre igual.
    anios = nivel[(nivel["modelo"] == "xgboost") & (nivel["anio"] >= 2009)].set_index("anio")["calibracion_ultimo"]
    cocientes, sin_cache, probabilidades = tuple(anios), funciones.horquilla_impagos.__wrapped__, todos.to_numpy()
    assert len(cocientes) == 17 and anios.index.max() == nivel["anio"].max(), anios
    inicio = time.perf_counter()
    bajo, alto = sin_cache(probabilidades, cocientes)
    tiempo = time.perf_counter() - inicio
    assert previstos.delta == f"entre {bajo} y {alto}", (previstos.delta, bajo, alto)
    assert f"(de 2009 a {nivel['anio'].max()})" in previstos.help, previstos.help
    exacta = horquilla_exacta(probabilidades, cocientes)
    por_tramos = sin_cache(probabilidades, cocientes, exacto=0)
    assert all(abs(h - e) <= 1 for h, e in [*zip((bajo, alto), exacta), *zip(por_tramos, exacta)]), (exacta, por_tramos)
    p10, p90 = anios.quantile([0.1, 0.9])  # la horquilla antigua: solo lo que se desvió el modelo
    antigua = (round(todos.sum() / p90), round(todos.sum() / p10))
    assert bajo <= round(todos.sum()) <= alto and alto - bajo > antigua[1] - antigua[0], (bajo, alto, antigua)
    assert sin_cache(probabilidades, cocientes) == (bajo, alto)  # la semilla es fija
    # Con 50.000 préstamos (la cartera de ejemplo cien veces), el azar pesa menos: la horquilla, en proporción a los
    # impagos previstos, es más estrecha. Va por tramos; sale como la exacta, salvo el 1 % por los sorteos, y rápida.
    grande = np.tile(probabilidades, 100)
    inicio = time.perf_counter()
    bajo_g, alto_g = sin_cache(grande, cocientes)
    tiempo_g = time.perf_counter() - inicio
    exacta_g = horquilla_exacta(probabilidades, cocientes, copias=100)
    assert all(abs(h - e) <= 1 + e / 100 for h, e in zip((bajo_g, alto_g), exacta_g)), ((bajo_g, alto_g), exacta_g)
    assert bajo_g <= round(grande.sum()) <= alto_g and (alto_g - bajo_g) / grande.sum() < (alto - bajo) / todos.sum()
    assert tiempo < 2 and tiempo_g < 2, (tiempo, tiempo_g)
    print(funciones.coma(f"horquilla con 500 préstamos: entre {bajo} y {alto}, en {tiempo:.2f} s (sin sorteos, entre "
                         f"{exacta[0]} y {exacta[1]}; por tramos, entre {por_tramos[0]} y {por_tramos[1]}; la antigua, "
                         f"entre {antigua[0]} y {antigua[1]})"))
    print(funciones.coma(f"   con 50000: entre {bajo_g} y {alto_g}, en {tiempo_g:.2f} s (sin sorteos, entre "
                         f"{exacta_g[0]} y {exacta_g[1]}); de ancho, {(alto - bajo) / todos.sum():.2f} y "
                         f"{(alto_g - bajo_g) / grande.sum():.2f} veces los previstos con 500 y con 50000"))
print(f"cifras: {' | '.join(f'{m.label}: {m.value} ({m.delta})' for m in lista.metric)}")
print("   ayuda de los impagos del año:", previstos.help)
print("   ayuda de los de la lista:", en_lista.help)
if UMBRALES_CSV.exists():  # lo medido en los años de prueba, con los modelos con nivel
    umbrales = pd.read_csv(UMBRALES_CSV)
    medido = umbrales[(umbrales["modelo"] == "xgboost") & (umbrales["revisado"] == 10)]
    recall = medido["sensibilidad"]
    assert (f"De {medido['anio'].min()} a {medido['anio'].max()}, una lista del 10% de la cartera de Freddie Mac "
            f"encontró entre el {recall.min():.0%} y el {recall.max():.0%} de los impagos del año "
            f"({recall.mean():.0%} de media)").replace("%", " %") in en_lista.help, en_lista.help
    if umbrales["anio"].nunique() < 18:
        pendiente.append(f"Lo medido de la lista: umbrales_nivel.csv tiene {umbrales['anio'].nunique()} de 18 años")
else:
    pendiente.append("Lo medido de la lista: falta umbrales_nivel.csv")

# Escenarios de tasa: cambiarla no cambia el orden de la lista, sí los impagos previstos. La horquilla solo sale con
# la base, la tasa con que se midió cuánto se desvió el modelo; con las demás, la ayuda dice por qué no sale.
escenario = widget(lista.selectbox, TASA)
opciones = [f"Base: la del último año cerrado ({DEFECTO:.2f} %, {TASAS['anio'].iloc[-1]})".replace(".", ","),
            "Prudente: la mayor de los cinco últimos años (0,69 %, 2021)",
            "Severa: la de 2009, el peor año de la crisis financiera (2,54 %)", "Otra"]
assert escenario.options == opciones and escenario.value == opciones[0], escenario.options
assert not any("anómalo" in w.value for w in lista.warning)  # con los datos actuales no salta
cifras = {}
for opcion, tasa in zip(opciones[:3], [DEFECTO, 0.69, 2.54]):
    widget(lista.selectbox, TASA).set_value(opcion).run()
    revisar(lista, opcion)
    assert list(mostrada(lista)["id"]) == list(tabla["id"]), "cambiar la tasa no debe cambiar el orden"
    cifra = widget(lista.metric, PREVISTOS)
    cifras[opcion.split(":")[0]] = f"{cifra.value} ({cifra.delta or 'sin horquilla'})"
    assert entero(cifra.value) == round(esperados(tasa).sum()), (opcion, cifras)
    if nivel is not None and opcion.startswith("Base"):
        assert cifra.delta == previstos.delta and SIN_HORQUILLA not in cifra.help, (opcion, cifra.delta)
    elif nivel is not None:
        assert not cifra.delta and SIN_HORQUILLA in cifra.help and "8 de cada 10" not in cifra.help, opcion
print("escenarios de tasa (impagos previstos en el año):", cifras)
print("   ayuda sin la base:", cifra.help)
# «Otra»: dos valores seguidos se quedan los dos, y el último pasa a «Evaluación de préstamo» y se conserva al volver.
widget(lista.selectbox, TASA).set_value("Otra").run()
assert widget(lista.number_input, OTRA).value == DEFECTO
for tasa in [1.0, 1.5]:
    widget(lista.number_input, OTRA).set_value(tasa).run()
    revisar(lista, tasa)
    assert widget(lista.number_input, OTRA).value == tasa, (tasa, widget(lista.number_input, OTRA).value)
    cifra = widget(lista.metric, PREVISTOS)
    assert entero(cifra.value) == round(esperados(tasa).sum()) and not cifra.delta
    assert nivel is None or SIN_HORQUILLA in cifra.help, cifra.help
lista.sidebar.radio[0].set_value("Evaluación de préstamo").run()
revisar(lista, "Evaluación de préstamo con 1,5 %")
assert any("del 1,50 % (escrita a mano), que se cambia en «Evaluación de cartera»." in c.value
           for c in lista.caption), [c.value for c in lista.caption]
con_otra = porcentaje(lista.metric[0].value)
lista.sidebar.radio[0].set_value("Evaluación de cartera").run()
assert widget(lista.selectbox, TASA).value == "Otra" and widget(lista.number_input, OTRA).value == 1.5
widget(lista.selectbox, TASA).set_value(opciones[0]).run()
cifra = widget(lista.metric, PREVISTOS)
assert (cifra.value, cifra.delta, cifra.help) == (previstos.value, previstos.delta, previstos.help)
print("tasa escrita a mano: dos cambios seguidos (1 % y 1,5 %) se quedan, sin horquilla; el último pasa a «Evaluación "
      "de préstamo» y se conserva al volver; con la base, vuelve la misma horquilla")

# Capacidad del equipo: 25 préstamos.
widget(lista.number_input, CAPACIDAD).set_value(25).run()
revisar(lista, "capacidad 25")
en_lista = widget(lista.metric, EN_LISTA)
assert len(mostrada(lista)) == 25 and entero(en_lista.value) == round(todos.head(25).sum())
assert en_lista.delta == f"{todos.head(25).sum() / todos.sum():.0%} del total".replace("%", " %"), en_lista.delta
assert len(descargada()) == 25
# Simulación de coste y ahorro: en un desplegable cerrado; los supuestos de quien la usa, en la ayuda.
simulacion = lambda: widget(lista.expander, SIMULACION)
cuenta = lambda: next(m.value for m in simulacion().markdown if m.value.startswith("Con estos supuestos"))
supuestos = [n.help for n in simulacion().number_input]
assert not simulacion().proto.expanded and all(a.startswith("Un supuesto suyo") for a in supuestos), supuestos
assert any("no un resultado medido" in a for a in supuestos) and any("no es evitarlo" in a for a in supuestos)
assert f"compensaría revisar {(todos >= 0.05).sum()} préstamos" in cuenta() and "Con los 25 que" in cuenta(), cuenta()
print("simulación:", cuenta())
widget(simulacion().number_input, "Pérdida evitada por impago detectado, $").set_value(4000).run()
assert f"compensaría revisar {(todos >= 0.025).sum()} préstamos" in cuenta(), cuenta()

# Abrir un préstamo marcando la casilla de su fila: el segundo de «Al corriente con alerta» se abre en
# «Evaluación de préstamo» con sus datos y la misma probabilidad, y al volver la lista no tiene fila marcada.
csv, claves = descargada(), [d.key for t in lista.tabs for d in t.dataframe]
fila = marcar(lista, "Al corriente con alerta", 1)
datos = csv[csv["puesto"] == fila["puesto"]].iloc[0]
assert lista.title[0].value == "Evaluación de préstamo" and datos["bloque"] == "Al corriente con alerta"
assert widget(lista.selectbox, "Préstamo").value == f"De la lista: puesto {fila['puesto']}"
assert lista.metric[0].value.strip() == fila["probabilidad"].strip(), (lista.metric[0].value, fila["probabilidad"])
assert all(widget(lista.number_input, funciones.NOMBRES[v]).value == datos[v]
           for v in ["fico", "dti", "importe", "saldo", "antiguedad", "mora_actual"])
assert widget(lista.selectbox, "Estado").value == datos["estado"]
print(f"de la lista a «Evaluación de préstamo»: al marcar la fila del puesto {fila['puesto']} se abre con "
      f"{lista.metric[0].value}, como en la lista")
lista.sidebar.radio[0].set_value("Evaluación de cartera").run()
revisar(lista, "vuelta a la lista")
nuevas = [d.key for t in lista.tabs for d in t.dataframe]
assert len(nuevas) == 2 and not set(nuevas) & set(claves), (claves, nuevas)
assert all(not lista.session_state[k]["selection"]["rows"] for k in nuevas)

# El ejemplo descargado, subido, da la misma lista que la cartera de ejemplo.
ejemplo = pd.read_csv(EJEMPLO)
tabla = subir(lista, "ejemplo_cartera_500.csv", ejemplo_500, 50)
assert tabla.equals(lista_ejemplo) and lista.expander[0].label == "Cartera: ejemplo_cartera_500.csv"
# CSV sin dos columnas: mensaje claro con las que faltan y sin error de Python.
assert subir(lista, "malo.csv", ejemplo.drop(columns=["fico", "dti"])) is None and "fico, dti" in lista.error[0].value
assert lista.expander[0].label == "Cartera: malo.csv"
print("CSV sin columnas:", lista.error[0].value)
# Los cinco préstamos de ejemplo_cartera.csv: con la capacidad por defecto, uno; el de más riesgo lleva cuotas
# impagadas, y el bloque de los préstamos al corriente queda vacío.
tabla = subir(lista, "ejemplo.csv", ejemplo)
assert len(tabla) == 1 and tabla.iloc[0]["bloque"] == "Con cuotas impagadas" and descargada()["mora_actual"][0] >= 1
assert lista.tabs[1].label == "Al corriente con alerta (0)"
assert any("Ningún préstamo" in c.value for c in lista.tabs[1].caption) and not lista.tabs[1].dataframe
# Con capacidad 5, los cinco; el identificador y las columnas propias del fichero llegan a la lista descargable.
propia = ejemplo.assign(id=[f"H-{i}" for i in range(1, 6)], oficina=["Norte", "Sur", "Este", "Oeste", "Centro"])
tabla = subir(lista, "propia.csv", propia, 5)
csv = descargada()
assert len(tabla) == 5 and tabla.iloc[0]["bloque"] == "Con cuotas impagadas", tabla
assert list(csv["id"]) == list(tabla["id"]) and set(csv["id"]) == set(propia["id"]), csv["id"]
assert set(DESCARGA) <= set(csv.columns), list(csv.columns)
assert list(csv["oficina"]) == list(propia.set_index("id").loc[csv["id"], "oficina"]), csv["oficina"]
# Un solo préstamo, un préstamo que ya está en impago y un FICO con el código de dato ausente.
assert len(subir(lista, "uno.csv", ejemplo.head(1))) == 1
tabla = subir(lista, "raro.csv", ejemplo.assign(mora_actual=[0, 3, 0, 0, 0], fico=[9999, 640, 700, 680, 790]), 4)
assert len(tabla) == 4 and any("ya están en impago" in aviso.value for aviso in lista.warning), lista.warning
assert descargada()["fico"].isna().sum() == 1
# Un importe de cero no rompe la lista; uno de más de un millón se escribe con sus cifras y se pide verificarlo.
tabla = subir(lista, "extremos.csv", ejemplo.assign(importe=[0, 210000, 450000, 180000, 2500000]), 5)
csv = descargada()
assert len(tabla) == 5 and not tabla["motivo"].str.contains("e+", regex=False).any(), tabla["motivo"]
assert "Dato poco habitual: Importe concedido" in csv.loc[csv["importe"] == 2500000, "verificar_datos"].iloc[0]
print("CSV subido: el ejemplo descargado da la misma lista; cinco préstamos ordenados; uno solo, uno en impago, un "
      "dato ausente, un importe de cero o enorme y las columnas propias del fichero se tratan bien")
# Un préstamo con datos vacíos da en «Evaluación de préstamo» la misma probabilidad que en la lista.
tabla = subir(lista, "vacio.csv", ejemplo.head(1).assign(fico=[9999], dti=[999]))
fila = marcar(lista, tabla.iloc[0]["bloque"], 0)
assert lista.title[0].value == "Evaluación de préstamo" and widget(lista.selectbox, "Préstamo").value == \
    "De la lista: puesto 1"
assert lista.metric[0].value.strip() == fila["probabilidad"].strip(), lista.metric[0].value
assert widget(lista.number_input, "Puntuación FICO").value is None
assert widget(lista.number_input, "Deuda sobre ingresos (DTI), %").value is None
print("datos vacíos: la misma probabilidad en la lista y en «Evaluación de préstamo»:", lista.metric[0].value)

app = abrir("Evaluación de préstamo")
# Formulario: enteros sin decimales y textos en lugar de códigos; el modelo sigue recibiendo el código.
fico = widget(app.number_input, "Puntuación FICO")
assert fico.value == 760 and isinstance(fico.value, int), fico.value
ocupacion, inmueble = widget(app.selectbox, "Ocupación"), widget(app.selectbox, "Tipo de inmueble")
assert ocupacion.value == "P" and "Vivienda habitual" in ocupacion.options, ocupacion.options
assert inmueble.value == "SF" and "Unifamiliar" in inmueble.options, inmueble.options

# Préstamo A del ejemplo (al corriente, FICO 760): riesgo bajo con los dos modelos, y sin aviso. Qué es cada
# modelo va en la ayuda de su cifra.
xgboost, logit, cero = porcentaje(app.metric[0].value), porcentaje(app.metric[1].value), [m.value for m in app.metric]
assert [m.label for m in app.metric] == ["Modelo principal", "Modelo de contraste"] and 0 < logit < 1 and 0 < xgboost < 1
assert "ordena la lista" in app.metric[0].help and "segunda opinión" in app.metric[1].help
assert not any(COMPROBAR in a.value for a in [*app.warning, *app.info])
assert any(f"del {DEFECTO:.2f} % (escenario «Base»)".replace(".", ",") in c.value for c in app.caption)
assert con_otra > xgboost  # con una tasa prevista del 1,5 %, A tenía más riesgo
# ¿Y si…?: a la vista, solo las cuotas impagadas; FICO, LTV y tipo de interés, en el desplegable. Que describe el
# modelo y no la realidad va en la ayuda de cada mando.
otros = widget(app.expander, "Otros datos, para entender el modelo")
assert {s.label for s in otros.slider} == {"Puntuación FICO", "LTV al conceder, %", "Tipo de interés, %"}
assert [s.label for s in app.slider if s.label not in {s.label for s in otros.slider}] == ["Cuotas impagadas hoy"]
assert all("no qué pasaría en la realidad" in s.help for s in app.slider)
# ¿Y si tiene una cuota impagada? El riesgo tiene que subir mucho con los dos.
widget(app.slider, "Cuotas impagadas hoy").set_value(1).run()
revisar(app, "una cuota")
assert porcentaje(app.metric[0].value) > 5 * xgboost and porcentaje(app.metric[1].value) > 5 * logit
uno, con_uno = [m.value for m in app.metric], [porcentaje(m.value) for m in app.metric]
widget(app.slider, "Cuotas impagadas hoy").set_value(2).run()  # y con dos, más todavía
assert not app.exception and all(porcentaje(m.value) > p for m, p in zip(app.metric, con_uno))
dos = [m.value for m in app.metric]
widget(app.slider, "Cuotas impagadas hoy").set_value(0).run()  # A vuelve a estar al corriente
print(f"préstamo A: {cero[0]} y {cero[1]}; con una cuota impagada: {uno[0]} y {uno[1]}; con dos: {dos[0]} y {dos[1]}")
# Cifras de la memoria con los modelos con nivel de 2025 y el nivel por defecto.
assert (xgboost, logit) == (0.1, 0.2) and uno == ["5,6 %", "3,5 %"] and dos == ["59,7 %", "40,9 %"], (uno, dos)
# Qué sube y qué baja el riesgo: multiplicadores en escala logarítmica, con la referencia en 1, y un pie llano.
especificacion = graficos(app)
assert "Multiplica el riesgo por" in especificacion and '"type": "log"' in especificacion
assert especificacion.count('"datum": 1') == 2, especificacion
assert any("2 lo duplica; 0,5 lo reduce a la mitad" in c.value for c in app.caption)
# Un FICO de 500 queda fuera del rango habitual y salta el aviso de dato poco habitual, en dos frases.
widget(app.number_input, "Puntuación FICO").set_value(500).run()
aviso = next(i.value for i in app.info if i.value.startswith("Dato poco habitual"))
assert "Puntuación FICO (500; lo habitual va de 528 a 823)" in aviso and frases(aviso) == 2, aviso
widget(app.number_input, "Puntuación FICO").set_value(760).run()  # A vuelve a su FICO
print("aviso de dato poco habitual:", aviso)

# Préstamos de ejemplo: cada uno se carga en el formulario y se puntúa sin error.
elegir, cifras = widget(app.selectbox, "Préstamo"), []
grupo = DISCREPANCIAS[DISCREPANCIAS["grupo"] == "xgboost da más riesgo"]
pasado = f"(en el pasado impagó el {grupo['impagos'].sum() / grupo['prestamos'].sum():.0%} de los préstamos así)"
for ejemplo in elegir.options:
    widget(app.selectbox, "Préstamo").set_value(ejemplo).run()
    revisar(app, ejemplo)
    cifras.append(f"{ejemplo[0]} {app.metric[0].value} y {app.metric[1].value}")
    # Solo en G, buen FICO y nueve meses con cuotas impagadas, el modelo principal ve mucho más riesgo y salta
    # el aviso de discrepancia: qué pasa, una sola cifra del pasado y qué hacer.
    assert any(COMPROBAR in w.value for w in app.warning) == ejemplo.startswith("G:"), ejemplo
    if ejemplo.startswith("G:"):
        aviso = next(w.value for w in app.warning if COMPROBAR in w.value)
        assert pasado.replace("%", " %") in aviso and len(re.findall(r"\d+ %", aviso)) == 1, aviso
        assert frases(aviso) == 2, aviso
        print("aviso de G:", aviso)
    if ejemplo.startswith("B:"):  # una cuota impagada: mucho más riesgo que A
        assert widget(app.number_input, "Puntuación FICO").value == 640
        assert porcentaje(app.metric[0].value) > 10 * xgboost and porcentaje(app.metric[1].value) > 10 * logit
print(f"ejemplos (modelo principal y de contraste): {' | '.join(cifras)}")
print(f"ejemplos: {len(elegir.options)} préstamos inventados se cargan y se puntúan; solo en G salta el aviso de "
      "discrepancia")

with tempfile.TemporaryDirectory() as carpeta:
    copia = Path(carpeta) / "v_nivel"
    shutil.copytree(AQUI / "resultados", copia / "resultados", ignore=shutil.ignore_patterns("*.png"))
    shutil.copytree(AQUI / "modelos", copia / "modelos")
    (copia / "descriptivo").mkdir()
    for nombre in ["5_app.py", "ejemplo_cartera.csv"]:
        shutil.copy(AQUI / nombre, copia / nombre)
    # Año anómalo: con la tabla de tasas cortada en 2020, avisa de que no conviene usar la tasa de 2020, y el
    # desplegable de la tasa se abre para que el aviso se vea.
    TASAS[TASAS["anio"] <= 2020].to_csv(copia / "descriptivo" / "por_anio.csv", index=False)
    prueba = abrir("Evaluación de cartera", str(copia / "5_app.py"))
    desplegable = widget(prueba.expander, TASA)
    aviso = next(w.value for w in desplegable.warning if "anómalo" in w.value)
    assert "2020" in aviso and "3,65 %" in aviso and "0,47 %" in aviso and desplegable.proto.expanded, aviso
    assert widget(prueba.selectbox, TASA).value == "Base: la del último año cerrado (3,65 %, 2020)"
    print("año anómalo, con el desplegable de la tasa abierto:", aviso)
    # Sin modelos: la misma copia, con la tabla de tasas entera y sin la carpeta modelos/.
    st.cache_resource.clear()  # los modelos y las tablas ya leídas quedan en la caché del proceso
    st.cache_data.clear()
    shutil.rmtree(copia / "modelos")
    shutil.copy(AQUI / "descriptivo" / "por_anio.csv", copia / "descriptivo" / "por_anio.csv")
    for pantalla in PANTALLAS:
        prueba = abrir(pantalla, str(copia / "5_app.py"))
        assert any("necesita los modelos" in w.value for w in prueba.warning) and not prueba.main.subheader, pantalla
        assert not prueba.expander, pantalla
print("sin modelos: abren las dos pantallas y avisan de que faltan los modelos")
if pendiente:
    print("Pendiente de confirmar con los 18 años de 6_nivel.py:\n  - " + "\n  - ".join(pendiente))
print("Todo correcto.")
