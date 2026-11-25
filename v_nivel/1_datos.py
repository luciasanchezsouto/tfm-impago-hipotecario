"""Del zip de Freddie Mac a una tabla anual de préstamos.

Cada fila es un préstamo vivo el 1 de enero de un año, con lo que se sabía de él ese día
y con el resultado que se quiere predecir: si impagó durante ese año.

    python v_nivel/1_datos.py            todos los trimestres (unos 15 minutos)
    python v_nivel/1_datos.py piloto     solo el primer trimestre de cada año
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[1]
ZIP = RAIZ / "data" / "raw" / "historical" / "full_standard_dataset.zip"
PILOTO = "piloto" in sys.argv
SALIDA = RAIZ / "data" / "sencillo" / ("panel_piloto" if PILOTO else "panel")

# Posición de cada campo en los ficheros de texto, que no traen cabecera.
ORIGINACION = {
    "fico": 0, "primer_pago": 1, "primer_comprador": 2, "seguro": 5, "unidades": 6, "ocupacion": 7,
    "dti": 9, "importe": 10, "ltv": 11, "tipo_inicial": 12, "canal": 13, "estado": 16, "inmueble": 17,
    "id": 19, "finalidad": 20, "plazo": 21, "prestatarios": 22,
}
MENSUAL = {"id": 0, "mes": 1, "saldo": 2, "mora": 3, "modificacion": 7, "salida": 8}
NUMERICAS = ["fico", "primer_pago", "seguro", "unidades", "dti", "importe", "ltv", "tipo_inicial", "plazo",
             "prestatarios"]
# Freddie Mac marca el dato que falta con estos códigos; aquí pasan a vacío.
SIN_DATO = {"fico": 9999, "seguro": 999, "unidades": 99, "dti": 999, "ltv": 999, "prestatarios": 99,
            "primer_comprador": "9", "ocupacion": "9", "canal": "9", "finalidad": "9", "inmueble": "99"}
# Códigos de salida que son una pérdida para el banco: venta a un tercero, venta en corto y adjudicación.
PERDIDA = [2, 3, 9]
PREPAGO = 1
ULTIMO_ANIO = 2025  # 2026 todavía no ha terminado


def leer(zip_trimestre, nombre, campos, carpeta):
    """Lee del zip un fichero de texto y se queda con los campos pedidos, todos como texto."""
    fichero = zip_trimestre.extract(nombre, carpeta)
    tabla = pl.scan_csv(fichero, separator="|", has_header=False, infer_schema=False, quote_char=None)
    return tabla.select(pl.nth(posicion).alias(campo) for campo, posicion in campos.items()).collect()


def tabla_anual(originacion, mensual):
    """Cruza la ficha de cada préstamo con su historia mensual y deja una fila por préstamo y año."""
    originacion = originacion.with_columns(pl.col(NUMERICAS).cast(pl.Float64, strict=False))
    for campo, codigo in SIN_DATO.items():
        originacion = originacion.with_columns(pl.when(pl.col(campo) != codigo).then(pl.col(campo)).alias(campo))

    # La mora viene en meses de retraso (0, 1, 2...) o como "RA" si la vivienda ya está adjudicada.
    mensual = mensual.with_columns(
        pl.col("mes").cast(pl.Int32),
        pl.col("saldo").cast(pl.Float64),
        pl.col("salida").cast(pl.Int16),
        adjudicado=pl.col("mora") == "RA",
        mora=pl.col("mora").cast(pl.Int16, strict=False),
    ).sort("id", "mes")

    # Impago: tres meses de retraso o más, o una salida con pérdida. Se guarda el primer mes en que ocurre.
    impago = (pl.col("mora") >= 3) | pl.col("adjudicado") | pl.col("salida").is_in(PERDIDA)
    por_prestamo = mensual.group_by("id").agg(
        primer_impago=pl.col("mes").filter(impago).min(),
        ultimo_mes=pl.col("mes").last(),
        ultima_salida=pl.col("salida").last(),
    )

    # Resumen de cada año natural: cómo acaba diciembre y cuántos retrasos hubo en los doce meses.
    por_anio = mensual.group_by("id", anio_visto=pl.col("mes") // 100, maintain_order=True).agg(
        mes_foto=pl.col("mes").last(),
        salida_foto=pl.col("salida").last(),
        saldo=pl.col("saldo").last(),
        mora_actual=pl.col("mora").last(),
        mora_maxima_12m=pl.col("mora").max(),
        meses_en_mora_12m=(pl.col("mora") >= 1).sum(),
        modificado=pl.col("modificacion").last().is_not_null(),
    )

    tabla = por_anio.join(por_prestamo, on="id").join(originacion, on="id")
    anio = pl.col("anio_visto") + 1  # el año que se predice es el siguiente al que se ha visto
    diciembre = pl.col("anio_visto") * 100 + 12
    # Entra quien llega a diciembre vivo y sin haber impagado nunca.
    tabla = tabla.filter(
        pl.col("mes_foto") == diciembre,
        pl.col("salida_foto").is_null(),
        pl.col("primer_impago").is_null() | (pl.col("primer_impago") > diciembre),
    )
    impaga = (pl.col("primer_impago") <= anio * 100 + 12).fill_null(False)
    # El resultado se conoce si impaga, si se cancela por prepago o si sigue en los datos al acabar el año.
    conocido = impaga | (pl.col("ultima_salida") == PREPAGO) | (pl.col("ultimo_mes") >= anio * 100 + 12)
    primer_pago = pl.col("primer_pago").cast(pl.Int32)
    tabla = tabla.filter(conocido.fill_null(False), anio <= ULTIMO_ANIO).with_columns(
        anio=anio,
        impago=impaga.cast(pl.Int8),
        antiguedad=anio * 12 + 1 - (primer_pago // 100 * 12 + primer_pago % 100),
        ltv_actual=pl.col("saldo") / pl.col("importe") * pl.col("ltv"),
        modificado=pl.col("modificado").cast(pl.Int8),
    )
    return tabla.drop("anio_visto", "mes_foto", "salida_foto", "primer_impago", "ultimo_mes", "ultima_salida",
                      "primer_pago")


SALIDA.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as zip_completo:
    for zip_del_anio in sorted(zip_completo.namelist()):
        with zipfile.ZipFile(zip_completo.open(zip_del_anio)) as zip_anio:
            for zip_del_trimestre in sorted(zip_anio.namelist()):
                trimestre = zip_del_trimestre[-10:-4]  # historical_data_2005Q1.zip -> 2005Q1
                destino = SALIDA / f"{trimestre}.parquet"
                if destino.exists() or (PILOTO and not trimestre.endswith("Q1")):
                    continue
                with zipfile.ZipFile(zip_anio.open(zip_del_trimestre)) as z, tempfile.TemporaryDirectory() as tmp:
                    originacion = leer(z, f"orig_{trimestre}.txt", ORIGINACION, tmp)
                    mensual = leer(z, f"perf_{trimestre}.txt", MENSUAL, tmp)
                tabla = tabla_anual(originacion, mensual)
                tabla.write_parquet(destino)
                print(f"{trimestre}: {originacion.height:,} préstamos, {tabla.height:,} filas", flush=True)
