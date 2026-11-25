"""Ejecuta 1_datos.py y 3_modelos.py, uno tras otro, y guarda lo que escriben en ejecucion.log.

    python v_nivel/ejecutar.py
"""

import subprocess
import sys
import time
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
with open(CARPETA / "ejecucion.log", "a", encoding="utf-8") as registro:
    for script in ["1_datos.py", "3_modelos.py"]:
        inicio = time.time()
        registro.write(f"--- {script}, {time.strftime('%d/%m %H:%M')}\n")
        registro.flush()
        subprocess.run([sys.executable, "-X", "utf8", CARPETA / script], stdout=registro, stderr=registro, check=True)
        registro.write(f"--- {script} terminado en {(time.time() - inicio) / 60:.0f} minutos\n")
        registro.flush()
