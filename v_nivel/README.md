# ¿Qué hipotecas impagarán el año que viene?

Trabajo Fin de Máster del Máster en Big Data, Data Science e Inteligencia Artificial de la Universidad Complutense de Madrid. Autor: Lucía Sánchez Souto. Repositorio: https://github.com/luciasanchezsouto/tfm-impago-hipotecario.

El trabajo compara un baseline (modelo de referencia que ordena los préstamos por sus meses de retraso), una regresión logística y XGBoost para predecir cada 1 de enero qué hipotecas impagarán durante el año, con los datos públicos de Freddie Mac: 46 millones de hipotecas de Estados Unidos y 219 millones de observaciones anuales. Cada modelo se entrena con los años anteriores y se prueba en el siguiente, de 2008 a 2025. De todos los impagos de cada año, el baseline había situado de media el 43 % entre el 10 % de préstamos con más riesgo; la regresión logística, el 63 %, y XGBoost, el 65 % (recall al 10 %). XGBoost gana a la regresión logística en 14 de los 18 años, y un Random Forest de comprobación lo iguala. Pero XGBoost pierde esa ventaja después de 2020, probablemente porque aprende de ese año, en el que impagaron muchos más clientes, relaciones que no se repiten. La memoria propone separar el riesgo de cada préstamo en dos partes, una idea tomada del análisis de series temporales: el nivel de impago del año, común a toda la cartera, y el perfil del préstamo. El modelo recibe el nivel aparte y aprende sobre todo el perfil. Con el nivel aparte, XGBoost ordena mejor que la regresión logística en 16 de los 18 años, también en los cinco de 2021 a 2025, y es el que mejor acierta cuántos impagos habrá. XGBoost con nivel es el modelo final del trabajo y el que usa la aplicación.

La memoria está en `memoria_impago_hipotecario.pdf`: ocho capítulos (resumen, introducción, datos, metodología, resultados, del modelo a la decisión, la aplicación y conclusiones), la bibliografía y nueve anexos, entre ellos la descripción del código, la reproducción de los resultados, el detalle técnico de la aplicación y los resultados año a año del modelo con nivel y de la comprobación sin 2020. El código completo son los scripts de esta carpeta.

## Proceso

```
full_standard_dataset.zip (Freddie Mac, 44 GB)
   -> 1_datos.py        tabla anual: un préstamo vivo cada 1 de enero y si impaga ese año
   -> 2_descriptivo.py  tablas y figuras del análisis exploratorio, calidad del dato, correlaciones y VIF
                                                         -> descriptivo/
   -> 3_modelos.py      modelos sin nivel: baseline, regresión logística, XGBoost y XGBoost ajustado,
                        entrenados y probados año a año              -> resultados/*.csv
   -> 6_nivel.py        modelos con nivel: el nivel de impago del año va aparte (35 min)
                        -> resultados/nivel.csv y cuatro tablas *_nivel.csv; los modelos de 2025 -> modelos/
   -> 9_seleccion_variables.py ocho años con las 3, 5 y 10 variables de más peso (30 min)
   -> 4_figuras.py      figuras de resultados, recalibración e intervalos -> resultados/
   -> 7_sensibilidad.py fig9_sensibilidad.png: un préstamo inventado al mover una variable; coeficientes
   -> 8_otros_modelos.py Random Forest de comprobación y tamaños de lista (unos 40 min)
   -> 10_discrepancias.py compara los dos modelos con nivel donde discrepan, año a año (10 min)
                                                         -> resultados/discrepancias.csv y rangos.csv
   -> 11_sin_2020.py    comprobación: los modelos sin nivel de 2021 a 2025, sin 2020 al entrenar (10 min)
                                                         -> resultados/sin_2020.csv y sin_2020_comparacion.csv
   -> 5_app.py          aplicación web de dos pantallas: evaluación de cartera y
                        evaluación de préstamo
```

## Estructura

```
v_nivel/
├── 1_datos.py ... 11_sin_2020.py  los once scripts (2.284 líneas); el orden de ejecución, más abajo
├── probar_app.py         pruebas automáticas de la aplicación
├── ejecutar.py           ejecuta 1_datos.py y 3_modelos.py y guarda su salida en ejecucion.log
├── ejecucion.log         salida de 1_datos.py y 3_modelos.py en las ejecuciones con que se obtuvieron los resultados
├── requirements.txt      versiones exactas de las librerías
├── ejemplo_cartera.csv   cinco préstamos inventados: ejemplos de la aplicación y préstamo A de 7_sensibilidad.py
├── modelos/              los dos modelos con nivel de 2025, con que puntúa la aplicación
├── descriptivo/          tablas y figuras del análisis exploratorio
├── resultados/           métricas, calibración, importancia, modelos con nivel, discrepancias y figuras de los modelos
├── figuras/              capturas de la aplicación que usa la memoria
├── memoria_impago_hipotecario.pdf   la memoria
└── DERECHOS_DE_USO.md    condiciones de los datos y qué se publica
```

Los datos no están en el repositorio: los scripts los leen y escriben fuera de esta carpeta, en `data/` en la raíz del repositorio. Sí están los dos modelos con nivel de 2025 con que puntúa la aplicación, en `modelos/`, de modo que la aplicación funciona sin descargar ni entrenar nada.

## Requisitos

- Python 3.14 y las librerías de `requirements.txt`:

  ```
  pip install -r v_nivel/requirements.txt
  ```

- Tarjeta gráfica para entrenar XGBoost y el Random Forest. `3_modelos.py`, `6_nivel.py`, `9_seleccion_variables.py` y `11_sin_2020.py` entrenan con `device="cuda"` en su función `xgboost()`, y `8_otros_modelos.py`, en su función `bosque()`. Sin tarjeta hay que cambiarlo por `device="cpu"`: funciona igual, pero tarda unas diez veces más. Los demás scripts, la aplicación y las pruebas no la necesitan.
- Espacio en disco: 44 GB para el zip de Freddie Mac, 3,4 GB para la tabla anual (0,8 GB la piloto) y 0,2 GB para los modelos de todos los años, más el espacio temporal donde `1_datos.py` descomprime cada trimestre.
- Los tiempos de abajo son de un ordenador con Windows 11, 24 núcleos y 135 GB de memoria. No se ha probado con menos memoria.
- Los datos: el *Single-Family Loan-Level Dataset* de Freddie Mac, versión 48. Hay que registrarse en el portal de descarga de Freddie Mac (Clarity), aceptar sus condiciones de uso y descargar el *Standard Dataset* completo, `full_standard_dataset.zip`. No se redistribuye aquí; véase `DERECHOS_DE_USO.md`.

## Reproducir los resultados

Todas las órdenes se lanzan desde la raíz del repositorio. Antes, el zip tiene que estar en:

```
data/raw/historical/full_standard_dataset.zip
```

Los scripts se lanzan en este orden: 1, 2, 3, 6, 9, 4, 7, 8, 10 y 11. El 5 es la aplicación.

1. Tabla anual, en `data/sencillo/panel/`, un fichero por trimestre. Unos 15 minutos. Si se interrumpe, al relanzarlo salta los trimestres ya escritos.

   ```
   python v_nivel/1_datos.py
   ```

2. Análisis exploratorio, en `v_nivel/descriptivo/`: nueve tablas (`por_anio.csv`, `por_tramo.csv`, `variables.csv`, `categorias.csv`, `calidad.csv`, `correlacion_impago.csv`, `correlacion_variables.csv`, `vif.csv` y `saltos_nivel.csv`) y cinco figuras (`fig2_cartera.png`, `fig3_quien_impaga.png`, `fig4_prestamos_nuevos.png`, `fig_saltos.png` y `fig_correlaciones.png`). Alrededor de un minuto. `variables.csv` resume cada variable numérica (media, percentiles 1, 25, 50, 75 y 99 y valores que faltan) y `categorias.csv` da el porcentaje de filas de cada categoría de las variables de texto y de los meses de retraso; de las dos salen el diccionario de variables de la memoria y su tabla de las variables numéricas principales. `por_anio.csv` da la tasa de impago de cada año, que usan como nivel los pasos 4, 5, 7 y 9 y la aplicación. `saltos_nivel.csv` da, para cada año, cuánto cambia esa tasa respecto al anterior, en toda la cartera y por separado en los préstamos al corriente y con retraso el 1 de enero; `fig_saltos.png` lo dibuja: el salto de cada año y, en 2008, 2009 y 2020, cuánto sube el riesgo en cada grupo.

   ```
   python v_nivel/2_descriptivo.py
   ```

3. Modelos sin nivel: baseline, regresión logística, XGBoost con parámetros fijos y XGBoost ajustado, con los parámetros elegidos cada año por validación. Escribe en `v_nivel/resultados/` `metricas.csv`, `segmentos.csv`, `calibracion.csv`, `importancia.csv` y `parametros.csv`, y en `data/sencillo/modelos/` los modelos de cada año y `categorias.joblib`. 102 minutos, de ellos unos 70 de la búsqueda de parámetros.

   ```
   python v_nivel/3_modelos.py
   ```

4. Modelos con nivel: la regresión logística y XGBoost con la misma tabla, la misma muestra, los mismos pesos y los mismos parámetros que en el paso 3, pero con el nivel de impago de cada año aparte. XGBoost lo recibe como punto de partida de cada préstamo; la regresión logística, como una variable más. Así el modelo solo aprende el perfil del préstamo. Para predecir un año hace falta su nivel, que el 1 de enero no se conoce, y se prueban tres previsiones: la tasa del último año, un suavizado exponencial simple y la mediana de los cinco últimos años. Escribe `nivel.csv`, `umbrales_nivel.csv` (qué parte de los impagos se encuentra revisando el 1, 5, 10 o 20 % de la cartera), `calibracion_nivel.csv` (impago previsto y real en diez grupos de riesgo), `segmentos_nivel.csv` (AUC en los préstamos al corriente y con retraso) e `importancia_nivel.csv` (cuánto pesa cada variable en XGBoost con nivel, con valores SHAP). Guarda los modelos de cada año en `data/sencillo/modelos_nivel/` y copia los de 2025, con `categorias.joblib`, a `v_nivel/modelos/`, que son los que usan la aplicación y el paso 7. Unos 35 minutos con tarjeta gráfica. Necesita la tabla anual y `por_anio.csv` del paso 2.

   ```
   python v_nivel/6_nivel.py
   ```

   Con uno o varios años, por ejemplo `python v_nivel/6_nivel.py 2025`, entrena solo esos años (de 2 a 4 minutos cada uno) y escribe sus tablas con el sufijo `_parcial` (por ejemplo, `nivel_parcial.csv`), sin tocar los resultados de los 18 años. Si entre ellos está 2025, también renueva `v_nivel/modelos/`.

5. Selección de variables: repite ocho años de prueba con menos variables, con los modelos con nivel, y escribe `seleccion.csv`. Unos 30 minutos con tarjeta gráfica. Necesita la tabla anual y `por_anio.csv` del paso 2. Con años sueltos, por ejemplo `python v_nivel/9_seleccion_variables.py 2025`, reescribe `seleccion.csv` solo con esos años y pisa la tabla completa, así que para probar conviene el modo piloto.

   ```
   python v_nivel/9_seleccion_variables.py
   ```

6. Figuras de resultados: `fig5_acierto.png` y `fig6_segmentos.png`, con las tablas del paso 3; `fig_nivel.png`, que compara la ventaja de XGBoost con y sin nivel; `fig_calibracion_nivel.png` y `fig_segmentos_nivel.png`, con la calibración y el AUC por situación de pago de los modelos con nivel, y `fig_importancia_nivel.png`, con las variables que más pesan en el modelo con nivel; más `recalibracion.csv` e `intervalos.csv`. Escribe además cuatro tablas de comprobación: `intervalos_bloques.csv`, los intervalos de `intervalos.csv` remuestreando bloques de 1, 2 y 3 años consecutivos (con bloques de un año coincide con `intervalos.csv`); `calibracion_pendiente.csv`, por año de prueba, la pendiente y el intercepto de la recta de log(odds real) sobre log(odds prevista) en los diez grupos de riesgo de los dos modelos con nivel (pendiente 1 e intercepto 0 es calibración perfecta); `frecuencia_variables.csv`, en cuántos años está cada variable entre las diez y entre las cinco de más peso del XGBoost con nivel, con el orden de `importancia_nivel.csv` (medido en el año de prueba) y, si ya existe `seleccion.csv` (paso 5), con el de sus ocho años, y `aviso_anomalo.csv`, que aplica a cada año de prueba el aviso de año anómalo de la aplicación como si fuera una regla automática: si la tasa del último año cerrado pasa del doble de la mediana de los cinco anteriores, el nivel previsto pasa a ser la mediana de los cinco últimos años. Da, para los dos modelos con nivel, si salta el aviso y la calibración con y sin la regla, y el script imprime el desvío mediano de 2009 a 2025 en los dos casos. Unos 5 segundos. Va después de `6_nivel.py` y de `9_seleccion_variables.py` porque lee sus tablas (`nivel.csv`, `calibracion_nivel.csv`, `segmentos_nivel.csv`, `importancia_nivel.csv` y `seleccion.csv`), y lee también `por_anio.csv` del paso 2; si a `nivel.csv` le falta alguno de los 18 años, se detiene con un aviso. Funciona sin los datos, con las tablas que ya están en `resultados/` y `descriptivo/`; si se repite el paso 4 o el 5, hay que volver a lanzarlo.

   ```
   python v_nivel/4_figuras.py
   ```

7. Sensibilidad de los modelos con nivel de 2025 en el primer préstamo de `ejemplo_cartera.csv`, con la tasa de impago del último año cerrado como nivel: `sensibilidad.csv`, `coeficientes.csv` y `fig9_sensibilidad.png`. Unos 3 segundos. Necesita los modelos de `v_nivel/modelos/` (paso 4, o los que ya están en el repositorio) y `por_anio.csv` del paso 2.

   ```
   python v_nivel/7_sensibilidad.py
   ```

8. Random Forest de comprobación y tamaños de lista de los modelos sin nivel: `otros_modelos.csv` y `umbrales.csv`. Unos 40 minutos con tarjeta gráfica. Necesita la tabla anual y los modelos del paso 3.

   ```
   python v_nivel/8_otros_modelos.py
   ```

9. Discrepancias: vuelve a puntuar cada año de prueba con los dos modelos con nivel entrenados antes de ese año, con la tasa de impago del año anterior como nivel. Separa los préstamos en que una probabilidad es al menos el triple de la otra y alguna llega al 5 %, y los que tienen algún valor muy poco habitual, y compara en cada grupo la tasa de impago real con la prevista. Escribe `discrepancias.csv` y `rangos.csv`, que usa la aplicación para sus avisos. No entrena nada ni usa la tarjeta gráfica; unos 10 minutos. Necesita la tabla anual, `por_anio.csv` del paso 2 y los modelos del paso 4.

   ```
   python v_nivel/10_discrepancias.py
   ```

10. Comprobación sin 2020: repite los años de prueba 2021 a 2025 con la regresión logística y XGBoost del paso 3, pero sin 2020 en el entrenamiento, para ver si XGBoost pierde su ventaja después de 2020 porque aprende de ese año. No es un modelo más: el modelo final sigue siendo XGBoost con nivel, del paso 4. Escribe `sin_2020.csv` (AUC, captura y calibración de cada año y modelo) y `sin_2020_comparacion.csv`, que resume para los dos modelos tres formas de entrenar: con todos los años, sin 2020 y con nivel. Guarda sus modelos en `data/sencillo/modelos_sin_2020/`, no en `v_nivel/modelos/`. Unos 10 minutos con tarjeta gráfica. Necesita la tabla anual, `metricas.csv` del paso 3 y `nivel.csv` del paso 4.

    ```
    python v_nivel/11_sin_2020.py
    ```

    Con `resumen`, `python v_nivel/11_sin_2020.py resumen`, no entrena: rehace `sin_2020_comparacion.csv` con las tres tablas de `resultados/`, en un segundo y sin los datos.

`python v_nivel/ejecutar.py` lanza seguidos los pasos 1 y 3 y añade su salida a `ejecucion.log`.

## Prueba rápida: modo piloto

Con la palabra `piloto`, `1_datos.py` usa solo el primer trimestre de cada año y `3_modelos.py` prueba solo 2010, 2016 y 2025. Escriben en carpetas aparte, que crean al ejecutarse y no vienen en el repositorio (`data/sencillo/panel_piloto/`, `data/sencillo/modelos_piloto/`, `v_nivel/resultados_piloto/`), así que no tocan los resultados completos; `2_descriptivo.py piloto` escribe en `v_nivel/descriptivo_piloto/`. En el ordenador descrito tardaron unos 3 y unos 5 minutos. Hace falta igualmente el zip completo de Freddie Mac.

```
python v_nivel/1_datos.py piloto
```

```
python v_nivel/2_descriptivo.py piloto
```

```
python v_nivel/3_modelos.py piloto
```

`6_nivel.py`, `8_otros_modelos.py` y `9_seleccion_variables.py` también admiten `piloto` y escriben en `v_nivel/resultados_piloto/`; `6_nivel.py` guarda sus modelos en `data/sencillo/modelos_nivel_piloto/`. Los dos últimos tardan unos 3 minutos cada uno. `4_figuras.py`, `7_sensibilidad.py`, `10_discrepancias.py` y `11_sin_2020.py` no tienen modo piloto: leen los resultados, la tabla o los modelos completos.

## La aplicación

```
streamlit run v_nivel/5_app.py
```

Está pensada para el analista de seguimiento del riesgo hipotecario que cada enero decide qué préstamos revisar con la capacidad que tiene su equipo. Tiene dos pantallas en un menú lateral:

- «Evaluación de cartera» puntúa una cartera en CSV y dice qué préstamos revisar primero. Arriba, dos desplegables cerrados guardan lo que casi nunca se toca: la cartera (sin fichero, una de ejemplo de 500 préstamos inventados, que también se puede descargar) y la tasa de impago prevista para el año, elegida entre tres escenarios calculados con `descriptivo/por_anio.csv` (base: la del último año cerrado; prudente: la mayor de los cinco últimos años; severa: la de 2009) o escrita a mano. Si la tasa del último año cerrado pasa del doble de la mediana de los cinco anteriores, el desplegable de la tasa se abre y avisa de que ese año fue anómalo. Debajo se indica cuántos préstamos puede revisar el equipo (por defecto, el 10 % de la cartera) y salen dos cifras: los impagos previstos en el año, con su horquilla (se explica más abajo), y cuántos de ellos están en la lista, con su parte del total. La lista de revisión va en dos pestañas (con cuotas impagadas y al corriente con alerta), siempre de más a menos probabilidad de impago, con cinco columnas: el puesto, el préstamo, la probabilidad de impago, el motivo principal en lenguaje llano y si conviene verificar sus datos. Al marcar la casilla de una fila, ese préstamo se abre en «Evaluación de préstamo». La lista se descarga con todas las columnas, también las que no se ven en pantalla: el tramo de riesgo frente a la tasa prevista, las cuotas impagadas, el saldo, el saldo ponderado por la probabilidad, la probabilidad del modelo de contraste y el resto de datos de cada préstamo. El coste y el ahorro de revisar van aparte, en una simulación opcional con los supuestos de quien la usa.
- «Evaluación de préstamo» explica la probabilidad de un préstamo y permite cambiar sus datos. Su «¿Y si…?» mueve las cuotas impagadas y, en un desplegable, el FICO, el LTV y el tipo de interés. Dice qué hacer si los dos modelos discrepan, avisa si algún dato es poco habitual y muestra por cuánto multiplica cada dato el riesgo.

La horquilla de los impagos previstos junta las dos razones por las que los impagos reales no son los previstos: lo que se desvió el modelo principal en los años de prueba y el azar de una cartera de ese tamaño, que en una pequeña es grande. Para cada uno de los 17 cocientes entre impagos previstos y reales de los años de prueba desde 2009 (`resultados/nivel.csv`, con la tasa del último año cerrado), la odds de cada préstamo se divide por el cociente, p' = p / (p + c (1 − p)), que es mover el nivel del modelo lo que se desvió ese año, y se sortea 2.000 veces qué préstamos impagan, cada uno con su probabilidad corregida. La horquilla va de los percentiles 10 a 90 de los 34.000 sorteos, redondeados: en la cartera de ejemplo, entre 3 y 13 impagos para unos 7 previstos (solo con lo que se desvió el modelo, salía entre 6 y 12). La semilla es fija, así que sale siempre igual, y se calcula en unas décimas de segundo con cualquier tamaño de cartera: con más de 2.000 préstamos, los de odds casi igual (a menos de un 5 %) se sortean juntos con una binomial de su probabilidad media, lo que apenas cambia el resultado. Solo se muestra con el escenario base, porque lo que se desvió el modelo se midió con esa tasa; con los demás, la ayuda de la cifra dice que no hay horquilla y por qué. No es una garantía: supone que el año se parecerá a alguno de los pasados.

En la barra lateral, una ficha breve dice qué predice la herramienta, con la definición de impago, y que usa dos modelos entrenados con las hipotecas de Freddie Mac de 2000 a 2024.

La aplicación está pensada para el uso diario en un banco, así que sus textos son cortos y no tienen términos técnicos: cada pantalla se abre con una frase, ningún pie de texto pasa de una frase, y lo que explica cada cifra, campo o columna (cómo se calcula la previsión, qué es la horquilla, qué parte de los impagos se encontró en el pasado con una lista de ese tamaño, por qué conviene verificar los datos de un préstamo, que la simulación usa los supuestos de quien la rellena) está en su ayuda, que se ve al pasar el cursor. Cómo se validó el modelo, y sus límites, está en la memoria.

Las dos pantallas puntúan con los modelos con nivel de 2025 de `modelos/`. En pantalla se llaman «modelo principal» (XGBoost con nivel), que da la cifra de referencia y ordena la lista, y «modelo de contraste» (regresión logística con nivel), una segunda opinión; los nombres técnicos no aparecen en la aplicación. Los dos parten de la tasa de impago prevista para el año, que se elige en «Evaluación de cartera», vale también para «Evaluación de préstamo» y se conserva al cambiar de pantalla. Cambiarla no cambia el orden de la lista, pero sí cuántos impagos se prevén. Además de `descriptivo/por_anio.csv`, la aplicación lee cuatro tablas de `resultados/`: `nivel.csv` (lo que se desvió el modelo en los años de prueba, para la horquilla), `umbrales_nivel.csv` (qué parte de los impagos se encontró en los años de prueba con una lista de ese tamaño), y `discrepancias.csv` y `rangos.csv` (los avisos de discrepancia entre los modelos y de dato poco habitual). Todo ello está en el repositorio, así que la aplicación funciona sin descargar los datos ni entrenar nada.

Si se borra la carpeta `modelos/`, la aplicación usa los modelos de `data/sencillo/modelos_nivel/` o los de la ejecución piloto; sin ninguna de esas carpetas, las dos pantallas avisan de que faltan los modelos. Los modelos sin nivel de `3_modelos.py` no sirven para la aplicación, porque no admiten el nivel del año. Para rehacer solo los modelos de la aplicación basta `python v_nivel/6_nivel.py 2025`.

Pruebas automáticas, sin navegador, en unos 10 segundos; necesitan los modelos de `modelos/`:

```
python -X utf8 v_nivel/probar_app.py
```

Abren las dos pantallas y comprueban que en ningún sitio, tampoco en la barra lateral, en las ayudas, en los desplegables cerrados, en las columnas ni en los gráficos, aparecen términos técnicos (los nombres de los modelos y de los métodos, AUC, percentiles, binomiales, desviaciones típicas o explicaciones de la validación); que la introducción de cada pantalla es una frase y cada pie, como mucho una, y que la ficha lateral es breve. Prueban aparte los escenarios de tasa y el aviso de año anómalo, con la serie de tasas cortada en 2020. Después comprueban «Evaluación de cartera» con la cartera de ejemplo: cómo está hecha la pantalla (los dos desplegables cerrados de arriba, el campo de la capacidad, las dos cifras, la lista y la simulación, sin bloques numerados ni botones) y el ejemplo que se descarga; el orden de la lista, sus dos pestañas y las cinco columnas visibles con sus ayudas; la lista descargable, con todas sus columnas en su orden, el tramo, el saldo ponderado y el aviso de verificar datos; las dos cifras, con lo medido y la horquilla, que se recalcula aparte sin sorteos (con la distribución exacta del número de impagos) y debe contener la cifra central, ser más ancha que la que solo contaba lo que se desvió el modelo y salir siempre igual, y que con 50.000 préstamos sale también como la exacta, en poco tiempo y más estrecha en proporción; que cambiar de escenario no cambia el orden pero sí los impagos previstos, y que la horquilla solo sale con el escenario base; que una tasa escrita a mano admite dos cambios seguidos, pasa a «Evaluación de préstamo» y se conserva al volver; la capacidad del equipo; la simulación de coste y ahorro; que marcar una fila abre ese préstamo en «Evaluación de préstamo» con sus datos y la misma probabilidad, y que al volver la lista no tiene ninguna fila marcada, y los ficheros subidos: el ejemplo descargado, que da la misma lista; uno al que le faltan columnas; los cinco préstamos de `ejemplo_cartera.csv`, también con un identificador y una columna propios, y los casos límite (un solo préstamo, uno ya en impago, datos ausentes e importes de cero o enormes). Siguen el préstamo A, con las cifras de la memoria; los mandos de «¿Y si…?», el gráfico de multiplicadores y el aviso de dato poco habitual, y los siete préstamos de ejemplo, con el aviso de discrepancia, corto y con una sola cifra del pasado, solo en el préstamo G. Por último abren la lista con una tabla de tasas que acaba en 2020, que debe abrir el desplegable de la tasa y avisar del año anómalo, y las dos pantallas sin modelos, y terminan con «Todo correcto.». Si `umbrales_nivel.csv` no tiene aún los 18 años, dicen qué comprobación queda pendiente.

AppTest, la herramienta de Streamlit con que se prueba sin navegador, no sabe marcar la casilla de una fila. Para abrir un préstamo de la lista, la prueba le envía lo mismo que mandaría el navegador al marcarla, así que corre el mismo código que en el uso real; que la casilla se vea y responda al pulsarla solo se puede comprobar en un navegador.

## Qué hay en el repositorio y qué no

Están las tablas agregadas y las figuras de `descriptivo/` y `resultados/`: cifras por año, por modelo, por tramo, por variable, por segmento o por grupo de riesgo, nunca por préstamo. De ellas salen todas las cifras de la memoria.

Están también los dos modelos con nivel de 2025, entrenados con los años 2000 a 2024 (`modelos/logit_2025.joblib` y `modelos/xgboost_2025.joblib`), y los niveles de sus variables categóricas (`modelos/categorias.joblib`), que son los que usa la aplicación. No guardan datos de ningún préstamo.

No están, porque las condiciones de Freddie Mac prohíben redistribuir los datos:

- los datos descargados y la tabla anual (`data/raw/`, `data/sencillo/panel*/`);
- las predicciones por préstamo;
- los demás modelos, de otros años o sin nivel (`data/sencillo/modelos*/`), que solo hacen falta para repetir los resultados.

Los préstamos de `ejemplo_cartera.csv`, de la cartera de ejemplo de 500 préstamos que genera la aplicación y de las capturas de la aplicación son inventados.
