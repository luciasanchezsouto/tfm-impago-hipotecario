# ¿Qué hipotecas impagarán el año que viene?

Trabajo Fin de Máster del Máster en Big Data, Data Science e Inteligencia Artificial de la Universidad Complutense de Madrid. Autor: Lucía Sánchez Souto.

Todo el trabajo está en la carpeta [`v_nivel/`](/v_nivel):

- [`memoria_impago_hipotecario.pdf`](/v_nivel/memoria_impago_hipotecario.pdf): la memoria y sus anexos.
- [`README.md`](/v_nivel/README.md): qué hace cada script, cómo reproducir los resultados y cómo abrir la aplicación.
- [`DERECHOS_DE_USO.md`](/v_nivel/DERECHOS_DE_USO.md): las condiciones de los datos de Freddie Mac, qué se publica aquí y con qué derechos.
- [`LICENSE`](/LICENSE): la licencia MIT del código. No cubre la memoria ni los modelos y tablas que salen de los datos de Freddie Mac.

Para abrir la aplicación, desde la raíz del repositorio:

```
pip install -r v_nivel/requirements.txt
streamlit run v_nivel/5_app.py
```
La aplicación también dispone de link: https://tfm-impago-hipotecario-gjzz8eulsxlxkimrbn7pue.streamlit.app/

La aplicación funciona sin descargar ni entrenar nada: los dos modelos con que puntúa están en `v_nivel/modelos/` y los préstamos de ejemplo son inventados. Los datos de Freddie Mac no están en el repositorio; para repetir los resultados hay que descargarlos por cuenta propia, como explica el `README.md` de la carpeta.
