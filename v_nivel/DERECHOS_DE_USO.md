# Derechos de uso

## Los datos

Este trabajo usa el *Single-Family Loan-Level Dataset* de Freddie Mac, versión 48. Freddie Mac lo ofrece gratis a quien se registra en su portal de descarga y acepta sus condiciones de uso. Esas condiciones prohíben licenciar, distribuir, publicar o poner a disposición de terceros el conjunto de datos, entero o en parte, y los productos derivados de él.

Por eso este repositorio contiene:

- el código;
- tablas agregadas (por año, modelo, tramo, variable, segmento o grupo de riesgo) y las figuras hechas con ellas;
- los dos modelos con nivel de 2025 con que puntúa la aplicación, XGBoost y la regresión logística, y los niveles de sus variables categóricas (`v_nivel/modelos/`). Son parámetros ajustados y no guardan datos de ningún préstamo. Se incluyen solo para que los tutores y el tribunal puedan ejecutar la aplicación de este trabajo académico;
- la memoria.

Y no contiene:

- los datos de Freddie Mac, ni la tabla anual construida con ellos;
- predicciones por préstamo;
- los demás modelos, de otros años o sin nivel.

Los préstamos de `ejemplo_cartera.csv`, de la cartera de ejemplo de 500 préstamos que genera la aplicación y de las capturas de la aplicación son inventados. No proceden de los datos de Freddie Mac.

Quien quiera ejecutar el código tiene que descargar los datos por su cuenta, registrándose y aceptando él mismo las condiciones de Freddie Mac, y es responsable de cumplirlas. Esta nota resume cómo se ha aplicado el trabajo a sí mismo esas condiciones; no es asesoramiento legal ni sustituye su texto, que es el que vale.

## Cita de la fuente

Como en la bibliografía de la memoria:

> Freddie Mac (2026). *Single-Family Loan-Level Dataset: general user guide and release notes*, release 48. freddiemac.com.

Si las condiciones de Freddie Mac piden una fórmula de atribución concreta, debe usarse esa.

## El código

© 2026 Lucía Sánchez Souto. Todos los derechos reservados. El código se facilita a los tutores y al tribunal para que puedan leerlo y reproducir los resultados de este trabajo académico. No se concede licencia para otros usos.
