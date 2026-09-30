# Plan de reentrenamiento

Este plan describe cuándo y cómo se reentrenaría el modelo si recibiera datos nuevos. Hoy el proyecto es un simulador local con un dataset fijo, así que el plan no está automatizado. El monitoreo (`make monitor`) sí está implementado.

## Señales

| Señal | Cómo se mide | Umbral | Acción |
|---|---|---|---|
| Drift de datos en variables clave | PSI contra train en `lead_time_days`, `age`, `has_history` y la probabilidad predicha (`reports/drift_report.md`) | PSI > 0,25 | Evaluar reentrenamiento (veredicto «reentrenar») |
| Drift de datos en otras variables | PSI del resto de features | PSI entre 0,10 y 0,25, o > 0,25 fuera de las clave | Vigilar: repetir el reporte con el siguiente lote |
| Caída de desempeño | PR-AUC en la ventana más reciente con resultados conocidos (al menos 2.000 citas) | Más de 0,03 por debajo de la PR-AUC de test (0,3325), o por debajo de la regla de antelación | Reentrenar |
| Descalibración | Diferencia entre la probabilidad media y la tasa observada, y Brier en esa ventana | Más de 3 puntos porcentuales, o Brier más de 0,01 peor que en test (0,1878) | Recalibrar (Platt con la ventana reciente); reentrenar si persiste |
| Política de acciones | Costo medio por cita de la política frente a "no hacer nada", con los mismos supuestos | La política cuesta más que no hacer nada | Revisar umbrales y supuestos antes de tocar el modelo |

Notas:
- **Los resultados llegan con retraso.** El resultado de una cita se conoce recién después de su fecha, y la antelación puede llegar a 176 días. Por eso el desempeño solo se mide sobre citas ya ocurridas, y el drift de datos es la alarma temprana.
- **El veredicto usa el PSI, no los p-valores.** Con miles de filas, KS y chi² dan p < 0,001 ante diferencias irrelevantes.
- **Una muestra pequeña no da veredicto.** Con menos de 100 filas el reporte no evalúa. Es el caso actual de las simulaciones registradas (2 filas con probabilidad).

## Qué muestra el reporte actual

- **Control sin drift** (remuestreo de train): estable, con PSI ≤ 0,015 en todas las variables.
- **Drift inducido** (antelación triplicada, edad a la mitad, todos los pacientes nuevos): «reentrenar». Hay alerta en `lead_time_days`, `age`, `prev_appointments`, `has_history` y la probabilidad; las variables que no se tocaron siguen estables. El detector funciona.
- **Test real (junio 2016)**: «reentrenar», por `has_history` (PSI 0,44) y `prev_appointments` (0,46).
  - No es un cambio de comportamiento de los pacientes. Es el artefacto de la ventana recortada: el dataset empieza el 2016-04-29, así que las citas de junio tienen más historial disponible que las de mayo (41,5 % contra 13,0 %).
  - La probabilidad predicha casi no cambia (PSI 0,009).
  - `appointment_weekday` sale en «vigilar» (0,11) por la composición del calendario: 2 de las 6 fechas de test son miércoles.
  - La acción correcta sería reentrenar con una ventana más larga, para que las features de historial se aprendan con datos representativos. Ajustar umbrales no lo resuelve.

## Cómo se reentrenaría

1. **Datos.** Exportar las citas nuevas con el mismo esquema y correr `make data`. La validación de esquema rechaza el lote si no cumple las reglas.
2. **Features y split.** Correr `make features` con cortes cronológicos nuevos en `SplitConfig`: las últimas dos semanas para validación y la última semana para test. El historial se sigue calculando solo con citas anteriores a `as_of`.
3. **Entrenamiento.** Correr `make train` con el mismo protocolo:
   - Regresión logística en train.
   - Calibración Platt o isotónica elegida en validación.
   - Umbrales por costo en validación.
   - LightGBM como comparación.
4. **Versión.** El artefacto nuevo se llama `<fecha>_<hash de datos>.joblib`. El anterior se conserva.

## Validación antes de reemplazar (campeón contra retador)

El modelo nuevo (retador) reemplaza al actual (campeón) solo si cumple todo lo siguiente:

1. `make lint`, `make test` y `make smoke` pasan.
2. Se evalúa en la ventana más reciente con resultados conocidos, que **ninguno** de los dos modelos usó para entrenar ni calibrar:
   - PR-AUC del retador ≥ PR-AUC del campeón − 0,005.
   - Brier del retador ≤ Brier del campeón + 0,002.
3. En la tabla de segmentos (edad, beca, género, historial), el recall de ningún grupo cae más de 5 puntos frente al campeón. El sesgo de asignación por edad y beca no empeora.
4. En validación, la política de umbrales del retador no cuesta más que "no hacer nada".
5. Una persona revisa las decisiones y los resultados antes de hacer el cambio.

El reemplazo consiste en actualizar `models/metadata.json` para que apunte al archivo nuevo. El rollback es volver a la metadata anterior; los artefactos no se borran.

## Limitaciones

- Los umbrales de PSI (0,10 y 0,25) son una convención de la industria, no salen de este problema.
- El simulador no produce resultados reales, así que las simulaciones registradas solo sirven para drift de datos, no para medir desempeño.
- Con un dataset fijo de 2016, este plan es un diseño y no un proceso en operación.
