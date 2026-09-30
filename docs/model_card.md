# Model card: noshow-guard

> **La probabilidad es ilustrativa.** El modelo se entrenó con citas de un sistema público de salud de Brasil de 2016, no con datos de una clínica colombiana ni actual. **La hora de la cita no influye en la predicción**, porque el dataset no la registra.

## Detalles del modelo

| Campo | Valor |
|---|---|
| Versión | `20260929_c823f970` (`models/metadata.json`) |
| Tipo | Regresión logística (scikit-learn 1.7.2) con calibración Platt |
| Comparación | LightGBM 4.6.0 calibrado; no se usa en producción ([ADR 0001](adr/0001-modelo-principal.md)) |
| Entrada | 14 features de `build_features(cita, historial, as_of)`: antelación, día de la semana, edad, género, beca, hipertensión, diabetes, alcoholismo, discapacidad (conteo), barrio (target encoding) e historial del paciente (`log1p` en conteos, [ADR 0002](adr/0002-log-conteos-historial.md)) |
| Salida | Probabilidad calibrada de no-show, nivel de riesgo relativo, acción recomendada y 3 factores SHAP |
| Semilla | 42; dependencias fijadas en `uv.lock` |

## Uso previsto

- Proyecto de portafolio que **demuestra** un flujo de ML responsable: split temporal, cero fuga, calibración, umbral por costo, explicabilidad, monitoreo e impacto simulado.
- Simulación local: priorizar qué citas futuras (antelación ≥ 1 día) recibirían un recordatorio. **Nada se agenda y nada se envía.**

## Fuera de alcance

- Decisiones reales sobre pacientes sin reentrenar con datos locales, validar y revisar la equidad.
- Citas del mismo día (antelación 0).
- **Negar, restringir o condicionar la atención.** La única acción prevista es enviar un recordatorio. Usar el puntaje para sobre-agendar, penalizar o dar menos prioridad a un paciente queda fuera de uso.
- Interpretar los factores SHAP o la curva "¿y si...?" como causas.

## Datos

- **Fuente:** [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments), CC BY-NC-SA 4.0. Son 110.527 citas, de las que quedan 110.511 tras la limpieza ([data_quality.md](data_quality.md)).
- **Alcance:** 71.952 citas con antelación ≥ 1 día.
- **Split cronológico** ([ADR 0003](adr/0003-split-temporal.md)):

| Split | Fechas | Citas | No-show |
|---|---|---|---|
| Train | 2016-04-29 a 2016-05-20 | 43.937 | 29,6 % |
| Validación | 2016-05-24 a 2016-05-31 | 10.671 | 28,0 % |
| Test | 2016-06-01 a 2016-06-08 | 17.344 | 26,0 % |

- **Excluidos a propósito:** `SMS_received`, mes, año y la hora de la cita ([ADR 0004](adr/0004-as-of-y-features-excluidas.md)).

## Métricas (test)

Todas se calcularon con test, sin redondear a favor. La elección del modelo se hizo después de ver test (ADR 0001), así que test ya no es una estimación completamente limpia.

| Modelo | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Clase mayoritaria | 0,260 | 0,500 | 0,1937 |
| Regla de antelación | 0,290 | 0,549 | n/a |
| **Regresión logística calibrada** | **0,333** | **0,602** | **0,1878** |
| LightGBM calibrado | 0,321 | 0,589 | 0,1891 |

- **Decisión con el umbral 0,36** (recordatorio estándar): se marca el 8,2 % de las citas, con precisión de 0,375 y recall de 0,118.
- **Recordatorio reforzado** (umbral 0,67): nunca se activa en test, porque la probabilidad máxima es 0,527.
- **Calibración:** según la curva de test, el modelo está bien calibrado entre 0,15 y 0,40.

## Análisis por grupos (test, umbral 0,36)

| Grupo | No-show real | % marcado | Recall | FPR |
|---|---|---|---|---|
| Mujer / hombre | 25,7 % / 26,6 % | 8,4 % / 7,7 % | 0,125 / 0,104 | 0,070 / 0,067 |
| Sin beca / con beca | 25,5 % / 31,4 % | 6,4 % / 26,4 % | 0,090 / 0,355 | 0,055 / 0,223 |
| Edad 18-35 | 32,2 % | 18,5 % | 0,221 | 0,167 |
| Edad 56-75 | 18,8 % | 0,2 % | **0,003** | 0,001 |
| Edad 76+ | 20,9 % | 0,0 % | **0,000** | 0,000 |
| Sin historial / con historial | 27,0 % / 24,6 % | 9,3 % / 6,6 % | 0,129 / 0,101 | 0,080 / 0,055 |

## Riesgos éticos

- **Son datos de salud.** Aunque el dataset es público, se trata como sensible:
  - El CSV y los parquet no se versionan.
  - Los reportes no incluyen `PatientId`.
  - De un paciente nuevo, el registro guarda solo un hash y las features usadas.
  - Las plantillas de mensajes solo contienen la fecha y la hora.
- **Sesgo de asignación por beca.** La beca (Bolsa Família) funciona como indicador socioeconómico. Se marca al 26,4 % de las citas con beca contra el 6,4 % sin beca, y el FPR es 4 veces mayor. Esto refleja en parte una tasa real más alta, pero la amplifica.
  - Con una acción de bajo costo, como un recordatorio, el daño es limitado.
  - Con acciones más duras (sobre-agendar, exigir confirmación para mantener el cupo), sería discriminatorio.
- **Casi no detecta a los mayores.** El recall es prácticamente 0 desde los 56 años, aunque ese grupo falta a casi 1 de cada 5 citas. Esas personas no recibirían recordatorios.
- **Retroalimentación.** Si los recordatorios funcionan, las citas marcadas faltarán menos y un modelo reentrenado con esos datos aprendería un riesgo menor para ellas. Reentrenar exige registrar qué acción se tomó.
- **Validez externa.** Brasil 2016 no es Colombia hoy: los patrones de barrio, beca y edad no se transfieren.

## Limitaciones

- **Discriminación modesta** (ROC-AUC 0,60). El modelo prioriza mejor que el azar, pero separa mal casos individuales.
- **Ventana de 27 días.** El historial está recortado (13 % de las citas de train lo tienen, contra 41 % en test) y el monitoreo lo detecta como drift.
- **Beneficio económico** casi nulo bajo los supuestos base (0,2 % frente a no hacer nada). El resultado depende del efecto real del recordatorio, que no se midió.
- **SHAP** explica el modelo lineal en log-odds sin calibrar. Las features de historial están correlacionadas y pueden repartirse el efecto con signos opuestos.

## Mantenimiento

- Monitoreo con `make monitor` (PSI, KS y chi²). Hoy da «estable» en el control y detecta el drift inducido.
- Señales de reentrenamiento y validación campeón/retador en [retraining_plan.md](retraining_plan.md).
