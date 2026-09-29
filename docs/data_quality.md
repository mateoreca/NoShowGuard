# Calidad de datos

## Fuente y licencia

- Dataset: **Medical Appointment No Shows**, de Joni Hoppen / Aquarela Analytics, en Kaggle:
  <https://www.kaggle.com/datasets/joniarroba/noshowappointments> (versión 5, actualizada el 2017-08-20).
- Licencia: **CC BY-NC-SA 4.0**, verificada en los metadatos de la API pública de Kaggle el 2026-09-29.
  Hay que atribuir la fuente, el uso no puede ser comercial y las obras derivadas se comparten con la misma licencia.
  Este proyecto es de portafolio y no redistribuye el CSV.
- Archivo usado: `data/raw/data.csv` (10.739.535 bytes). SHA-256:
  `9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d`.
  `scripts/download_data.py` comprueba este hash y explica cómo descargarlo a mano.
- Contexto: 110.527 citas del sistema público de salud de Brasil, de 2016. Por los nombres de los barrios parecen ser de Vitória (Espírito Santo), aunque Kaggle no lo dice.

## Reproducir

```bash
make data   # verifica el hash, limpia, valida y escribe data/processed/appointments_clean.parquet
make eda    # ejecuta notebooks/01_eda.ipynb
```

El conteo de filas eliminadas queda en `reports/data_cleaning.json`.

## Reglas de limpieza

Se aplican en este orden en `noshow_guard.data.clean`:

| # | Regla | Filas | Justificación |
|---|---|---|---|
| 1 | Renombrar columnas a snake_case y corregir errores (`Hipertension` → `hypertension`, `Handcap` → `handicap`) | 0 | Legibilidad y consistencia |
| 2 | Objetivo explícito: `no_show = 1` si `No-show == "Yes"` | 0 | En el CSV, "Yes" significa **no asistió**, y es fácil leerlo al revés |
| 3 | Eliminar `PatientId` no numéricos | 5 | Vienen en notación científica como texto (ej. `9377952927E-5`). El ID original se perdió y no se puede enlazar su historial |
| 4 | Eliminar edades fuera de [0, 110] | 6 | 1 fila con -1 y 5 con 115 (de 2 pacientes). Se conservan las edades 0 (bebés, 3.539 filas) y hasta 102 |
| 5 | Eliminar citas con fecha anterior a la de agendamiento | 5 | Son imposibles, y las 5 son no-show, así que la etiqueta tampoco es confiable |
| 6 | `handicap` se mantiene como conteo 0-4 | 0 | El diccionario de Kaggle dice que es True/False, pero los datos traen valores de 0 a 4 |

Resultado: de 110.527 filas pasan **110.511** (se eliminan 16, el 0,014 %).

Las citas del mismo día (antelación 0, 38.559 filas) **se conservan** en los datos limpios. Salen del modelado en la Fase 2, porque no dan tiempo para enviar un recordatorio y casi siempre se cumplen (4,6 % de no-show).

## Esquema validado

`noshow_guard.data.validate` revisa lo siguiente sobre el parquet limpio:

- Columnas en el orden esperado y con sus dtypes.
- Sin nulos y sin `appointment_id` duplicados.
- `patient_id` numérico y `gender` dentro de {F, M}.
- Edad dentro de [0, 110] y `handicap` dentro de [0, 4].
- Columnas binarias en {0, 1}.
- `appointment_date` sin hora y nunca anterior a `scheduled_date`.

Se usan validaciones propias en lugar de pandera porque son pocas reglas y así se evita una dependencia más.

## Hallazgos y limitaciones

- **La cita no tiene hora.** `AppointmentDay` siempre viene a las 00:00, mientras que `ScheduledDay` sí trae hora. La antelación se calcula en **días calendario** (`appointment_date - scheduled_date`, ambas normalizadas a fecha), nunca en horas.
- **La ventana es de 27 días.** Las citas van del 2016-04-29 al 2016-06-08, en 27 fechas. Faltan días hábiles (23, 26 y 27 de mayo), y solo hay un sábado, con 31 citas en alcance. El agendamiento empieza en 2015-11-10. Por eso validación y test quedarán de más o menos una semana cada uno.
- **El historial está recortado.** Solo el 22,8 % de las citas en alcance tiene alguna cita previa del mismo paciente ya terminada antes del agendamiento. Las citas anteriores al 2016-04-29 no están en el dataset.
- **La tasa base cambia con el alcance.** El no-show es del 20,19 % en total y del 28,52 % cuando la antelación es mayor a 0.
- **La tasa varía por fecha**, entre 24 % y 33 %. La última semana (6 al 8 de junio) está entre las más bajas, así que el conjunto de test puede tener una tasa base distinta a la de train.
- **El SMS está confundido con la antelación.** Nunca hay SMS con menos de 3 días de antelación. En crudo, con SMS el no-show parece mayor (27,6 % contra 16,7 %). Al estratificar por antelación pasa lo contrario, pero no es un efecto causal, porque el SMS no se asignó al azar. Por eso queda fuera del modelo.
- **Hay citas múltiples.** 7.479 casos de un paciente con varias citas el mismo día, y 1.333 citas que comparten paciente, instante de agendamiento y fecha. Se conservan porque cada una tiene su `appointment_id`. Como el historial solo cuenta citas con fecha **anterior** al agendamiento, las citas hermanas no se filtran entre sí.
- **Barrios pequeños.** De los 80 barrios en alcance, 6 tienen menos de 100 citas y sus tasas son ruidosas.
- **Representatividad.** Los datos son de un solo municipio brasileño en 2016, así que las probabilidades del simulador son ilustrativas para otra clínica.
- **Privacidad.** Aunque el dataset es público, se trata como sensible: el CSV y el parquet no se versionan, y ningún reporte guarda `patient_id`.
