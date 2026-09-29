# noshow-guard

Simulador local de agendamiento: dado un paciente, una fecha y una hora, estima la probabilidad calibrada de inasistencia (no-show), recomienda una acción de recordatorio y explica el caso. Nada se agenda de verdad y nada sale de la máquina.

> Estado: Fase 2 (features y split temporal). El README completo se construye en la Fase 8.

## Requisitos

- [uv](https://docs.astral.sh/uv/) 0.8.x (instala Python 3.11 automáticamente)
- GNU Make (en Windows: `winget install ezwinports.make`)

## Inicio rápido

```bash
make setup      # dependencias exactas desde uv.lock
make test
make lint       # ruff + mypy
make data       # verifica hash, limpia y valida -> data/processed/appointments_clean.parquet
make features   # features sin fuga + split cronológico -> data/processed/features/
make eda        # ejecuta notebooks/01_eda.ipynb
```

## Datos

Dataset público [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Kaggle, CC BY-NC-SA 4.0). El CSV va en `data/raw/data.csv` y no se versiona. Las reglas de limpieza, la fuente y las limitaciones están en [docs/data_quality.md](docs/data_quality.md).

## Features

Todas salen de una sola función, `build_features(cita, historial, as_of)` en `src/noshow_guard/features.py`, que usan tanto el entrenamiento como el simulador. `as_of` es el momento de la predicción. En los datos históricos es la fecha de `ScheduledDay`.

| Feature | Definición |
|---|---|
| `lead_time_days` | Días calendario desde `as_of` hasta la cita. No usa horas, porque la cita no las tiene |
| `appointment_weekday` | De 0 (lunes) a 4 (viernes). El fin de semana se agrupa con el viernes, porque el único sábado es una sola fecha con 31 citas |
| `age`, `is_male`, `scholarship`, `hypertension`, `diabetes`, `alcoholism`, `handicap` | Atributos del paciente. `handicap` es un conteo de 0 a 4 |
| `neighbourhood` | Texto crudo. Se codifica con target encoding suavizado dentro del modelo, ajustado solo con train (Fase 3) |
| `has_history`, `prev_appointments`, `prev_no_shows`, `prev_no_show_rate` | Historial del paciente, contando solo citas con fecha **estrictamente anterior** a `as_of`. Si no hay historial, todo vale 0 y `has_history = 0` |

Se excluyen a propósito:
- `sms_received`, porque es una intervención confundida con la antelación.
- Mes y año, porque son fechas absolutas que no generalizan.
- La hora de la cita, que no existe en los datos.
- Las citas del mismo día (antelación 0), que quedan fuera del alcance del modelo.

**Garantías contra la fuga**, cubiertas en `tests/test_features.py`:
- El historial de una cita no incluye la propia cita, ni citas posteriores, ni citas del mismo día de `as_of`.
- Cambiar el resultado de una cita no cambia sus features.
- Cambiar resultados futuros no cambia las features de citas anteriores.
- `build_features` filtra el historial por sí misma, así que pasarle citas futuras no produce fuga.

Además, se recalculó el historial por fuerza bruta en 2.000 citas reales al azar y no hubo ninguna diferencia.

## Split temporal

El split es cronológico por fecha de cita y no aleatorio. El modelo se usará para predecir citas futuras con lo aprendido del pasado, y un split aleatorio mezclaría citas de la misma semana en train y test. Eso infla las métricas con patrones de fechas concretas y deja que el historial de un paciente vea el futuro.

Un mismo paciente puede aparecer en train y en test, porque en producción los pacientes vuelven. Su historial siempre se calcula sin fuga, y las métricas se reportarán por separado para citas con y sin historial.

Resultados de `make features` (`reports/split_summary.json`, solo citas con antelación de 1 día o más):

| Split | Fechas de cita | Citas | % | No-show | Con historial | Paciente visto en train |
|---|---|---|---|---|---|---|
| train | 2016-04-29 a 2016-05-20 (17 días) | 43.937 | 61,1 % | 29,64 % | 13,0 % | 100 % |
| val | 2016-05-24 a 2016-05-31 (4 días) | 10.671 | 14,8 % | 28,01 % | 32,9 % | 44,5 % |
| test | 2016-06-01 a 2016-06-08 (6 días) | 17.344 | 24,1 % | 26,00 % | 41,5 % | 42,0 % |

Hay dos cambios de distribución entre splits, y son reales, no errores:
- **La tasa base baja** de 29,6 % a 26,0 %. Las métricas que dependen de la prevalencia, como PR-AUC, no son comparables entre splits sin tenerlo en cuenta.
- **La proporción de citas con historial se triplica**, del 13 % al 41 %. Es un efecto de la ventana recortada: el dataset empieza el 2016-04-29, así que las citas tempranas casi nunca tienen historial. El modelo aprende las features de historial con relativamente pocos ejemplos.
