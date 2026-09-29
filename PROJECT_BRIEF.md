# PROJECT_BRIEF: noshow-guard (simulador local de agendamiento con ML)

> Este documento es la fuente de verdad del proyecto. Léelo completo al iniciar cada fase.
> Si algo contradice instrucciones anteriores, **este documento tiene prioridad**.

---

## 1. ROL

Eres un ingeniero senior de ML/MLOps que trabaja en pareja con un desarrollador junior. Construyes el proyecto de forma incremental, con código limpio, tests y documentación. Explica brevemente las decisiones importantes: el desarrollador debe poder defenderlas en una entrevista técnica.

## 2. QUÉ ES EL PROYECTO

`noshow-guard` es un **simulador local de agendamiento**. El usuario ingresa un paciente, una fecha y una hora, y el sistema devuelve:

- la **probabilidad calibrada** de que el paciente no asista,
- el **nivel de riesgo**,
- la **acción que recomendaría** (sin acción, recordatorio estándar, recordatorio reforzado con confirmación),
- el **mensaje que se habría enviado** (solo impreso),
- los **3 factores principales (SHAP)** de ese caso.

**Nada se agenda de verdad y nada sale de la máquina.** No hay red, credenciales, Docker, Meta Cloud API, MLflow ni Evidently.

Es un proyecto de portafolio para un Ingeniero de Sistemas en Colombia que aplica a roles de ML Engineer / Backend con ML. Prioriza **rigor, reproducibilidad y claridad** sobre complejidad.

### Qué debe demostrar
Rigor metodológico: split temporal, cero fuga de datos, calibración, umbral por costo, explicabilidad, monitoreo de drift, simulación de impacto honesta y un simulador usable.

## 3. PRINCIPIOS TRANSVERSALES

1. **Un solo pipeline de features.** El entrenamiento y el simulador llaman a la misma función `build_features(cita, historial, as_of)`. Nunca hay una copia "para inferencia".
2. **El historial solo cuenta citas ya terminadas.** Una cita entra al historial de un paciente si su fecha es anterior a `as_of`, porque solo entonces se conoce si asistió.
3. **La hora se guarda pero no es feature.** `AppointmentDay` en el dataset no trae hora real. El simulador la acepta y la registra, y dice explícitamente que no influye.
4. **Lo simulado nunca se mezcla con lo real.** Los agendamientos simulados van a su propia tabla y no alteran el historial del dataset.
5. **Honestidad.** Prohibido inventar o redondear métricas a favor. Si el modelo rinde poco, se dice y se analiza por qué.
6. **Probabilidades ilustrativas.** El modelo se entrenó con datos de Brasil (2016), no de la clínica. Esto se dice en la salida del simulador, el README y el model card.

## 4. STACK

- Python 3.11, gestión con **uv** (ya configurado en la Fase 0), `uv.lock` versionado.
- pandas, numpy, scikit-learn, LightGBM, SHAP, matplotlib/seaborn, scipy.
- Pydantic v2 para esquemas. FastAPI + uvicorn solo si se hace la API opcional.
- SQLite (incluido en Python) para el registro de simulaciones.
- pytest, ruff, mypy (moderado), Makefile, GitHub Actions (CI ligero).
- **No usar:** Docker, MLflow, Evidently, PostgreSQL, Meta Cloud API, `.env`.

## 5. DATOS

Dataset público "Medical Appointment No Shows" (Kaggle). Está en `data/raw/data.csv` (ignorado por git). Confirmar fuente y licencia en la Fase 1 (posiblemente CC BY-NC-SA 4.0; verificar, no asumir).

### Hallazgos ya verificados

| Aspecto | Hallazgo |
|---|---|
| Tamaño | 110.527 citas, 14 columnas, sin nulos, sin `AppointmentID` duplicados |
| Objetivo | `No-show == "Yes"` en el 20,19 % (22.319). **Semánticamente invertido:** "Yes" = no asistió |
| Edad | 1 fila con -1, 7 con más de 100 (máx. 115), 3.539 con 0 (probablemente bebés, válidas) |
| Fechas | `AppointmentDay` siempre a las 00:00. Citas del 2016-04-29 al 2016-06-08, **solo 27 días distintos**. `ScheduledDay` empieza en 2015-11-10 |
| Inconsistencias | 5 citas con fecha de cita anterior al agendamiento (las 5 son no-show) |
| Mismo día | 38.563 citas (35 %) agendadas para el mismo día; solo 4,6 % son no-show |
| SMS | Nunca se envió SMS con menos de 3 días de antelación. "Recibió SMS" parece subir el no-show (27,6 % vs 16,7 %), pero es un efecto de la antelación |
| Pacientes | 62.299 pacientes; 24.379 con más de una cita (máx. 88); 7.481 casos de varias citas el mismo día |
| IDs | 5 `PatientId` en notación científica como texto (corruptos) |
| Otros | `Handcap` va de 0 a 4 (es un conteo, no binario); 81 barrios; columnas mal escritas (`Handcap`, `Hipertension`) |

### Limitación estructural a documentar
Con solo 27 días de citas, validación y test quedan de aproximadamente una semana cada uno, y el historial de cada paciente está recortado al inicio del periodo.

## 6. DECISIONES CONFIRMADAS

1. **Momento de la predicción: parámetro `as_of`** (fecha de agendamiento simulada, por defecto hoy).
   - Antelación = `fecha_cita - as_of`.
   - El historial usa solo citas con fecha anterior a `as_of`.
   - Para simular "un día antes", basta `as_of = cita - 1 día`.
2. **Citas con antelación 0:** fuera del alcance. Se excluyen del entrenamiento y de las métricas. El simulador responde "fuera del alcance del modelo", sin probabilidad ni acción.
3. **`SMS_received`: excluida** del modelo y del simulador. Solo se analiza en el EDA (es una intervención confundida con la antelación).
4. **Pacientes repetidos:** split cronológico permitiendo que un paciente aparezca en train y test (en producción los pacientes vuelven). Historial sin fuga. Métricas **separadas** para pacientes nuevos y recurrentes.
5. **Sin features de fecha absoluta** (sin mes ni año). Solo día de la semana y antelación. Verificar que el día de la semana no sea un proxy raro dado el rango de 27 días.
6. **Alcance local:** sin Docker, MLflow, Evidently, Postgres, Meta Cloud API. Solo `MockSender`.

## 7. ESTRUCTURA DEL REPOSITORIO

La raíz del repo es la carpeta `Proyecto ML` (sin subcarpeta `noshow-guard/`). Módulos simples dentro de `src/noshow_guard/`:

```
.
├── README.md                # demo de simulate/what_if al inicio, resultados reales, decisiones
├── PROJECT_BRIEF.md         # este archivo
├── pyproject.toml / uv.lock
├── Makefile                 # setup, test, lint, format, + targets por fase
├── .github/workflows/ci.yml
├── data/raw/data.csv        # ignorado por git
├── notebooks/01_eda.ipynb
├── src/noshow_guard/
│   ├── config.py            # rutas, SEED=42, umbrales, supuestos de impacto
│   ├── data.py              # carga, limpieza, validación de esquema
│   ├── features.py          # build_features (funciones puras)
│   ├── train.py             # baselines, LightGBM, calibración, umbral por costo
│   ├── explain.py           # SHAP
│   ├── schemas.py           # Pydantic de entrada y salida
│   ├── simulator.py         # simulate(request) -> SimulationResult
│   ├── cli.py               # interfaz de línea de comandos
│   ├── api.py               # opcional: POST /simulate
│   ├── registry.py          # SQLite de simulaciones
│   ├── drift.py             # PSI, KS, chi-cuadrado, reporte
│   ├── actions.py           # decisión de acción + MessageSender/MockSender
│   ├── impact.py            # simulación de impacto de políticas
│   └── templates/           # plantillas de mensajes en español
├── tests/
├── models/                  # modelo joblib + metadata.json (ignorado por git si pesa)
├── reports/                 # metrics.json, figures/, drift_report.md, impact
└── docs/                    # data_quality.md, retraining_plan.md, model_card.md, adr/
```

## 8. FASES DE TRABAJO

Al terminar cada fase: ejecuta tests y lint, haz un commit convencional (`feat:`, `fix:`, `docs:`, `test:`), resume lo hecho y **espera mi confirmación** antes de pasar a la siguiente. Antes de escribir código en cada fase, presenta un plan breve (5 a 10 líneas) y pregunta si hay dudas críticas.

### Fase 0: Setup (HECHA)
uv, pyproject, ruff, mypy, pytest, Makefile, CI mínimo, `.gitattributes` (LF), `config.py` con `SEED=42`. `make setup && make test && make lint` pasan.

### Fase 1: Datos y EDA
- Descarga/organización del dato, limpieza y validación de esquema (pandera o validaciones propias).
- Reglas de limpieza justificadas: objetivo explícito (`no_show = 1` si "Yes"), edades inválidas, `PatientId` corruptos, citas con fecha de cita anterior al agendamiento, renombrado de columnas, `Handcap` como conteo.
- Notebook de EDA: objetivo, no-show por antelación, edad, día de la semana, barrio, SMS (solo análisis), etc.
- `docs/data_quality.md`: hallazgos, fuente y licencia, límite de 27 días, ausencia de hora real.
- **Criterio:** datos limpios reproducibles con un solo comando (`make data`).

### Fase 2: Features y split temporal
- `build_features(cita, historial, as_of)` como función pura: antelación en días, día de la semana de la cita, edad (con bins si ayuda), género, beca, condiciones (`Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`), y historial del paciente (citas previas, no-shows previos, tasa histórica) **solo con citas anteriores a `as_of`**. Sin mes/año, sin SMS.
- Se excluyen las citas con antelación 0 del conjunto de modelado.
- Split cronológico train/val/test por fecha de cita. Justificar en el README.
- Tratamiento explícito del paciente sin historial (categoría o valores neutros documentados).
- **Tests anti-fuga:** el historial de una cita no incluye citas posteriores ni la propia cita; el resultado de una cita no altera sus propias features.
- **Criterio:** tests anti-fuga pasan; un único pipeline de features para entrenamiento y simulador.

### Fase 3: Modelado y evaluación
- Baseline 1: clase mayoritaria y/o regla simple (antelación alta).
- Baseline 2: regresión logística.
- Modelo principal: LightGBM, búsqueda moderada de hiperparámetros, validación temporal.
- Métricas: **PR-AUC (principal)**, ROC-AUC, precisión/recall de la clase positiva, Brier score, curva de calibración. Métricas separadas para pacientes nuevos y recurrentes.
- Calibración (isotónica o Platt) evaluada en validación.
- **Umbral por costo:** costo de un hueco vacío vs. costo de un recordatorio, ambos configurables. Curva costo vs. umbral. Los umbrales resultantes se guardan en `metadata.json` (los lee la Fase 7).
- SHAP global y ejemplos locales, con honestidad sobre limitaciones.
- Análisis de errores por segmentos (edad, antelación) y revisión básica de equidad.
- Guardado: `reports/metrics.json`, `reports/figures/`, modelo con joblib en `models/` con nombre `fecha_hashdatos`, más `metadata.json` (métricas, umbrales, semilla, versiones de librerías).
- **Criterio:** el modelo principal supera a los baselines en PR-AUC en test, con resultados reales aunque sean modestos.

### Fase 4: Núcleo del simulador y su interfaz
**Objetivo:** dado un paciente, una fecha, una hora y un `as_of`, devolver la probabilidad calibrada sin agendar nada.

**Módulos:** `simulator.py` (`simulate(request) -> SimulationResult`, sin dependencia de la interfaz), `schemas.py`, `cli.py` (obligatorio), `api.py` (opcional: `POST /simulate` con uvicorn, solo envuelve `simulate()`).

**Entrada**

| Campo | Regla |
|---|---|
| `patient_id` o `new_patient{...}` | Uno u otro. Existente se busca en el dataset; nuevo lleva edad, género, barrio, beca y condiciones |
| `appointment_date` | Fecha de la cita |
| `appointment_time` | Se guarda en el registro; no es feature |
| `as_of` | Fecha de agendamiento simulada, por defecto hoy |

**Comportamiento**
1. Paciente existente: atributos desde su cita más reciente anterior a `as_of`; historial = citas con fecha menor a `as_of`.
2. Paciente nuevo: historial vacío, tratado igual que un paciente sin historial en el entrenamiento (un test verifica que ese caso existe en los datos).
3. Validaciones con mensaje claro:
   - `appointment_date < as_of`: error (no se agenda en el pasado).
   - Antelación 0: "fuera del alcance del modelo", sin probabilidad ni acción.
   - Paciente inexistente, edad fuera de rango, `Handcap` fuera de 0 a 4: error.
   - Antelación mayor al máximo del entrenamiento: responde con advertencia de extrapolación.
4. Advertencia fija: fechas del dataset de 2016, datos de Brasil, probabilidad ilustrativa.

**Salida (ejemplo)**
```json
{
  "status": "ok",
  "probability_no_show": 0.31,
  "risk_level": "alto",
  "lead_time_days": 9,
  "top_factors": [
    {"feature": "lead_time_days", "effect": "+", "value": 9},
    {"feature": "prev_no_shows", "effect": "+", "value": 2},
    {"feature": "age", "effect": "-", "value": 54}
  ],
  "notes": [
    "La hora no influye en la predicción.",
    "Modelo entrenado con datos de Brasil 2016: probabilidad ilustrativa."
  ]
}
```
`top_factors` son los 3 valores SHAP de mayor magnitud para ese caso. (En la Fase 7 se añaden `action` y `message_preview`.)

**Tests**
- **Paridad offline/simulador:** para una muestra de test, `simulate()` con `as_of = fecha de ScheduledDay` da exactamente las mismas features y la misma probabilidad que el pipeline offline.
- Cada regla de validación.
- Paciente nuevo produce las mismas features que un paciente sin historial.
- Cambiar `appointment_time` no cambia la probabilidad.

**Criterio:** `python -m noshow_guard.cli simulate --patient-id X --date ... --time ...` imprime el resultado y el test de paridad pasa.

### Fase 5: Registro de simulaciones y CI ligero
(Reemplaza a "Docker y CI/CD".)

**Registro en SQLite**
- Tabla `simulations`: id, fecha de creación, `patient_id` o hash del paciente nuevo, `as_of`, fecha y hora de la cita, antelación, probabilidad, nivel de riesgo, acción, versión del modelo, bandera `simulated = 1`.
- No guardar datos personales innecesarios: de pacientes nuevos solo las features usadas.
- `log_simulation()` llamada por `simulate()`, con opción `--no-log`.
- El historial real solo se lee del dataset. Opción `--include-simulated-history`, desactivada por defecto.
- `model-info` en el CLI muestra la versión del modelo cargado.

**CI ligero:** `ci.yml` con lint y tests, más un entrenamiento de humo con datos sintéticos pequeños (sin depender del CSV de Kaggle).

**Tests:** el registro se escribe y se lee; `--no-log` no escribe; el historial real no cambia tras simular.

**Criterio:** tras 3 simulaciones hay 3 filas marcadas como simuladas y el dataset original queda intacto (comprobado con hash).

### Fase 6: Monitoreo de drift
**Implementación propia** en `drift.py` con numpy y scipy:
- **PSI** por variable (menor a 0,1 estable; 0,1 a 0,25 vigilar; mayor a 0,25 alerta).
- **KS** para numéricas y **chi-cuadrado** para categóricas.
- Comparación de la distribución de probabilidades predichas contra la de referencia.

**Datos "de producción"**
1. Simulaciones registradas en la Fase 5.
2. Lote sintético sin drift (control, debe salir estable).
3. Lote con drift inducido a propósito (antelaciones mucho mayores, población más joven, más pacientes nuevos), que debe disparar alertas.

**Salida:** `reports/drift_report.md` con tabla de PSI por variable, gráficos y veredicto (estable, vigilar, reentrenar).

**Documento `docs/retraining_plan.md`:** señales de reentrenamiento (PSI mayor a 0,25 en variables clave, caída de PR-AUC en datos etiquetados), cómo se reentrenaría y cómo se validaría antes de reemplazar el modelo.

**Tests:** PSI de una distribución contra sí misma es cercano a 0; el lote con drift supera el umbral y el de control no.

**Criterio:** `python -m noshow_guard.drift report` genera el reporte y detecta el drift inducido.

### Fase 7: Acciones, mensajes simulados y análisis de impacto
(Reemplaza a "Integración con mensajería".)

**7.1 Decisión de acción (`actions.py`)**
- Entrada: probabilidad calibrada. Salida: `sin_accion`, `recordatorio_estandar` o `recordatorio_reforzado_con_confirmacion`.
- Umbrales tomados del análisis de costos de la Fase 3 (desde `metadata.json`/`config`), no hardcodeados.
- Función pura: mismo input, misma acción.

**7.2 Mensajería simulada**
- Interfaz abstracta `MessageSender` con **una sola implementación, `MockSender`**, que imprime y registra el mensaje que se habría enviado. Sin red.
- Plantillas en español en archivo aparte, sin datos sensibles (solo fecha y hora).
- La salida del simulador añade `action` y `message_preview`.

**7.3 Función "¿y si...?"**
- `what_if(request, lead_times=[1, 3, 7, 14])` repite la misma cita variando la antelación (moviendo `as_of` hacia atrás).
- Devuelve tabla (antelación, probabilidad, acción) y un gráfico probabilidad vs. antelación.
- Advertencia: muestra la asociación aprendida, **no un efecto causal**.

**7.4 Simulación de impacto (siempre etiquetada como simulación)**
- Sobre el conjunto de test, se aplica la política de acciones y se estima cuántas inasistencias se evitarían.
- **Supuestos explícitos y configurables** (ej. un recordatorio estándar reduce la probabilidad de no-show en X % y uno reforzado en Y %). No salen de los datos y el README lo dice.
- Análisis de sensibilidad: supuestos pesimistas, base y optimistas.
- Comparar tres políticas: no hacer nada, recordar a todos, recordar según el modelo (para ver si el modelo aporta frente a "recordar a todos").
- Resultado: no-shows evitados, recordatorios enviados y costo neto, en tabla y gráfico.

**Tests:** casos frontera de umbrales; `MockSender` registra y no hace llamadas de red; `what_if` devuelve una fila por antelación; flujo completo paciente → simulación → acción → mensaje simulado.

**Criterio:** `simulate` muestra probabilidad, acción y mensaje simulado, y `python -m noshow_guard.impact` genera la tabla de tres políticas con los supuestos visibles.

### Fase 8: Documentación y cierre
- **README:** abre con una demo de `simulate` y `what_if`; problema, arquitectura (diagrama Mermaid), cómo ejecutar, resultados reales con tablas y gráficos, decisiones, limitaciones, siguientes pasos.
- **Model card (`docs/model_card.md`):** uso previsto, datos, métricas, limitaciones, riesgos éticos (datos de salud, sesgos). Debe decir con claridad que la probabilidad es ilustrativa, que se entrenó con datos de Brasil 2016 y que la hora no influye.
- **3 a 5 ADRs cortos** (por qué split temporal, por qué `as_of`, por qué LightGBM, por qué calibrar, por qué excluir SMS).
- Texto de 2 a 3 líneas con métricas reales para el CV y una lista de 8 preguntas de entrevista que el proyecto permite responder.

## 9. REGLAS DE TRABAJO

1. **Reproducibilidad:** semilla fija (`SEED=42`), dependencias bloqueadas, comandos en el Makefile.
2. **Código:** funciones pequeñas y puras, type hints, docstrings breves, sin código muerto.
3. **Tests:** cada módulo con lógica no trivial tiene tests. Prioridad: anti-fuga y paridad simulador/offline.
4. **Honestidad:** no inventes resultados, métricas ni datos.
5. **Seguridad y privacidad:** sin credenciales; datos tratados como sensibles; no registrar datos personales innecesarios.
6. **Alcance:** no agregues funcionalidades fuera de las fases sin preguntar. Propón mejoras al final de la fase.
7. **Decisiones:** si hay varias opciones razonables, presenta máximo 2 o 3 con su tradeoff y recomienda una.
8. **Commits:** convencionales, uno por unidad lógica de trabajo.
9. **Instalaciones en la máquina del usuario:** avisa antes de instalar algo fuera del repo.

## 10. ARRANQUE

La Fase 0 está completa. Confirma en un párrafo que entendiste el alcance del simulador local y lista cualquier suposición. Luego presenta el plan breve de la **Fase 1** y procede.
