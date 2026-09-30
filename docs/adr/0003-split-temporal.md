# ADR 0003: Split cronológico por fecha de cita

- Estado: aceptada (Fase 2)

## Contexto

El modelo predice si un paciente faltará a una cita **futura** usando lo aprendido de citas **pasadas**. El dataset cubre 27 fechas de cita, del 2016-04-29 al 2016-06-08. La tasa de no-show varía por fecha entre 24 % y 33 %, y 24.377 pacientes tienen más de una cita.

## Opciones

1. **Split aleatorio.** Es simple y deja más datos en cada parte. Pero mezcla citas de la misma semana, e incluso del mismo paciente y el mismo día, entre train y test. Las métricas salen infladas por patrones de fechas concretas, y el conjunto de evaluación deja de representar "el futuro".
2. **Split por paciente (GroupKFold).** Evita repetir pacientes, pero en producción los pacientes **sí** vuelven, así que evaluar solo pacientes nuevos subestima el valor del historial. Tampoco respeta el tiempo.
3. **Split cronológico por fecha de cita** (elegida).

## Decisión

| Split | Fechas | Citas en alcance |
|---|---|---|
| train | 2016-04-29 a 2016-05-20 | 43.937 |
| val | 2016-05-24 a 2016-05-31 | 10.671 |
| test | 2016-06-01 a 2016-06-08 | 17.344 |

- Validación elige hiperparámetros, calibración, umbrales y cortes de riesgo. Test solo se usa para reportar.
- Un paciente puede estar en train y en test. Su historial se calcula siempre sin fuga (ADR 0004), y las métricas se reportan por separado para citas con y sin historial.
- Para comparar modelos sin tocar test se usaron además 3 cortes con origen móvil sobre train + validación (ADR 0001).

## Consecuencias

- Las métricas de test simulan honestamente "entrenar con mayo y predecir junio".
- Validación y test quedan de una semana cada uno, así que las métricas tienen varianza.
- La tasa base baja de 29,6 % en train a 26,0 % en test. La PR-AUC no es comparable entre splits sin considerar la prevalencia.
- El historial disponible se triplica entre train y test (13 % contra 41 %). Es un cambio de distribución real que el monitoreo de drift detecta.
