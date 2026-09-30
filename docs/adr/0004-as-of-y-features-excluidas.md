# ADR 0004: `as_of` como momento de la predicción, y features excluidas (SMS, fechas, hora)

- Estado: aceptada (Fases 2 y 4)

## Contexto

El mayor riesgo del proyecto es la fuga de datos: usar información que no existía en el momento de decidir. Hay tres fuentes concretas:
- El historial del paciente puede incluir citas posteriores, o citas cuyo resultado aún no se conocía.
- `SMS_received` es una intervención que ocurre **después** del agendamiento, y se envió solo a citas con 3 días o más de antelación.
- `AppointmentDay` no trae hora, y las fechas absolutas (mes, año) no se repiten en el futuro.

## Decisión

1. **Un parámetro `as_of`**, que es la fecha de agendamiento simulada. En los datos históricos es la fecha de `ScheduledDay`.
   - Antelación = `fecha_cita − as_of`, en días calendario.
   - El historial solo cuenta citas con fecha **estrictamente anterior** a `as_of`. Una cita del mismo día todavía no terminó.
   - Una sola función, `build_features(cita, historial, as_of)`, sirve al entrenamiento y al simulador, y filtra el historial por sí misma.
2. **Citas con antelación 0 fuera del alcance.** Son el 35 % de las citas, casi siempre se cumplen (4,6 % de no-show) y no dejan tiempo para enviar un recordatorio.
3. **`SMS_received` se excluye.** En crudo parece subir el no-show (27,6 % contra 16,7 %), pero dentro de cada tramo de antelación va asociado a menos no-show. Es confusión con la antelación, y además sería una fuga en un sistema que decide los recordatorios.
4. **Sin mes, año ni hora.** Solo se usa el día de la semana, y el fin de semana se agrupa con el viernes porque el único sábado es una fecha con 31 citas. El simulador acepta la hora, la registra y dice que no influye.
5. **Las simulaciones nunca entran al historial.** Una simulación no tiene resultado real. La opción `--include-simulated-history` se eliminó.

## Consecuencias

- Hay tests anti-fuga: el historial no incluye la propia cita, ni citas posteriores, ni citas del día de `as_of`. Cambiar resultados futuros no altera features pasadas.
- Se verificó por fuerza bruta en 2.000 citas reales, sin diferencias.
- La paridad simulador/offline se verificó en las 17.344 citas de test. Hay 18 diferencias, todas en la edad, por registros del mismo día.
- El simulador puede responder "qué pasaría si se agendara esta cita en la fecha X" (`what-if`), sin mezclar la hora ni las simulaciones con los datos reales.
