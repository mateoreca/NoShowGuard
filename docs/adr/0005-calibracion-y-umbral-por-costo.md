# ADR 0005: Probabilidades calibradas y umbrales por costo

- Estado: aceptada (Fase 3)

## Contexto

La salida del sistema es una **decisión**: no hacer nada, enviar un recordatorio estándar o enviar uno reforzado con confirmación. Para decidir con costos hace falta una probabilidad en la que se pueda confiar como probabilidad, no solo como ranking. Un umbral fijo de 0,5 no tiene sentido cuando la prevalencia es 26-30 % y los errores cuestan distinto.

## Decisión

1. **Calibrar en validación.** Se comparan isotónica y Platt por Brier fuera de fold (validación cruzada de 5 partes dentro de validación). Ganó Platt: 0,1973, contra 0,1975 con isotónica y 0,1984 sin calibrar. En test, el Brier del modelo principal baja de 0,1890 a 0,1878. Platt no cambia el orden de las predicciones, así que PR-AUC y ROC-AUC se mantienen.
2. **Umbrales que minimizan el costo esperado en validación.** Los parámetros son configurables en `CostConfig` y son **supuestos**: hueco vacío 20, estándar 1, reforzado 3, con efectos de 15 % y 30 %.
   - Con probabilidades calibradas, una acción conviene si `p > costo / (costo_hueco × efecto)`. Eso da umbrales analíticos de 0,33 y 0,67.
   - El umbral estándar elegido en validación es 0,36, y el reforzado queda en 0,67 (el analítico).
   - Los empates se resuelven hacia el umbral analítico.
   - Los umbrales se guardan en `metadata.json`, y de ahí los leen el simulador y las acciones.
3. **Nivel de riesgo aparte de la acción.** El riesgo es un ranking relativo: percentiles 50 y 90 de validación. La acción depende solo de los umbrales por costo.

## Alternativas descartadas

- **Umbral que maximiza F1:** ignora los costos y la asimetría entre errores.
- **Calibración isotónica:** en validación quedó prácticamente empatada con Platt, y produce probabilidades escalonadas.

## Consecuencias

- Un primer juego de costos (hueco 10, efecto 10 %) hacía que recordar nunca conviniera. Un test documenta ese caso.
- Con los supuestos base, la probabilidad máxima en test es 0,527, así que **el reforzado nunca se activa**. No se ajustaron los supuestos para forzarlo.
- El valor de la política depende del efecto real del recordatorio, que no se midió. La simulación de impacto re-optimiza los umbrales en validación para cada escenario de efecto y muestra esa dependencia.
