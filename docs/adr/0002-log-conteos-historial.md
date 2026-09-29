# ADR 0002: `log1p` en los conteos de historial del modelo lineal

- Estado: aceptada (2026-09-29)
- Decidida con validación; implica una **segunda mirada a test**

## Contexto

En la regresión logística, `prev_appointments` y `prev_no_shows` entraban escalados de forma lineal. El máximo de `prev_appointments` en train es 29. Para un paciente con 56 citas previas, la contribución SHAP de esa sola variable era −2,46 en log-odds: el modelo extrapolaba una recta mucho más allá de los datos. El problema es de comportamiento, no de métricas.

## Regla fijada antes de comparar

Adoptar `log1p` antes del escalado **si no empeora la PR-AUC en más de 0,002**, tanto en validación como en el promedio de los 3 cortes con origen móvil. La comparación se corrió excluyendo test del script.

## Resultado (sin test)

| Variante | PR-AUC val | Cortes con origen móvil | Promedio cortes | SHAP de `prev_appointments = 56` |
|---|---|---|---|---|
| Lineal | 0,35220 | 0,3760 / 0,3859 / 0,3522 | 0,3714 | −2,459 |
| `log1p` | 0,35030 | 0,3760 / 0,3864 / 0,3503 | 0,3709 | −0,701 |

La diferencia es −0,0019 en validación y −0,0005 en el promedio de los cortes. Las dos quedan dentro del margen de 0,002, así que se adopta `log1p`. El Brier en validación es 0,1984 en ambas variantes.

## Consecuencias

- `make_logistic_pipeline` aplica `log1p` y luego escalado a `prev_appointments` y `prev_no_shows`. `prev_no_show_rate`, que ya está en [0, 1], y `handicap` se escalan sin logaritmo.
- **Segunda mirada a test:** al reentrenar, la PR-AUC de test pasó de 0,3337 a 0,3325 y el Brier calibrado se mantuvo en 0,1878. El cambio se mantiene porque la regla se fijó antes y no se revierte por el resultado en test.
- Después del cambio, el umbral estándar elegido en validación pasó de 0,35 a 0,36.
- Las cifras de la comparación con LightGBM en ADR 0001 son las del momento de esa decisión, con conteos lineales. `reports/metrics.json` muestra las actuales.
