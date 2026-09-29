# ADR 0001: Regresión logística calibrada como modelo principal

- Estado: aceptada (2026-09-29)
- Decidida **después de ver los resultados de test**

## Contexto

El plan de la Fase 3 fijaba LightGBM como modelo principal y la regresión logística como baseline. Con el protocolo acordado de antemano (hiperparámetros elegidos por PR-AUC en validación, calibración y umbrales en validación, test solo para reportar), LightGBM **no superó** a la regresión logística en test:

| Modelo | PR-AUC val | PR-AUC test | Brier test (calibrado) |
|---|---|---|---|
| Regresión logística | 0,3522 | 0,3337 | 0,1878 |
| LightGBM (30 configuraciones) | 0,3522 | 0,3209 | 0,1891 |

Para entender si la diferencia en test era ruido o una ventaja real, se compararon ambos modelos con origen móvil. Solo se usaron fechas de train y validación, sin tocar test:

| Train hasta | Evalúa hasta | Citas evaluadas | Reg. logística | LightGBM | Diferencia (LGBM − RL) |
|---|---|---|---|---|---|
| 2016-05-06 | 2016-05-13 | 14.006 | 0,3760 | 0,3830 | +0,007 |
| 2016-05-13 | 2016-05-20 | 14.211 | 0,3859 | 0,3852 | −0,001 |
| 2016-05-20 | 2016-05-31 | 10.671 | 0,3522 | 0,3522 | 0,000 |

En estos cortes, LightGBM usa los mejores hiperparámetros de validación y hace early stopping en el mismo bloque que se evalúa, lo que lo favorece un poco. Aun así, no hay un ganador consistente. LightGBM además sobreajusta: su PR-AUC en train es 0,459, contra 0,392 de la regresión logística.

## Decisión

El modelo principal pasa a ser la **regresión logística calibrada** (Platt, ajustado en validación). LightGBM se sigue entrenando y reportando como comparación, pero no se guarda como artefacto.

Justificación: en validación y en los cortes temporales los dos modelos empatan. Ante un empate se prefiere el modelo más simple, más estable y más explicable:
- Las contribuciones SHAP de un modelo lineal son exactas y fáciles de auditar.
- Tiene menos hiperparámetros.
- El artefacto es más pequeño.

## Honestidad del proceso

- La decisión se tomó **después** de ver test. Por eso test ya no es una estimación sin sesgo del modelo elegido. El sesgo es pequeño, porque la elección se justifica con evidencia que no viene de test, pero existe y se declara.
- **No se hizo otra selección de LightGBM contra test.** No se probaron más configuraciones ni más regularización después de ver test.
- Los números de test de ambos modelos se reportan tal cual.

## Consecuencias

- `metadata.json` guarda `principal_model`, `model_decision` (con `decidido_despues_de_ver_test: true`), los umbrales y los costos. El simulador carga el modelo con `noshow_guard.model.load_model()`, que lee el archivo indicado en la metadata.
- Las explicaciones SHAP se calculan en log-odds sin calibrar como `coef × (x − media de train)` y se suman por feature original.
- Limitación heredada del modelo lineal: los conteos de historial entran escalados de forma lineal. Un paciente con muchas citas previas (por ejemplo 56) recibe una contribución muy negativa, porque el modelo extrapola más allá de los casos frecuentes en train.
