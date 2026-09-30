"""App visual local del simulador (Streamlit). NO guarda ni envía nada.

Uso: ``make app`` o ``uv run streamlit run src/noshow_guard/app.py``, y luego abrir
http://localhost:8501. Solo escucha en esta máquina (``.streamlit/config.toml``).
"""

from __future__ import annotations

import random
from datetime import date, time
from typing import Any

import streamlit as st
from pydantic import ValidationError

from noshow_guard import ui
from noshow_guard.config import DATA_RULES
from noshow_guard.schemas import SimulationRequest, SimulationResult
from noshow_guard.simulator import DEFAULT_LEAD_TIMES, SimulationError, Simulator

MODE_DATASET = "Paciente del dataset"
MODE_NEW = "Mis datos (paciente nuevo)"
OTHER_NEIGHBOURHOOD = "Otro (escribir)"
WHAT_IF_LEADS = (*DEFAULT_LEAD_TIMES, 30, 60)


@st.cache_resource(show_spinner="Cargando modelo y datos...")
def _load_simulator() -> Simulator:
    return ui.read_only_simulator()


def get_simulator() -> Simulator:
    """El simulador de solo lectura (los tests pueden inyectar uno en ``session_state``)."""
    injected = st.session_state.get("simulator")
    return injected if isinstance(injected, Simulator) else _load_simulator()


def validation_text(exc: ValidationError) -> str:
    return "; ".join(err["msg"].removeprefix("Value error, ") for err in exc.errors())


# --- Formularios -----------------------------------------------------------------


def appointment_inputs() -> tuple[date, date, time]:
    st.subheader("1. La cita")
    as_of = st.date_input(
        "Fecha en que se agenda (as_of)",
        value=date(2016, 6, 1),
        help="La predicción se hace con lo que se sabía ese día.",
    )
    appointment_date = st.date_input("Fecha de la cita", value=date(2016, 6, 10))
    appointment_time = st.time_input(
        "Hora de la cita", value=time(9, 0), help="Se muestra, pero no influye en la predicción."
    )
    return as_of, appointment_date, appointment_time


def dataset_patient(sim: Simulator, as_of: date) -> dict[str, Any] | None:
    """Selector de paciente real, con su perfil e historial en ``as_of``."""
    st.subheader("2. El paciente")
    options = ui.sample_patients(sim.records, as_of)
    if not options:
        st.error("No hay pacientes registrados en o antes de esa fecha as_of.")
        return None
    if st.session_state.get("patient_id") is None:
        st.session_state["patient_id"] = options[0]
    if st.button("Elegir un paciente al azar"):
        st.session_state["patient_id"] = random.choice(options)
    patient_id = st.text_input(
        "PatientId (puedes pegar cualquiera del dataset)", key="patient_id"
    ).strip()

    try:
        profile = ui.patient_profile(sim, patient_id, as_of)
    except SimulationError as exc:
        st.error(str(exc))
        return None
    st.table({"Dato": list(profile), "Valor": [str(v) for v in profile.values()]})
    history = ui.patient_history(sim, patient_id, as_of)
    with st.expander(f"Historial que usa el modelo ({len(history)} citas antes de as_of)"):
        if history.empty:
            st.write("Sin citas terminadas antes de as_of: se trata como paciente sin historial.")
        else:
            st.dataframe(history, hide_index=True)
    return {"patient_id": patient_id}


def new_patient(sim: Simulator) -> dict[str, Any]:
    """Formulario con los datos propios del usuario (no se guardan)."""
    st.subheader("2. Tus datos")
    st.caption("Solo se usan para esta simulación en memoria; no se guardan en ningún lado.")
    age = st.number_input(
        "Edad", min_value=DATA_RULES.age_min, max_value=DATA_RULES.age_max, value=30, step=1
    )
    gender = st.radio(
        "Género",
        ["F", "M"],
        horizontal=True,
        format_func=lambda g: {"F": "Mujer", "M": "Hombre"}[g],
    )
    neighbourhoods = [*sorted(sim.known_neighbourhoods), OTHER_NEIGHBOURHOOD]
    choice = st.selectbox("Barrio (de Vitória, Brasil, como en el dataset)", neighbourhoods)
    neighbourhood = (
        st.text_input("Escribe el barrio", value="MI BARRIO")
        if choice == OTHER_NEIGHBOURHOOD
        else choice
    )
    cols = st.columns(2)
    flags = {
        "scholarship": cols[0].checkbox("Beca (Bolsa Família)"),
        "hypertension": cols[0].checkbox("Hipertensión"),
        "diabetes": cols[1].checkbox("Diabetes"),
        "alcoholism": cols[1].checkbox("Alcoholismo"),
    }
    handicap = st.number_input(
        "Discapacidad (conteo 0-4)", min_value=0, max_value=DATA_RULES.handicap_max, value=0
    )
    return {
        "new_patient": {
            "age": int(age),
            "gender": gender,
            "neighbourhood": neighbourhood,
            "handicap": int(handicap),
            **flags,
        }
    }


# --- Resultado -------------------------------------------------------------------


def show_result(sim: Simulator, request: SimulationRequest, result: SimulationResult) -> None:
    st.subheader("Resultado")
    if result.status == "fuera_de_alcance":
        st.info(" ".join(result.notes))
        return
    p = float(result.probability_no_show or 0.0)
    cols = st.columns(3)
    cols[0].metric("Probabilidad de no asistir", ui.percent(p))
    cols[1].metric("Nivel de riesgo", ui.RISK_LABELS.get(result.risk_level or "", "-"))
    cols[2].metric("Antelación", f"{result.lead_time_days} días")
    st.progress(min(max(p, 0.0), 1.0), text=f"Probabilidad calibrada: {ui.percent(p)}")

    st.markdown(f"**Acción recomendada:** {ui.ACTION_LABELS.get(result.action or '', '-')}")
    if result.message_preview:
        st.info(
            f"Mensaje que se habría enviado (simulado, NO se envía):\n\n{result.message_preview}"
        )

    st.markdown("**Por qué (3 factores principales, SHAP)**")
    st.dataframe(ui.factors_frame(result), hide_index=True)
    for warning in result.warnings:
        st.warning(warning)

    with st.expander("¿Y si se agendara con otra antelación?"):
        table = ui.what_if_frame(sim.what_if(request, WHAT_IF_LEADS))
        if not table.empty:
            st.line_chart(table, x="Antelación (días)", y="Probabilidad")
            st.dataframe(table, hide_index=True)
        st.caption(
            "Asociación aprendida por el modelo, NO un efecto causal. Para un paciente del "
            "dataset, mover la fecha de agendamiento también cambia su historial disponible."
        )
    with st.expander("Variables exactas que vio el modelo"):
        st.dataframe(ui.features_frame(result), hide_index=True)
    for note in result.notes:
        st.caption(note)


def main() -> None:
    st.set_page_config(page_title="noshow-guard: simulador", layout="wide")
    st.title("Simulador de inasistencia a citas")
    st.warning(
        "Simulación local: nada se agenda, se guarda ni se envía. El modelo se entrenó con "
        "citas de Brasil (2016), así que la probabilidad es ilustrativa."
    )
    try:
        sim = get_simulator()
    except FileNotFoundError:
        st.error("Faltan el modelo o los datos. Ejecuta: make data, make features y make train.")
        st.stop()

    mode = st.radio("¿A quién le programas la cita?", [MODE_DATASET, MODE_NEW], horizontal=True)
    form, output = st.columns([1, 1.4], gap="large")
    with form:
        as_of, appointment_date, appointment_time = appointment_inputs()
        patient = dataset_patient(sim, as_of) if mode == MODE_DATASET else new_patient(sim)
        run = st.button("Simular cita", type="primary", disabled=patient is None)

    with output:
        if not run or patient is None:
            st.caption("Completa los datos y presiona «Simular cita».")
            return
        try:
            request = SimulationRequest.model_validate(
                {
                    **patient,
                    "appointment_date": appointment_date,
                    "appointment_time": appointment_time,
                    "as_of": as_of,
                }
            )
            result = sim.evaluate(request)  # evaluate(): sin registro ni envío
        except ValidationError as exc:
            st.error(validation_text(exc))
            return
        except SimulationError as exc:
            st.error(str(exc))
            return
        show_result(sim, request, result)


main()
