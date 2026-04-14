import streamlit as st

from riskguard.config import STYLES_PATH
from riskguard.services.risk_model import cargar_y_entrenar_optimizado
from riskguard.ui.dashboard import render_dashboard


def configure_page():
    # Define el marco visual base antes de renderizar cualquier componente.
    st.set_page_config(
        page_title="RiskGuard AI - Sistema Elite",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def load_styles():
    # Inyecta la hoja CSS para personalizar la apariencia por encima de Streamlit.
    with STYLES_PATH.open(encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


def main():
    # Prepara primero la página y los estilos globales.
    configure_page()
    load_styles()

    try:
        # Entrena o recupera del cache el pipeline completo antes de mostrar la UI.
        with st.spinner("⚡ Inicializando sistema elite... (30 segundos)"):
            (
                modelo,
                scaler,
                label_encoders,
                feature_names,
                df,
                metricas,
                info_pred,
            ) = cargar_y_entrenar_optimizado()
    except Exception as exc:
        st.error(f"❌ Error: {exc}")
        return

    # Entrega a la interfaz el modelo y las métricas para la sesión actual.
    render_dashboard(
        modelo,
        scaler,
        label_encoders,
        feature_names,
        df,
        metricas,
        info_pred,
    )
