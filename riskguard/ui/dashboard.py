import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from riskguard.services.risk_model import predecir_riesgo


def render_dashboard(modelo, scaler, label_encoders, feature_names, df, metricas, info_pred):
    acc = metricas["accuracy"]
    recall_alto = metricas["report"]["ALTO"]["recall"]
    precision_alto = metricas["report"]["ALTO"]["precision"]
    f1_alto = metricas["report"]["ALTO"]["f1-score"]

    # Resume las métricas del modelo en el banner principal.
    
    st.markdown(
        f"""
    <div class="hero-banner">
        <div class="hero-left">
            <div class="hero-title">🛡️ RISKGUARD AI</div>
            <div class="hero-subtitle">Sistema Elite de Detección de Riesgo</div>
        </div>
        <div class="hero-metrics">
            <div class="hero-metric">
                <div class="hero-metric-icon">📊</div>
                <div class="hero-metric-value">{acc:.1%}</div>
                <div class="hero-metric-label">Accuracy</div>
            </div>
            <div class="hero-metric">
                <div class="hero-metric-icon">🎯</div>
                <div class="hero-metric-value">{recall_alto:.1%}</div>
                <div class="hero-metric-label">Recall Alto</div>
            </div>
            <div class="hero-metric">
                <div class="hero-metric-icon">✅</div>
                <div class="hero-metric-value">{precision_alto:.1%}</div>
                <div class="hero-metric-label">Precision</div>
            </div>
            <div class="hero-metric">
                <div class="hero-metric-icon">⚡</div>
                <div class="hero-metric-value">{f1_alto:.1%}</div>
                <div class="hero-metric-label">F1-Score</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Separa el flujo entre evaluación interactiva y analítica del sistema.
    tab1, tab2 = st.tabs(["🎯 Evaluar Proyecto", "📊 Análisis del Sistema"])

    with tab1:
        evaluar_proyecto(modelo, scaler, label_encoders, feature_names, info_pred)

    with tab2:
        analisis_sistema(df, modelo, feature_names, metricas)


def evaluar_proyecto(modelo, scaler, label_encoders, feature_names, info_pred):
    st.header("🎯 Evaluación de Proyecto")

    # Agrupa la captura de datos en tres bloques para acelerar la lectura del formulario.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏢 General")
        project_type = st.selectbox(
            "Tipo",
            ["IT", "Construction", "Healthcare", "Manufacturing", "R&D", "Marketing"],
        )
        team_size = st.number_input("Tamaño Equipo", 1, 100, 10)
        budget = st.number_input("Presupuesto ($)", 10000, 10000000, 500000, 10000)
        timeline = st.number_input("Timeline (meses)", 1, 60, 12)
        complexity = st.slider("Complejidad", 1.0, 10.0, 5.0, 0.1)
        stakeholders = st.number_input("Stakeholders", 1, 50, 8)

    with col2:
        st.subheader("👥 Equipo")
        methodology = st.selectbox(
            "Metodología", ["Agile", "Waterfall", "Scrum", "Kanban", "Hybrid"]
        )
        experience = st.selectbox("Experiencia", ["Junior", "Mixed", "Senior", "Expert"])
        past_projects = st.number_input("Proyectos Previos", 0, 20, 2)
        turnover = st.slider("Rotación", 0.0, 1.0, 0.2, 0.05)
        pm_exp = st.selectbox(
            "Exp. PM",
            ["Junior PM", "Mid-level PM", "Senior PM", "Certified PM"],
        )
        colocation = st.selectbox(
            "Ubicación",
            ["Fully Colocated", "Partially Colocated", "Hybrid", "Fully Remote"],
        )

    with col3:
        st.subheader("⚙️ Riesgo")
        dependencies = st.number_input("Dependencias", 0, 10, 2)
        change_freq = st.slider("Frecuencia Cambios", 0.0, 5.0, 1.5, 0.1)
        phase = st.selectbox(
            "Fase",
            ["Initiation", "Planning", "Execution", "Monitoring", "Closure"],
        )
        req_stability = st.selectbox(
            "Estabilidad Requisitos", ["Stable", "Moderate", "Volatile"]
        )
        tech_fam = st.selectbox("Familiaridad Tech", ["Expert", "Familiar", "New"])
        comm_freq = st.slider("Comunicación/semana", 0.0, 10.0, 2.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🛡️ ANALIZAR CON RISKGUARD AI", type="primary"):
        # Completa variables no expuestas con valores neutros para alimentar el modelo completo.
        datos = {
            "Project_Type": project_type,
            "Team_Size": team_size,
            "Project_Budget_USD": budget,
            "Estimated_Timeline_Months": timeline,
            "Complexity_Score": complexity,
            "Stakeholder_Count": stakeholders,
            "Methodology_Used": methodology,
            "Team_Experience_Level": experience,
            "Past_Similar_Projects": past_projects,
            "External_Dependencies_Count": dependencies,
            "Change_Request_Frequency": change_freq,
            "Project_Phase": phase,
            "Requirement_Stability": req_stability,
            "Team_Turnover_Rate": turnover,
            "Technology_Familiarity": tech_fam,
            "Communication_Frequency": comm_freq,
            "Project_Manager_Experience": pm_exp,
            "Team_Colocation": colocation,
            "Budget_per_Month": budget / (timeline + 1),
            "Risk_Score": complexity * (turnover + 0.1),
            "Team_Adequacy": team_size / (complexity + 1),
            "Vendor_Reliability_Score": 0.8,
            "Historical_Risk_Incidents": 1,
            "Regulatory_Compliance_Level": "Medium",
            "Geographical_Distribution": 3,
            "Stakeholder_Engagement_Level": "Medium",
            "Schedule_Pressure": 0.0,
            "Budget_Utilization_Rate": 0.7,
            "Executive_Sponsorship": "Moderate",
            "Funding_Source": "Internal",
            "Market_Volatility": 0.5,
            "Integration_Complexity": 3.0,
            "Resource_Availability": 0.8,
            "Priority_Level": "Medium",
            "Organizational_Change_Frequency": 1.0,
            "Cross_Functional_Dependencies": 3,
            "Previous_Delivery_Success_Rate": 0.75,
            "Technical_Debt_Level": 0.0,
            "Org_Process_Maturity": "Managed",
            "Data_Security_Requirements": "Medium",
            "Key_Stakeholder_Availability": "Moderate",
            "Tech_Environment_Stability": "N/A",
            "Contract_Type": "Fixed-Price",
            "Resource_Contention_Level": "Medium",
            "Industry_Volatility": "Moderate",
            "Client_Experience_Level": "Regular",
            "Change_Control_Maturity": "Basic",
            "Risk_Management_Maturity": "Basic",
            "Documentation_Quality": "Good",
            "Project_Start_Month": 6,
            "Current_Phase_Duration_Months": 3,
            "Seasonal_Risk_Factor": 1.0,
        }

        # Ejecuta la inferencia solo cuando el usuario confirma el análisis.
        with st.spinner("🔍 Analizando con IA..."):
            resultado = predecir_riesgo(
                modelo,
                scaler,
                label_encoders,
                feature_names,
                datos,
                info_pred,
            )

        mostrar_resultados(resultado, datos)


def mostrar_resultados(resultado, datos):
    st.markdown("<br><br>", unsafe_allow_html=True)

    nivel = resultado["nivel"]
    conf = resultado["confianza"]

    # Ajusta los colores de la tarjeta y del indicador según el nivel predicho.
    if nivel == "ALTO":
        risk_class = "risk-alto"
        icon = "🚨"
        color_text = "white"
        gauge_color = "darkred"
        gauge_value = resultado["probabilidades"].get("ALTO", conf) * 100
    elif nivel == "MEDIO":
        risk_class = "risk-medio"
        icon = "⚡"
        color_text = "#2c3e50"
        gauge_color = "#e67e22"
        gauge_value = resultado["probabilidades"].get("MEDIO", conf) * 100
    else:
        risk_class = "risk-bajo"
        icon = "✅"
        color_text = "white"
        gauge_color = "#27ae60"
        gauge_value = resultado["probabilidades"].get("BAJO", conf) * 100

    # Presenta el veredicto principal con una tarjeta visual dominante.
    st.markdown(
        f"""
    <div class="risk-card {risk_class}">
        <div class="risk-title">{icon} RIESGO {nivel}</div>
        <div class="risk-confidence" style="color: {color_text};">Confianza: {conf:.1%}</div>
        <div class="risk-model-info" style="color: {color_text};">RiskGuard AI - Ensemble XGBoost + Random Forest</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Indicador de Riesgo")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=gauge_value,
            number={"suffix": "%", "font": {"size": 48, "color": "#2c3e50"}},
            title={
                "text": f"Probabilidad · Riesgo <b>{nivel}</b>",
                "font": {"size": 20, "color": "#667eea"},
            },
            delta={
                "reference": 50,
                "increasing": {"color": "#e74c3c"},
                "decreasing": {"color": "#27ae60"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 2,
                    "tickcolor": "#667eea",
                    "ticksuffix": "%",
                },
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#dfe6e9",
                "steps": [
                    {"range": [0, 40], "color": "#d5f5e3"},
                    {"range": [40, 70], "color": "#fef9e7"},
                    {"range": [70, 100], "color": "#fde8e8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.85,
                    "value": gauge_value,
                },
            },
        )
    )

    fig_gauge.update_layout(
        height=350,
        margin=dict(t=60, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Probabilidades")
        # Mantiene el mismo color para cada clase en todas las visualizaciones.
        colores = {"ALTO": "#ff6b6b", "MEDIO": "#ffd93d", "BAJO": "#6bcf7f"}

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(resultado["probabilidades"].keys()),
                    y=list(resultado["probabilidades"].values()),
                    marker_color=[
                        colores[k] for k in resultado["probabilidades"].keys()
                    ],
                    text=[
                        f"{v:.1%}" for v in resultado["probabilidades"].values()
                    ],
                    textposition="auto",
                    textfont=dict(size=18, family="Arial Black"),
                )
            ]
        )
        fig.update_layout(
            height=400,
            yaxis=dict(tickformat=".0%"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔍 Top 10 Factores")
        factores_df = pd.DataFrame(resultado["factores"])
        factores_df["importancia"] = factores_df["importancia"].apply(lambda x: f"{x:.4f}")
        factores_df.columns = ["Factor", "Importancia", "Contribución"]
        st.dataframe(factores_df, use_container_width=True, height=400)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🕸️ Salud del Proyecto")

    # Normaliza factores heterogéneos a una escala 0-10 para el radar comparativo.
    exp_map = {"Junior": 2, "Mixed": 5, "Senior": 8, "Expert": 10}
    req_map = {"Volatile": 2, "Moderate": 5, "Stable": 10}

    radar_labels = [
        "Complejidad\n(inv.)",
        "Exp. Equipo",
        "Dependencias\n(inv.)",
        "Cambios\n(inv.)",
        "Estabilidad\nReq.",
        "Comunicación",
    ]

    radar_values = [
        max(0, 10 - datos["Complexity_Score"]),
        exp_map.get(datos.get("Team_Experience_Level", "Mixed"), 5),
        max(0, 10 - datos["External_Dependencies_Count"]),
        max(0, 10 - datos["Change_Request_Frequency"] * 2),
        req_map.get(datos.get("Requirement_Stability", "Moderate"), 5),
        min(10, datos["Communication_Frequency"]),
    ]

    radar_labels_closed = radar_labels + [radar_labels[0]]
    radar_values_closed = radar_values + [radar_values[0]]

    fig_radar = go.Figure()

    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_labels_closed,
            fill="toself",
            fillcolor="rgba(102, 126, 234, 0.25)",
            line=dict(color="#667eea", width=3),
            marker=dict(size=8, color="#764ba2"),
            name="Proyecto",
        )
    )

    fig_radar.add_trace(
        go.Scatterpolar(
            r=[10, 10, 10, 10, 10, 10, 10],
            theta=radar_labels_closed,
            fill="toself",
            fillcolor="rgba(39, 174, 96, 0.07)",
            line=dict(color="#27ae60", width=1, dash="dot"),
            name="Ideal",
        )
    )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[2, 4, 6, 8, 10],
                gridcolor="#dfe6e9",
                linecolor="#dfe6e9",
            ),
            angularaxis=dict(gridcolor="#dfe6e9"),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=450,
        margin=dict(t=40, b=60, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("💡 Plan de Acción")

    # Traduce la predicción a una recomendación operativa inmediata.
    if nivel == "ALTO":
        st.error(
            """
        ### 🚨 ALERTA CRÍTICA

        **Acciones inmediatas (24h):**
        1. 🔴 Reunión de emergencia
        2. 🔴 Equipo de crisis
        3. 🔴 Plan de contingencia
        4. 🔴 Re-evaluación completa

        **Responsable:** CEO/Director  
        **Seguimiento:** Cada 6 horas
        """
        )
    elif nivel == "MEDIO":
        st.warning(
            """
        ### ⚡ ATENCIÓN NECESARIA

        **Acciones (48h):**
        1. 🟡 Revisión semanal
        2. 🟡 Controles adicionales
        3. 🟡 Atender top 5 factores

        **Responsable:** Director Proyectos  
        **Seguimiento:** Semanal
        """
        )
    else:
        st.success(
            """
        ### ✅ PROYECTO SALUDABLE

        **Acciones:**
        1. 🟢 Monitoreo mensual
        2. 🟢 Mantener prácticas
        3. 🟢 Documentar éxitos

        **Responsable:** PM  
        **Seguimiento:** Mensual
        """
        )


def analisis_sistema(df, modelo, feature_names, metricas):
    st.header("📊 Análisis del Sistema")
    st.subheader("📦 Distribución de Riesgos en el Dataset")

    # Muestra si las clases del dataset están equilibradas para entrenar el modelo.
    fig_dist = px.histogram(
        df,
        x="Risk_Level_3",
        color="Risk_Level_3",
        color_discrete_map={
            "ALTO": "#ff6b6b",
            "MEDIO": "#ffd93d",
            "BAJO": "#6bcf7f",
        },
        text_auto=True,
        category_orders={"Risk_Level_3": ["BAJO", "MEDIO", "ALTO"]},
    )

    fig_dist.update_traces(textfont_size=16, textposition="outside")
    fig_dist.update_layout(
        title=dict(
            text="Distribución de clases — muestra si el dataset está balanceado",
            font=dict(size=14, color="#666"),
        ),
        xaxis_title="Nivel de Riesgo",
        yaxis_title="Número de proyectos",
        showlegend=False,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )

    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Métricas Detalladas")
        # Reordena las métricas por clase para inspeccionar precisión, recall y F1 a la vez.
        metricas_df = pd.DataFrame(
            {
                "Clase": metricas["classes"],
                "Precision": [
                    metricas["report"][c]["precision"] for c in metricas["classes"]
                ],
                "Recall": [
                    metricas["report"][c]["recall"] for c in metricas["classes"]
                ],
                "F1": [metricas["report"][c]["f1-score"] for c in metricas["classes"]],
            }
        )

        st.dataframe(
            metricas_df.style.format(
                {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}
            ).background_gradient(cmap="RdYlGn", subset=["Precision", "Recall", "F1"]),
            use_container_width=True,
        )

    with col2:
        st.subheader("🎯 Matriz de Confusión")

        # Cruza valores reales y predichos para localizar errores frecuentes del clasificador.
        fig_cm = px.imshow(
            metricas["confusion_matrix"],
            text_auto=True,
            x=list(metricas["classes"]),
            y=list(metricas["classes"]),
            color_continuous_scale="Blues",
            aspect="auto",
        )

        fig_cm.update_traces(textfont=dict(size=20, family="Arial Black"))

        fig_cm.update_layout(
            title=dict(
                text="Filas = real · Columnas = predicción",
                font=dict(size=13, color="#666"),
            ),
            xaxis_title="Predicción",
            yaxis_title="Real",
            height=380,
            margin=dict(t=60, b=40, l=60, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔍 Features Más Importantes")

    if hasattr(modelo.estimators_[0], "feature_importances_"):
        # Usa el primer estimador del ensemble como referencia rápida de importancia global.
        importancias_df = pd.DataFrame(
            {
                "Factor": feature_names,
                "Importancia": modelo.estimators_[0].feature_importances_,
            }
        ).sort_values("Importancia", ascending=False).head(15)

        fig = px.bar(
            importancias_df,
            x="Importancia",
            y="Factor",
            orientation="h",
            color="Importancia",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)
