import warnings

import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from riskguard.config import TRAINING_DATASET_PATH

warnings.filterwarnings("ignore")


@st.cache_data
def cargar_y_entrenar_optimizado():
    # Carga el dataset y reduce la salida a tres niveles de riesgo comparables.
    df = pd.read_csv(TRAINING_DATASET_PATH)

    def agrupar_riesgo(nivel):
        if nivel == "Low":
            return "BAJO"
        if nivel == "Medium":
            return "MEDIO"
        return "ALTO"

    df["Risk_Level_3"] = df["Risk_Level"].apply(agrupar_riesgo)

    X = df.drop(["Risk_Level", "Risk_Level_3", "Project_ID"], axis=1, errors="ignore")
    y = df["Risk_Level_3"]

    label_encoders = {}
    columnas_categoricas = X.select_dtypes(include=["object"]).columns

    # Codifica texto a enteros para que el pipeline pueda entrenarse con sklearn.
    X_encoded = X.copy()
    for col in columnas_categoricas:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Añade variables derivadas para resumir presión presupuestaria y capacidad del equipo.
    X_encoded["Budget_per_Month"] = (
        X_encoded["Project_Budget_USD"] / (X_encoded["Estimated_Timeline_Months"] + 1)
    )
    X_encoded["Risk_Score"] = X_encoded["Complexity_Score"] * (
        X_encoded["Team_Turnover_Rate"] + 0.1
    )
    X_encoded["Team_Adequacy"] = X_encoded["Team_Size"] / (
        X_encoded["Complexity_Score"] + 1
    )

    feature_names_all = list(X_encoded.columns)
    X_encoded = X_encoded.fillna(X_encoded.mean())

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    label_encoders["Risk_Level"] = le_target

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    # Filtra las variables más útiles y homogeneiza sus escalas antes del entrenamiento.
    selector = SelectKBest(f_classif, k=35)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_mask = selector.get_support()
    feature_names = [
        name for name, selected in zip(feature_names_all, selected_mask) if selected
    ]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Balancea la clase objetivo para que el modelo no se sesgue hacia la mayoría.
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Combina dos modelos distintos para suavizar errores individuales.
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft",
        weights=[2, 1],
    )

    ensemble.fit(X_train_resampled, y_train_resampled)

    y_pred_test = ensemble.predict(X_test_scaled)
    y_pred_proba_test = ensemble.predict_proba(X_test_scaled)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    report = classification_report(
        y_test,
        y_pred_test,
        target_names=le_target.classes_,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred_test)

    metricas = {
        "accuracy": accuracy_test,
        "report": report,
        "confusion_matrix": cm,
        "classes": le_target.classes_,
        "n_features": len(feature_names),
    }

    info_prediccion = {
        "selector": selector,
        "feature_names_all": feature_names_all,
        "selected_mask": selected_mask,
    }

    return (
        ensemble,
        scaler,
        label_encoders,
        feature_names,
        df,
        metricas,
        info_prediccion,
    )


def predecir_riesgo(modelo, scaler, label_encoders, feature_names, datos, info_pred):
    # Reconstruye la entrada respetando el orden de variables usado en entrenamiento.
    X_pred = []

    for feature in info_pred["feature_names_all"]:
        if feature in datos:
            valor = datos[feature]

            if feature in label_encoders:
                try:
                    # Cuando llega una categoría nueva, usa el valor base para no romper la predicción.
                    if str(valor) in label_encoders[feature].classes_:
                        valor_cod = label_encoders[feature].transform([str(valor)])[0]
                    else:
                        valor_cod = 0
                except Exception:
                    valor_cod = 0
                X_pred.append(valor_cod)
            else:
                X_pred.append(float(valor))
        else:
            X_pred.append(0.0)

    X_pred_df = pd.DataFrame([X_pred], columns=info_pred["feature_names_all"])
    X_pred_df = X_pred_df.fillna(0)

    # Reaplica exactamente el mismo selector y escalador usados al entrenar.
    X_pred_selected = info_pred["selector"].transform(X_pred_df)
    X_pred_scaled = scaler.transform(X_pred_selected)

    pred = modelo.predict(X_pred_scaled)[0]
    proba = modelo.predict_proba(X_pred_scaled)[0]

    clase = label_encoders["Risk_Level"].inverse_transform([pred])[0]

    if hasattr(modelo.estimators_[0], "feature_importances_"):
        importancias = modelo.estimators_[0].feature_importances_
    else:
        importancias = np.ones(len(feature_names)) / len(feature_names)

    # Mezcla importancia global y magnitud local para priorizar factores explicativos.
    contribuciones = importancias * np.abs(X_pred_scaled[0])
    top_idx = np.argsort(contribuciones)[-10:][::-1]

    factores = []
    for idx in top_idx:
        factores.append(
            {
                "factor": feature_names[idx],
                "importancia": importancias[idx],
                "contribucion": contribuciones[idx],
            }
        )

    return {
        "nivel": clase,
        "confianza": proba[pred],
        "probabilidades": dict(zip(label_encoders["Risk_Level"].classes_, proba)),
        "factores": factores,
    }
