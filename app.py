
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# LOAD MODEL, SCALER AND METADATA
# =========================================================

with open("diabetes_model.pkl", "rb") as file:
    classifier = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("model_metadata.pkl", "rb") as file:
    metadata = pickle.load(file)

dataset_averages = metadata["dataset_averages"]
cm = metadata["confusion_matrix"]
metrics = metadata["metrics"]

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #777;
    font-size: 18px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Diabetes Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Diabetes Risk Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the patient parameters from the sidebar and click "
    "**Analyze Risk** to generate the prediction."
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("👤 Patient Information")

pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.sidebar.number_input(
    "Glucose Level",
    min_value=0.0,
    max_value=300.0,
    value=100.0
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.sidebar.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.30
)

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

predict_button = st.sidebar.button(
    "🔍 Analyze Risk",
    use_container_width=True
)

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Standardize
    std_data = scaler.transform(input_data)

    # Prediction
    prediction = classifier.predict(std_data)

    # Probability
    probability = classifier.predict_proba(std_data)

    risk_percentage = probability[0][1] * 100

    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader("🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        if prediction[0] == 0:
            st.success("🟢 NOT DIABETIC")
        else:
            st.error("🔴 DIABETIC")

    with col2:

        st.metric(
            "Estimated Risk",
            f"{risk_percentage:.2f}%"
        )

    with col3:

        confidence = max(probability[0]) * 100

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

    # =====================================================
    # RISK LEVEL
    # =====================================================

    st.subheader("📊 Risk Level")

    st.progress(
        min(int(risk_percentage), 100)
    )

    if risk_percentage < 30:

        st.success(
            f"Low model-estimated risk: {risk_percentage:.2f}%"
        )

    elif risk_percentage < 60:

        st.warning(
            f"Moderate model-estimated risk: {risk_percentage:.2f}%"
        )

    else:

        st.error(
            f"Higher model-estimated risk: {risk_percentage:.2f}%"
        )

    # =====================================================
    # PATIENT PARAMETERS
    # =====================================================

    st.divider()

    st.subheader("📋 Patient Parameters")

    patient_data = pd.DataFrame({

        "Parameter": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],

        "Your Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ],

        "Dataset Average": [
            dataset_averages["Pregnancies"],
            dataset_averages["Glucose"],
            dataset_averages["BloodPressure"],
            dataset_averages["SkinThickness"],
            dataset_averages["Insulin"],
            dataset_averages["BMI"],
            dataset_averages["DiabetesPedigreeFunction"],
            dataset_averages["Age"]
        ]
    })

    st.dataframe(
        patient_data,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # BAR CHART
    # =====================================================

    st.subheader("📈 Patient vs Dataset Average")

    chart_data = patient_data.set_index("Parameter")

    st.bar_chart(
        chart_data[
            ["Your Value", "Dataset Average"]
        ]
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.divider()

    st.subheader("🔲 Confusion Matrix")

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.imshow(cm)

    ax.set_title(
        "Confusion Matrix",
        fontsize=12
    )

    ax.set_xlabel(
        "Predicted",
        fontsize=9
    )

    ax.set_ylabel(
        "Actual",
        fontsize=9
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Not Diabetic", "Diabetic"],
        fontsize=8
    )

    ax.set_yticklabels(
        ["Not Diabetic", "Diabetic"],
        fontsize=8
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=12
            )

    st.pyplot(
        fig,
        width="content"
    )

    plt.close(fig)

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{metrics['f1'] * 100:.2f}%"
        )

# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.info(
    "⚠️ This application provides a machine-learning based "
    "prediction for educational purposes only. The estimated "
    "risk is generated by the trained model and is not a "
    "medical diagnosis. Please consult a qualified healthcare "
    "professional for medical advice."
)
