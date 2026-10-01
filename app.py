import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Risk Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# LOAD MODEL, SCALER AND EXPECTED COLUMNS
# ---------------------------------------------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

    /* ---------- MAIN PAGE ---------- */

    .stApp {
        background: #eef5f8;
    }

    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- TOP HEADER ---------- */

    .hero {
        background: linear-gradient(135deg, #0f4c5c, #087f8c);
        padding: 42px 45px 38px 45px;
        border-radius: 20px 20px 0px 0px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(15, 76, 92, 0.20);
    }

    .hero-icon {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .hero p {
        margin-top: 10px;
        font-size: 15px;
        opacity: 0.88;
    }

    /* ---------- FORM CARD ---------- */

    .form-card {
        background: white;
        padding: 32px 40px 30px 40px;
        border-radius: 0px 0px 20px 20px;
        box-shadow: 0 12px 35px rgba(30, 60, 70, 0.12);
        margin-bottom: 25px;
    }

    /* ---------- STEPS ---------- */

    .steps {
        text-align: center;
        color: #75858c;
        font-size: 13px;
        margin-bottom: 25px;
        letter-spacing: 0.2px;
    }

    .steps .active {
        color: #087f8c;
        font-weight: 700;
    }

    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 20px;
        font-weight: 750;
        color: #183b45;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #7b8a90;
        margin-bottom: 18px;
    }

    /* ---------- STREAMLIT LABELS ---------- */

    label {
        color: #304850 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* ---------- INPUT BOXES ---------- */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px !important;
        border: 1px solid #dce6e9 !important;
        background-color: #fbfdfe !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #087f8c !important;
    }

    /* ---------- SLIDER ---------- */

    div[data-testid="stSlider"] {
        padding-top: 3px;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #087f8c, #0f9da8);
        color: white;
        font-size: 16px;
        font-weight: 750;
        box-shadow: 0 5px 14px rgba(8, 127, 140, 0.25);
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #066b76, #087f8c);
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(8, 127, 140, 0.30);
    }

    /* ---------- RESULT BOX ---------- */

    .result-title {
        text-align: center;
        color: #183b45;
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .result-note {
        text-align: center;
        color: #7b8a90;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #8a999e;
        font-size: 12px;
        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# (HTML kept flush-left with no blank lines, so Streamlit
#  does not turn it into a code block)
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
<div class="hero-icon">❤️</div>
<h1>Heart Risk Prediction</h1>
<p>Enter your health information to receive a machine-learning based heart disease risk prediction.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FORM CARD START
# ---------------------------------------------------------
st.markdown('<div class="form-card">', unsafe_allow_html=True)

st.markdown("""
<div class="steps">
<span class="active">Patient Details</span> &nbsp;&nbsp;›&nbsp;&nbsp; <span>Health Metrics</span> &nbsp;&nbsp;›&nbsp;&nbsp; <span>Prediction</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------------
st.markdown("""
<div class="section-title">Patient Information</div>
<div class="section-subtitle">Tell us a few basic details about the patient.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        18,
        100,
        40
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )


# ---------------------------------------------------------
# HEART HEALTH INFORMATION
# ---------------------------------------------------------
st.markdown("""
<div class="section-title">Heart & Health Information</div>
<div class="section-subtitle">Enter the available clinical measurements.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0,
        step=0.1
    )


# ---------------------------------------------------------
# ST SLOPE
# ---------------------------------------------------------
st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# ---------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

left_space, center_col, right_space = st.columns([1, 2, 1])

with center_col:
    predict_clicked = st.button(
        "🔍  Predict Heart Disease Risk",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if predict_clicked:

    # Create raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create dataframe
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Remove any unexpected columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown("""
<div class="result-title">Prediction Result</div>
<div class="result-note">The result below is generated by the trained machine-learning model.</div>
""", unsafe_allow_html=True)

    if prediction == 1:

        st.error(
            "⚠️ High Risk of Heart Disease"
        )

        st.warning(
            "The model classified this input as higher risk. "
            "This is a machine-learning prediction and is not a medical diagnosis."
        )

    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )

        st.info(
            "The model classified this input as lower risk. "
            "This is a machine-learning prediction and is not a medical diagnosis."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">❤️ Heart Risk Prediction System &nbsp;•&nbsp; Machine Learning Model</div>
""", unsafe_allow_html=True)