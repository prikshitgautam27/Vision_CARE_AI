import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras
import time
import os

# --------------- Page config ---------------
st.set_page_config(page_title="VISIONCARE AI", page_icon="👁️", layout="wide")
PRIMARY = "#2E86C1"; ACCENT = "#117A65"

st.markdown(f"""
    <style>
    .reportbox {{ background-color:#f7fbff; border: 1px solid {PRIMARY}22; padding: 20px; border-radius: 14px; }}
    .hero-title {{ font-size: 50px; color: {PRIMARY}; text-align: center; font-weight: 700; margin-bottom: 0; }}
    .hero-tagline {{ font-size: 22px; color: {ACCENT}; text-align: center; margin-top: 6px; margin-bottom: 10px; }}
    @keyframes fadeIn {{ from {{opacity: 0; transform: translateY(10px);}} to {{opacity: 1; transform: translateY(0);}} }}
    .fade-in {{ animation: fadeIn 1.8s ease-in-out; }}
    .pill {{ display:inline-block; padding:8px 14px; border-radius:999px; font-weight:600; color:white; }}
    .pill-green {{ background:#2ecc71; }} .pill-red {{ background:#e74c3c; }}
    .quote-box {{ text-align:center; font-style:italic; color:#4c4c4c; margin-top:10px; }}
    .section-title {{ color:{PRIMARY}; margin-top:8px; margin-bottom:12px; }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="fade-in">
        <div class="hero-title">VISIONCARE AI</div>
        <div class="hero-tagline">Early Detection, Clearer Tomorrow</div>
    </div>
    <div class="quote-box">"Precision diagnostics for clinics. Confidence for patients."</div>
""", unsafe_allow_html=True)
time.sleep(1.0)
st.write("---")

# --------------- Safe model loading ---------------
@st.cache_resource
def safe_load_model(path_candidates, custom_objects=None):
    last_exception = None
    for path in path_candidates:
        if not os.path.exists(path):
            continue
        try:
            # Try modern format first, then H5 with compile=False
            if path.endswith(".keras"):
                return keras.models.load_model(path)
            else:
                return keras.models.load_model(path, compile=False, custom_objects=custom_objects or {})
        except Exception as e:
            last_exception = e
    if last_exception:
        raise last_exception
    raise FileNotFoundError(f"No model file found from candidates: {path_candidates}")

try:
    # Prefer .keras, fallback to .h5
    pred_model_1 = safe_load_model(["base_model_1.keras", "base_model_1.h5"])
    pred_model_2 = safe_load_model(["base_model_2.keras", "base_model_2.h5"])
    pred_meta_model = safe_load_model(["meta_model.keras"])  # meta already in new format
except Exception as e:
    st.error("Models failed to load. Check file paths, format (.keras preferred), and any custom layers you used.")
    st.exception(e)
    st.stop()

# --------------- Preprocess helper ---------------
def preprocess_image(uploaded_file, image_size=224):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("Failed to read image. Please upload a valid JPG/PNG.")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cx, cy = gray.shape[1] // 2, gray.shape[0] // 2
    radius = int(min(gray.shape) // 2 * 0.8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    vignette = cv2.merge([gray * (mask / 255.0)] * 3)
    resized = cv2.resize(vignette, (image_size, image_size))
    normalized = resized / 255.0
    return normalized, original

# --------------- Persistence (CSV dashboard) ---------------
REPORTS_CSV = "reports_log.csv"
def init_reports_store():
    if not os.path.exists(REPORTS_CSV):
        pd.DataFrame(columns=[
            "Timestamp","Patient Name","Age","Gender","Eye Side","Meta Probability","Final Class"
        ]).to_csv(REPORTS_CSV, index=False)

def append_report(row_dict):
    init_reports_store()
    df = pd.read_csv(REPORTS_CSV)
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(REPORTS_CSV, index=False)

def load_reports():
    init_reports_store()
    return pd.read_csv(REPORTS_CSV)

# --------------- Sidebar (patient + controls) ---------------
st.sidebar.header("🧑 Patient information")
patient_name = st.sidebar.text_input("Patient name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
eye_side = st.sidebar.radio("Eye side", ["Left Eye", "Right Eye"])
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.40, 0.05)
uploaded_file = st.file_uploader("📤 Upload fundus image", type=["jpg", "jpeg", "png"])

# --------------- Main: processing + report ---------------
if uploaded_file is not None:
    try:
        image, original_image = preprocess_image(uploaded_file)
        colA, colB = st.columns(2)
        with colA: st.image(original_image, caption="Original image", use_column_width=True)
        with colB: st.image(image, caption="Preprocessed image", use_column_width=True)

        image_batch = np.expand_dims(image, axis=0)
        p1 = pred_model_1.predict(image_batch, verbose=0)
        p2 = pred_model_2.predict(image_batch, verbose=0)
        stacked = np.hstack([p1, p2])
        p_meta = float(pred_meta_model.predict(stacked, verbose=0)[0, 0])

        final_class = "Cataract" if p_meta > threshold else "Normal"
        pill_class = "pill-red" if final_class == "Cataract" else "pill-green"

        st.subheader("📄 Diagnostic report")
        st.markdown(f"""
            <div class="reportbox">
                <h4 class="section-title">👤 Patient details</h4>
                <p><b>Name:</b> {patient_name or "—"}</p>
                <p><b>Age:</b> {age}</p>
                <p><b>Gender:</b> {gender}</p>
                <p><b>Eye side:</b> {eye_side}</p>
                <hr style="margin: 10px 0 16px 0;">
                <h4 class="section-title">🔬 Model result</h4>
                <p><b>Meta-model probability:</b> {p_meta:.4f}</p>
                <div class="pill {pill_class}">Final classification: {final_class}</div>
            </div>
        """, unsafe_allow_html=True)

        report_text = f"""VISIONCARE AI - Cataract Detection Report
-----------------------------------------
Patient Name: {patient_name}
Age: {age}
Gender: {gender}
Eye Side: {eye_side}
Meta-Model Probability: {p_meta:.4f}
Final Classification: {final_class}
"""
        st.download_button("📥 Download report (.txt)", report_text, file_name="diagnostic_report.txt")

        append_report({
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Patient Name": patient_name or "",
            "Age": age, "Gender": gender, "Eye Side": eye_side,
            "Meta Probability": round(p_meta, 4), "Final Class": final_class
        })
        st.success("This report has been added to the public dashboard below.")
    except Exception as e:
        st.error("Processing failed. Please verify the image and try again.")
        st.exception(e)

# --------------- Public dashboard ---------------
st.write("---")
st.subheader("📊 Reports dashboard (public view)")
df = load_reports()
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
with col1: f_name = st.text_input("Filter by patient name")
with col2: f_class = st.selectbox("Filter by class", ["All", "Cataract", "Normal"])
with col3: f_gender = st.selectbox("Filter by gender", ["All", "Male", "Female", "Other"])
with col4: f_eye = st.selectbox("Filter by eye", ["All", "Left Eye", "Right Eye"])

if f_name: df = df[df["Patient Name"].str.contains(f_name, case=False, na=False)]
if f_class != "All": df = df[df["Final Class"] == f_class]
if f_gender != "All": df = df[df["Gender"] == f_gender]
if f_eye != "All": df = df[df["Eye Side"] == f_eye]
st.dataframe(df, use_container_width=True)

# --------------- Business info ---------------
st.write("---")
st.subheader("🏥 Clinic registration & pricing")
st.markdown("""
- **How clinics register:** Reach us via email/phone to set up your account. We provide onboarding and user access.
- **Pricing:** ₹250 per scan
- **Membership:** Yearly plans with discounts for high-volume clinics
""")

st.write("---")
st.subheader("📞 Contact us")
st.markdown("""
- **Email:** colab@gmail.com  
- **Phone:** 9877035742
""")




# -----------------------------
# Developer info (styled + clickable LinkedIn)
# -----------------------------
st.write("---")
st.markdown("""
    <div style="text-align:center; margin-top:30px;">
        <h2 style="color:#2E86C1; font-size:36px; font-weight:bold;">
            👨‍💻 Developer Info
        </h2>
        <p style="font-size:22px; font-weight:bold; color:#117A65;">
            Developed by: ._P_.Gautam .K.Bansal & Team 
        </p>
        <p style="font-size:20px;">
            <a href="https://www.linkedin.com/in/prikshit-gautam-76b1a9308/" target="_blank" style="text-decoration:none; color:#0A66C2; font-weight:bold;">
                🔗 Connect on LinkedIn
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)

