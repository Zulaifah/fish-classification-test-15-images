
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

st.set_page_config(page_title="Mugilidae Fish Classifier (Test)", page_icon="🐟", layout="wide")

st.title("🐟 Mugilidae Fish Classification System - TEST VERSION")
st.markdown("### Identify 5 Mullet Species")
st.markdown("---")

# Warning about test version
st.warning("⚠️ TEST VERSION: Using limited images (15 per species). Accuracy will be lower. Final version will use 100 images per species for 85%+ accuracy.")

@st.cache_resource
def load_models():
    models = {}

    # Load CNN model (test version)
    if os.path.exists('mugilidae_cnn_test.h5'):
        models['cnn'] = load_model('mugilidae_cnn_test.h5', compile=False)
        with open('cnn_class_names.txt', 'r') as f:
            models['cnn_species'] = [line.strip() for line in f.readlines()]
        st.success("✅ CNN Test Model loaded")
    else:
        models['cnn'] = None
        st.info("📸 CNN model not found")

    # Load measurement models
    try:
        models['ann'] = joblib.load('ann_pso_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        st.success("✅ Measurement model loaded")
    except:
        models['ann'] = None

    return models

models = load_models()

st.sidebar.header("📋 About")
st.sidebar.info("""
**TEST VERSION**
- Limited images (15 per species)
- Expected accuracy: 40-60%
- Final version: 100 images/species → 85-95%

**5 Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis
""")

tab1, tab2 = st.tabs(["📸 Image Classification (Test)", "📏 Measurement Classification"])

with tab1:
    st.header("Identify from Photo (TEST VERSION)")
    st.caption("⚠️ Accuracy lower due to limited test images (15 per species)")

    if models['cnn'] is not None:
        uploaded = st.file_uploader("Upload fish image", type=['jpg','jpeg','png'])

        if uploaded:
            image = Image.open(uploaded)
            col1, col2 = st.columns([1,2])
            with col1:
                st.image(image, width=200)

            if st.button("🔍 Identify Species", key="img_predict"):
                with st.spinner("Analyzing image..."):
                    img = image.resize((224,224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    pred = models['cnn'].predict(img_array, verbose=0)[0]
                    top_idx = np.argmax(pred)

                    with col2:
                        st.success(f"### {models['cnn_species'][top_idx].replace('_', ' ')}")
                        st.progress(int(pred[top_idx]*100))
                        st.caption(f"Confidence: {pred[top_idx]*100:.1f}%")

                    st.info(f"💡 Test version accuracy: ~{pred[top_idx]*100:.0f}%. Final version with 500 images will be much more accurate!")
    else:
        st.info("📸 CNN model not available. Please train with images first.")

with tab2:
    st.header("Identify from Measurements")

    if models['ann'] is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Meristic Features")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
            nd2 = st.number_input("ND2_Total", value=6.0, step=1.0)
            np_val = st.number_input("NP", value=14.0, step=1.0)
            nc = st.number_input("NC", value=14.0, step=1.0)
            nv = st.number_input("NV_Total", value=6.0, step=1.0)
            na = st.number_input("NA_Total", value=10.0, step=1.0)

        with col2:
            st.subheader("Morphometric Features (mm)")
            sl = st.number_input("SL", value=150.0, step=10.0)
            pl = st.number_input("PL", value=35.0, step=5.0)
            bh = st.number_input("BH", value=40.0, step=5.0)
            hl = st.number_input("HL", value=35.0, step=5.0)

            st.subheader("Truss Features (mm)")
            head = st.number_input("Head_Truss", value=80.0, step=10.0)
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
            tail = st.number_input("Tail_Truss", value=200.0, step=20.0)

        if st.button("🔍 Identify Species", key="meas_predict", type="primary"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]])
            scaled = models['scaler'].transform(features)
            pred = models['ann'].predict(scaled)[0]
            species = models['label_encoder'].inverse_transform([pred])[0]
            st.success(f"### 🎯 {species}")
    else:
        st.error("Measurement model not found")

st.markdown("---")
st.caption("TEST VERSION | Final version with 500 images coming soon | FYP Project")
