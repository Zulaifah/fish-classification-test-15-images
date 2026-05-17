# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER (TEST VERSION)
# 15 images per species | Measurements + Image Classification
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mugilidae Fish Classifier",
    page_icon="🐟",
    layout="wide"
)

# Title
st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Identify 5 Mullet Species")
st.markdown("---")

# ===============================
# GOOGLE DRIVE FILE ID (GANTIKAN DENGAN ID ANDA)
# ===============================
# Gunakan tanda petik! " ... "
CNN_FILE_ID = "1UC9Ep19pOWTYgM2i7roWb8bcugpmcWLl"

# ===============================
# LOAD MODELS
# ===============================

@st.cache_resource
def load_models():
    """Load all models (CNN from Google Drive, measurements from GitHub)"""
    models = {}
    
    # Load CNN model from Google Drive
    st.info("📸 Loading image classification model...")
    try:
        url = f"https://drive.google.com/uc?export=download&id={CNN_FILE_ID}"
        output = "mugilidae_cnn_test.h5"
        
        if not os.path.exists(output):
            with st.spinner("Downloading CNN model from Google Drive (this may take 1-2 minutes)..."):
                gdown.download(url, output, quiet=False)
        
        if os.path.exists(output):
            models['cnn'] = load_model(output, compile=False)
            models['cnn_species'] = [
                "Planiliza subviridis",
                "Moolgarda seheli", 
                "Osteomugil perusii",
                "Moolgarda tade",
                "Ellochelon vaigiensis"
            ]
            st.success("✅ CNN model loaded successfully!")
        else:
            models['cnn'] = None
            st.error("❌ CNN model file not found")
            
    except Exception as e:
        st.error(f"CNN model error: {str(e)[:200]}")
        models['cnn'] = None
    
    # Load measurement models from GitHub
    st.info("📏 Loading measurement models...")
    try:
        models['ann'] = joblib.load('ann_pso_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        st.success("✅ Measurement models loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please upload these files to GitHub:\n"
                "  - ann_pso_model.pkl\n"
                "  - scaler.pkl\n"
                "  - label_encoder.pkl")
        models['ann'] = None
    except Exception as e:
        st.error(f"Measurement model error: {e}")
        models['ann'] = None
    
    return models

# Load all models
with st.spinner("Loading models... Please wait..."):
    models = load_models()

# ===============================
# SIDEBAR INFORMATION
# ===============================

st.sidebar.header("📋 About")
st.sidebar.info("""
**🐟 5 Mugilidae Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis

**📊 TEST VERSION:**
- 15 images per species
- Expected accuracy: 40-60%
- Final version: 500 images → 85%+

**🔬 Methods:**
- 📸 CNN (Image Classification)
- 📏 ANN-PSO (Measurements)
""")

st.sidebar.markdown("---")
st.sidebar.caption("FYP Project | Universiti Malaysia Terengganu")

# ===============================
# MAIN TABS
# ===============================

tab1, tab2 = st.tabs(["📸 Image Classification", "📏 Measurement Classification"])

# ===============================
# TAB 1: IMAGE CLASSIFICATION
# ===============================

with tab1:
    st.header("Identify Fish from Photo")
    st.caption("⚠️ TEST VERSION: Using limited images (15 per species). Accuracy will improve with more images.")
    
    if models['cnn'] is not None:
        uploaded_file = st.file_uploader(
            "Upload a fish image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear side-profile photo of the fish"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption='Uploaded Fish', width=250)
            
            with col2:
                if st.button("🔍 Identify Species", key="img_predict", type="primary"):
                    with st.spinner("Analyzing image..."):
                        img = image.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        predictions = models['cnn'].predict(img_array, verbose=0)[0]
                        top_idx = np.argmax(predictions)
                        confidence = predictions[top_idx] * 100
                        
                        st.success(f"### 🎯 {models['cnn_species'][top_idx]}")
                        st.progress(int(confidence))
                        st.caption(f"Confidence: {confidence:.1f}%")
                        
                        st.markdown("---")
                        st.subheader("📊 Top 3 Predictions")
                        top_3_idx = np.argsort(predictions)[::-1][:3]
                        for idx in top_3_idx:
                            st.write(f"- **{models['cnn_species'][idx]}**: {predictions[idx]*100:.1f}%")
                        
                        st.info("💡 **Note:** This is a test version with 15 images per species.")
    else:
        st.error("❌ CNN model not available.")
        st.info("Please check your Google Drive file permission.")

# ===============================
# TAB 2: MEASUREMENT CLASSIFICATION
# ===============================

with tab2:
    st.header("Identify Fish from Morphometric Measurements")
    st.caption("Enter the 15 measurements to identify the fish species")
    
    if models['ann'] is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Meristic Features (Counts)")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
            nd2 = st.number_input("ND2_Total", value=6.0, step=1.0)
            np_val = st.number_input("NP", value=14.0, step=1.0)
            nc = st.number_input("NC", value=14.0, step=1.0)
            nv = st.number_input("NV_Total", value=6.0, step=1.0)
            na = st.number_input("NA_Total", value=10.0, step=1.0)
        
        with col2:
            st.subheader("📏 Morphometric Features (mm)")
            sl = st.number_input("SL", value=150.0, step=10.0)
            pl = st.number_input("PL", value=35.0, step=5.0)
            bh = st.number_input("BH", value=40.0, step=5.0)
            hl = st.number_input("HL", value=35.0, step=5.0)
            
            st.subheader("📐 Truss Features (Sum in mm)")
            head = st.number_input("Head_Truss", value=80.0, step=10.0)
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
            tail = st.number_input("Tail_Truss", value=200.0, step=20.0)
        
        if st.button("🔍 Identify Species", key="meas_predict", type="primary"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                                  head, ant, mid, post, tail]])
            scaled = models['scaler'].transform(features)
            prediction = models['ann'].predict(scaled)[0]
            species = models['label_encoder'].inverse_transform([prediction])[0]
            
            st.success(f"### 🎯 {species}")
    else:
        st.error("❌ Measurement models not available.")

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🐟 Hybrid ANN-PSO + CNN Classification System</p>
    <p>TEST VERSION (15 images/species) | FYP Project</p>
    </div>
    """,
    unsafe_allow_html=True
)
