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
# Cara dapatkan File ID:
# 1. Buka file di Google Drive
# 2. Right-click → Get link
# 3. Copy bahagian antara /d/ dan /view
# Contoh: https://drive.google.com/file/d/1ABC123xyz789/view
# File ID: 1ABC123xyz789

CNN_FILE_ID = "1UC9Ep19pOWTYgM2i7roWb8bcugpmcWLl"  # ← GANTIKAN INI!

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
        # Try different URL formats
        url1 = f"https://drive.google.com/uc?export=download&id={1UC9Ep19pOWTYgM2i7roWb8bcugpmcWLl}"
        url2 = f"https://drive.google.com/uc?id={1UC9Ep19pOWTYgM2i7roWb8bcugpmcWLl}&export=download"
        url3 = f"https://drive.google.com/file/d/{1UC9Ep19pOWTYgM2i7roWb8bcugpmcWLl}/view?usp=sharing"
        
        output = "mugilidae_cnn_test.h5"
        
        if not os.path.exists(output):
            with st.spinner("Downloading CNN model from Google Drive (this may take 1-2 minutes)..."):
                try:
                    # Try first URL
                    gdown.download(url1, output, quiet=False, fuzzy=False)
                except:
                    try:
                        # Try second URL
                        gdown.download(url2, output, quiet=False, fuzzy=False)
                    except:
                        st.warning("Auto-download failed. Please download manually.")
                        st.markdown(f"[Click here to download CNN model]({url3})")
                        
                        # Manual upload option
                        uploaded = st.file_uploader("Or upload the .h5 file manually:", type=['h5'])
                        if uploaded:
                            with open(output, 'wb') as f:
                                f.write(uploaded.getbuffer())
                            st.success("✅ Model uploaded manually!")
        
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
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a fish image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear side-profile photo of the fish"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption='Uploaded Fish', width=250)
            
            with col2:
                if st.button("🔍 Identify Species", key="img_predict", type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Preprocess image
                        img = image.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Predict
                        predictions = models['cnn'].predict(img_array, verbose=0)[0]
                        top_idx = np.argmax(predictions)
                        confidence = predictions[top_idx] * 100
                        
                        # Display result
                        st.success(f"### 🎯 {models['cnn_species'][top_idx]}")
                        st.progress(int(confidence))
                        st.caption(f"Confidence: {confidence:.1f}%")
                        
                        # Show top 3 predictions
                        st.markdown("---")
                        st.subheader("📊 Top 3 Predictions")
                        top_3_idx = np.argsort(predictions)[::-1][:3]
                        for idx in top_3_idx:
                            st.write(f"- **{models['cnn_species'][idx]}**: {predictions[idx]*100:.1f}%")
                        
                        # Note about test version
                        st.info("💡 **Note:** This is a test version with 15 images per species. "
                                "The final version with 500 images will achieve 85-95% accuracy.")
    else:
        st.error("❌ CNN model not available.")
        st.info("""
        **How to fix:**
        1. Upload your `mugilidae_cnn_test.h5` file to Google Drive
        2. Set permission to "Anyone with the link"
        3. Copy the File ID
        4. Update `CNN_FILE_ID` in the code
        5. Redeploy the app
        """)

# ===============================
# TAB 2: MEASUREMENT CLASSIFICATION
# ===============================

with tab2:
    st.header("Identify Fish from Morphometric Measurements")
    st.caption("Enter the 15 measurements to identify the fish species")
    
    if models['ann'] is not None:
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Meristic Features (Counts)")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0, help="First dorsal fin spines + rays")
            nd2 = st.number_input("ND2_Total", value=6.0, step=1.0, help="Second dorsal fin spines + rays")
            np_val = st.number_input("NP", value=14.0, step=1.0, help="Pectoral fin rays")
            nc = st.number_input("NC", value=14.0, step=1.0, help="Caudal fin rays")
            nv = st.number_input("NV_Total", value=6.0, step=1.0, help="Ventral/pelvic fin spines + rays")
            na = st.number_input("NA_Total", value=10.0, step=1.0, help="Anal fin spines + rays")
        
        with col2:
            st.subheader("📏 Morphometric Features (mm)")
            sl = st.number_input("SL", value=150.0, step=10.0, help="Standard length")
            pl = st.number_input("PL", value=35.0, step=5.0, help="Pectoral fin length")
            bh = st.number_input("BH", value=40.0, step=5.0, help="Body height")
            hl = st.number_input("HL", value=35.0, step=5.0, help="Head length")
            
            st.subheader("📐 Truss Features (Sum in mm)")
            head = st.number_input("Head_Truss", value=80.0, step=10.0, help="A-B + A-C + A-D")
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0, help="B-C + B-D + C-D")
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0, help="C-E + C-F + D-E + D-F + E-F")
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0, help="E-G + E-H + F-G + F-H + G-H")
            tail = st.number_input("Tail_Truss", value=200.0, step=20.0, help="G-I + G-J + H-I + H-J + I-J")
        
        # Predict button
        if st.button("🔍 Identify Species", key="meas_predict", type="primary"):
            # Prepare features
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                                  head, ant, mid, post, tail]])
            
            # Scale features
            scaled = models['scaler'].transform(features)
            
            # Predict
            prediction = models['ann'].predict(scaled)[0]
            species = models['label_encoder'].inverse_transform([prediction])[0]
            
            # Get probabilities
            probabilities = models['ann'].predict_proba(scaled)[0]
            
            # Display result
            st.success(f"### 🎯 {species}")
            
            # Confidence meter
            confidence = max(probabilities) * 100
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.1f}%")
            
            # Show all probabilities
            st.markdown("---")
            st.subheader("📊 Species Probabilities")
            
            prob_df = pd.DataFrame({
                'Species': models['label_encoder'].classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            # Display as bar chart
            st.bar_chart(prob_df.set_index('Species'))
            
    else:
        st.error("❌ Measurement models not available.")
        st.info("""
        **How to fix:**
        Please upload these files to your GitHub repository:
        - `ann_pso_model.pkl`
        - `scaler.pkl`
        - `label_encoder.pkl`
        """)

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🐟 Hybrid ANN-PSO + CNN Classification System</p>
    <p>TEST VERSION (15 images/species) | Final Version Coming Soon</p>
    <p>FYP Project | Universiti Malaysia Terengganu</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# HELPER FUNCTIONS
# ===============================

# Display reference values in expander
with st.expander("📖 Reference Values for Each Species"):
    st.markdown("""
    | Species | ND1_Total | ND2_Total | NP | NC | Typical SL |
    |---------|-----------|-----------|-----|-----|-------------|
    | **Planiliza subviridis** | 4 | 6 | 12-15 | 12-15 | 250-350 mm |
    | **Moolgarda seheli** | 4 | 7-8 | 14-17 | 14-17 | 120-180 mm |
    | **Osteomugil perusii** | 4 | 6-7 | 12-15 | 12-15 | 120-180 mm |
    | **Moolgarda tade** | 4 | 8 | 15-17 | 13-19 | 250-350 mm |
    | **Ellochelon vaigiensis** | 4 | 5-8 | 10-16 | 11-16 | 120-180 mm |
    """)
