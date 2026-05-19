# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# MEASUREMENT ONLY (ANN-PSO) - NO CNN
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Identify 5 Mullet Species from Morphometric Measurements")
st.markdown("---")

# Warning about test version
st.info("⚠️ **TEST VERSION**: Using 15 images per species. Image classification coming soon!")

@st.cache_resource
def load_models():
    """Load measurement models only"""
    models = {}
    
    try:
        models['ann'] = joblib.load('ann_pso_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        st.success("✅ ANN-PSO Model loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please upload these files to GitHub:\n"
                "  - ann_pso_model.pkl\n"
                "  - scaler.pkl\n"
                "  - label_encoder.pkl")
        models['ann'] = None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        models['ann'] = None
    
    return models

# Load models
models = load_models()

# Sidebar
st.sidebar.header("📋 About")
st.sidebar.info("""
**🐟 5 Mugilidae Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis

**🔬 Method:** ANN-PSO (Particle Swarm Optimization)

**📏 Input:** 15 morphometric measurements

**📸 Image Classification:** Coming soon (model compatibility fix in progress)
""")

st.sidebar.markdown("---")
st.sidebar.caption("FYP Project | Universiti Malaysia Terengganu")

# Main content
if models['ann'] is not None:
    # Create two columns for inputs
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
    if st.button("🔍 Identify Species", type="primary"):
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
        st.markdown("---")
        col_result1, col_result2 = st.columns([1, 2])
        
        with col_result1:
            st.success(f"### 🎯 {species}")
        
        with col_result2:
            confidence = max(probabilities) * 100
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.1f}%")
        
        # Show all probabilities
        st.subheader("📊 Species Probabilities")
        
        prob_df = pd.DataFrame({
            'Species': models['label_encoder'].classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Species'))
        
        # Reference values
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

else:
    st.error("❌ Models not loaded. Please check your GitHub repository files.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🐟 ANN-PSO Optimization | Morphometric Classification</p>
    <p>📸 Image Classification Coming Soon (model compatibility fix)</p>
    <p>FYP Project | Universiti Malaysia Terengganu</p>
    </div>
    """,
    unsafe_allow_html=True
)
