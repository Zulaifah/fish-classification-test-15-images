# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# Dengan Comparison: ANN vs ANN-RS vs ANN-CGS vs ANN-PSO
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Compare 4 Optimization Methods: ANN vs ANN-RS vs ANN-CGS vs ANN-PSO")
st.markdown("---")

# Load all 4 models
@st.cache_resource
def load_all_models():
    models = {}
    
    # Try to load all 4 models
    try:
        models['ann'] = joblib.load('ann_standalone.pkl')
        st.success("✅ Standalone ANN model loaded")
    except:
        models['ann'] = None
        st.warning("⚠️ Standalone ANN model not found")
    
    try:
        models['rs'] = joblib.load('ann_rs_model.pkl')
        st.success("✅ ANN-RS model loaded")
    except:
        models['rs'] = None
        st.warning("⚠️ ANN-RS model not found")
    
    try:
        models['cgs'] = joblib.load('ann_cgs_model.pkl')
        st.success("✅ ANN-CGS model loaded")
    except:
        models['cgs'] = None
        st.warning("⚠️ ANN-CGS model not found")
    
    try:
        models['pso'] = joblib.load('ann_pso_model.pkl')
        st.success("✅ ANN-PSO model loaded")
    except:
        models['pso'] = None
        st.warning("⚠️ ANN-PSO model not found")
    
    # Load scaler and label encoder
    try:
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        st.success("✅ Scaler and Label Encoder loaded")
    except:
        models['scaler'] = None
        models['label_encoder'] = None
        st.error("❌ Scaler or Label Encoder not found")
    
    return models

# Sidebar info
st.sidebar.header("📋 About")
st.sidebar.info("""
**🐟 5 Mugilidae Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis

**🔬 4 Optimization Methods:**
1. **ANN** - Default (10,5) architecture
2. **ANN-RS** - Random Search
3. **ANN-CGS** - Coarse Grid Search
4. **ANN-PSO** - Particle Swarm Optimization 🏆

**📏 Input:** 15 morphometric measurements
""")

st.sidebar.markdown("---")
st.sidebar.caption("FYP Project | Universiti Malaysia Terengganu")

# Load models
with st.spinner("Loading models..."):
    models = load_all_models()

# ===============================
# INPUT FORM
# ===============================

st.header("📏 Enter Morphometric Measurements")

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
    head = st.number_input("Head_Truss", value=80.0, step=10.0)
    ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
    mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
    post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
    tail = st.number_input("Tail_Truss", value=200.0, step=20.0)

# Prepare features array
features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                      head, ant, mid, post, tail]])

# ===============================
# PREDICT BUTTON
# ===============================

if st.button("🔍 Compare All Methods", type="primary"):
    
    if models['scaler'] is None:
        st.error("❌ Scaler not loaded. Please check your files.")
    else:
        # Scale features
        features_scaled = models['scaler'].transform(features)
        
        # Results storage
        results = []
        
        # 1. Standalone ANN
        if models['ann'] is not None:
            start_time = time.time()
            pred = models['ann'].predict(features_scaled)[0]
            inference_time = time.time() - start_time
            species = models['label_encoder'].inverse_transform([pred])[0]
            proba = models['ann'].predict_proba(features_scaled)[0]
            results.append({
                'Method': 'ANN (Standalone)',
                'Predicted Species': species,
                'Confidence': f"{max(proba)*100:.1f}%",
                'Inference Time': f"{inference_time*1000:.2f} ms",
                'Architecture': '10 → 5 neurons'
            })
        
        # 2. ANN-RS
        if models['rs'] is not None:
            start_time = time.time()
            pred = models['rs'].predict(features_scaled)[0]
            inference_time = time.time() - start_time
            species = models['label_encoder'].inverse_transform([pred])[0]
            proba = models['rs'].predict_proba(features_scaled)[0]
            results.append({
                'Method': 'ANN-RS (Random Search)',
                'Predicted Species': species,
                'Confidence': f"{max(proba)*100:.1f}%",
                'Inference Time': f"{inference_time*1000:.2f} ms",
                'Architecture': 'Optimized by RS'
            })
        
        # 3. ANN-CGS
        if models['cgs'] is not None:
            start_time = time.time()
            pred = models['cgs'].predict(features_scaled)[0]
            inference_time = time.time() - start_time
            species = models['label_encoder'].inverse_transform([pred])[0]
            proba = models['cgs'].predict_proba(features_scaled)[0]
            results.append({
                'Method': 'ANN-CGS (Grid Search)',
                'Predicted Species': species,
                'Confidence': f"{max(proba)*100:.1f}%",
                'Inference Time': f"{inference_time*1000:.2f} ms",
                'Architecture': 'Optimized by CGS'
            })
        
        # 4. ANN-PSO
        if models['pso'] is not None:
            start_time = time.time()
            pred = models['pso'].predict(features_scaled)[0]
            inference_time = time.time() - start_time
            species = models['label_encoder'].inverse_transform([pred])[0]
            proba = models['pso'].predict_proba(features_scaled)[0]
            results.append({
                'Method': 'ANN-PSO (Particle Swarm) 🏆',
                'Predicted Species': species,
                'Confidence': f"{max(proba)*100:.1f}%",
                'Inference Time': f"{inference_time*1000:.2f} ms",
                'Architecture': 'Optimized by PSO'
            })
        
        # Display results table
        st.markdown("---")
        st.subheader("📊 Comparison Results")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Find best method (highest confidence)
        best_idx = 0
        best_conf = 0
        for i, r in enumerate(results):
            conf_val = float(r['Confidence'].replace('%', ''))
            if conf_val > best_conf:
                best_conf = conf_val
                best_idx = i
        
        st.success(f"🏆 **BEST METHOD: {results[best_idx]['Method']}** with {results[best_idx]['Confidence']} confidence")
        
        # Show probability distribution for each method
        st.subheader("📊 Probability Distribution by Method")
        
        # Get probabilities for each model
        prob_data = []
        species_list = models['label_encoder'].classes_
        
        if models['ann'] is not None:
            proba = models['ann'].predict_proba(features_scaled)[0]
            for i, sp in enumerate(species_list):
                prob_data.append({'Method': 'ANN', 'Species': sp, 'Probability': proba[i]})
        
        if models['rs'] is not None:
            proba = models['rs'].predict_proba(features_scaled)[0]
            for i, sp in enumerate(species_list):
                prob_data.append({'Method': 'RS', 'Species': sp, 'Probability': proba[i]})
        
        if models['cgs'] is not None:
            proba = models['cgs'].predict_proba(features_scaled)[0]
            for i, sp in enumerate(species_list):
                prob_data.append({'Method': 'CGS', 'Species': sp, 'Probability': proba[i]})
        
        if models['pso'] is not None:
            proba = models['pso'].predict_proba(features_scaled)[0]
            for i, sp in enumerate(species_list):
                prob_data.append({'Method': 'PSO', 'Species': sp, 'Probability': proba[i]})
        
        prob_df = pd.DataFrame(prob_data)
        
        # Create bar chart
        import plotly.express as px
        fig = px.bar(prob_df, x='Method', y='Probability', color='Species', 
                     title='Prediction Probability by Method',
                     barmode='group',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("📖 About Each Optimization Method"):
            st.markdown("""
            | Method | Description | Strengths | Weaknesses |
            |--------|-------------|-----------|-------------|
            | **ANN (Standalone)** | Default architecture (10,5) | Fast, simple | May not be optimal |
            | **ANN-RS** | Random Search hyperparameter tuning | Explores wide range | No guarantee of optimal |
            | **ANN-CGS** | Coarse Grid Search | Systematic | Limited combinations |
            | **ANN-PSO** | Particle Swarm Optimization | Global optimization, adaptive | Slightly slower training |
            
            **🏆 PSO Advantages:**
            - Swarm intelligence shares information
            - Adaptive inertia for exploration/exploitation
            - Memory of best positions
            - Avoids local optima
            """)

else:
    st.info("👆 Click 'Compare All Methods' after entering measurements")

# Reference values expander
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

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🐟 ANN-PSO vs ANN-RS vs ANN-CGS vs Standalone ANN | Comparison Study</p>
    <p>FYP Project | Universiti Malaysia Terengganu</p>
    </div>
    """,
    unsafe_allow_html=True
)
