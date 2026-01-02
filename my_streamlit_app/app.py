import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CreditGuard AI | Dark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DARK MODE STYLING ---
st.markdown("""
<style>
    /* Card Styling for Metrics - Dark Mode Friendly */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #1a73e8;
        box-shadow: 0 8px 12px rgba(0,0,0,0.5);
    }

    /* Force text colors to be visible */
    div[data-testid="metric-container"] label {
        color: #d0d0d0 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Button Styling */
    div.stButton > button {
        width: 100%;
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = '/home/olayiwola/Desktop/easyBuy/tuned_random_forest_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file missing at {model_path}")
        st.stop()
    return joblib.load(model_path)

try:
    model = load_model()
    all_features = model.feature_names_in_
except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

# --- 4. HELPER LOGIC ---
def get_categories(prefix):
    return sorted([f.replace(prefix, "") for f in all_features if f.startswith(prefix)])

# Extract Categories from Model
product_names = get_categories("Accounts Product Name_")
areas = get_categories("Accounts Area_")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("CreditGuard AI")
    st.caption("Dark Mode Enterprise Edition")
    
    st.divider()
    
    st.subheader("⚙️ Policy Control")
    threshold = st.slider("Approval Confidence (%)", 50, 95, 65) / 100.0
    
    st.subheader("👤 Customer Profile")
    
    with st.expander("Financial Details", expanded=True):
        total_paid = st.number_input("Total Paid (₦)", value=1000.0, step=250.0)
        balance = st.number_input("Balance Owed (₦)", value=500.0, step=100.0)
        percent_paid = st.slider("Percent Paid", 0.0, 1.0, 0.5)
        
    with st.expander("Account History"):
        days_last_pay = st.number_input("Days Since Last Pay", value=10)
        account_age = st.slider("Account Age", 0, 720, 90)
    
    # --- ADDED PRODUCT SELECTION HERE ---
    st.subheader("📍 Segmentation")
    selected_area = st.selectbox("Region", ["Unknown"] + areas)
    selected_product = st.selectbox("Product Type", ["Unknown"] + product_names)

# --- 6. PREDICTION LOGIC ---
input_data = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)

# Fill Numerical
input_data['Accounts Total Paid'] = total_paid
input_data['Accounts Balance to Collect'] = balance
input_data['Accounts Percent Paid Off'] = percent_paid
input_data['Accounts Days From Last Payment'] = days_last_pay
input_data['Account_Age_Days'] = account_age

# Fill Categorical (Area)
if f"Accounts Area_{selected_area}" in all_features:
    input_data[f"Accounts Area_{selected_area}"] = 1

# Fill Categorical (Product - NEW)
if f"Accounts Product Name_{selected_product}" in all_features:
    input_data[f"Accounts Product Name_{selected_product}"] = 1

# Predict
probs = model.predict_proba(input_data)[0]
score = probs[1]
status = "APPROVED" if score >= threshold else "DECLINED"

# --- 7. MAIN DASHBOARD ---

st.title(f"Risk Assessment Dashboard")
st.markdown(f"**Current Policy:** Approvals require a confidence score > **{threshold*100:.0f}%**")

# --- SECTION A: HUD ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Determination", status, delta="Final Decision", delta_color="normal" if status=="APPROVED" else "inverse")
    
with col2:
    st.metric("AI Confidence Score", f"{score*100:.1f}%", delta=f"{((score-threshold)*100):.1f}% vs Threshold")

with col3:
    st.metric("Outstanding Balance", f"₦{balance:,.2f}")

with col4:
    if status == "APPROVED":
        action_text = "✅ Grant Credit"
    else:
        action_text = "🛑 Request Deposit"
    st.metric("Recommended Action", action_text)

st.divider()

# --- SECTION B: ANALYSIS & SIMULATION ---

tab1, tab2 = st.tabs(["📊 Risk Factors", "🧪 Scenario Simulator"])

with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Key Drivers")
        importances = model.feature_importances_
        indices = np.argsort(importances)[-6:] 
        
        chart_data = pd.DataFrame({
            'Factor': [all_features[i] for i in indices],
            'Importance': importances[indices]
        })
        
        bar_chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
            x=alt.X('Importance', title='Influence Weight'),
            y=alt.Y('Factor', sort='-x', title=None),
            color=alt.value('#4FC3F7'), 
            tooltip=['Factor', alt.Tooltip('Importance', format='.3f')]
        ).properties(height=300).configure_axis(
            labelColor='white',
            titleColor='white'
        ).configure_view(strokeWidth=0)
        
        st.altair_chart(bar_chart, use_container_width=True)
        
    with c2:
        st.subheader("Risk Distribution")
        donut_df = pd.DataFrame({
            'Category': ['Risk (Default)', 'Safe (Pay)'],
            'Value': [1-score, score]
        })
        
        base = alt.Chart(donut_df).encode(theta=alt.Theta("Value", stack=True))
        pie = base.mark_arc(innerRadius=70, outerRadius=110).encode(
            color=alt.Color("Category", scale=alt.Scale(range=['#e57373', '#81c784'])),
            tooltip=["Category", "Value"]
        )
        text = base.mark_text(radius=0).encode(
            text=alt.value(f"{score*100:.0f}%"),
            size=alt.value(24),
            color=alt.value("white")
        )
        st.altair_chart(pie + text, use_container_width=True)

with tab2:
    st.subheader("🧪 'What-If' Simulation")
    st.info("Adjust the slider to see if a payment would change the decision.")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        sim_payment = st.slider("Simulate Extra Payment (₦)", 0, 50000, 0, step=500)
        
        sim_input = input_data.copy()
        sim_input['Accounts Total Paid'] += sim_payment
        sim_input['Accounts Balance to Collect'] = max(0, balance - sim_payment)
        
        sim_probs = model.predict_proba(sim_input)[0]
        sim_score = sim_probs[1]
        sim_status = "APPROVED" if sim_score >= threshold else "DECLINED"
        
    with col_sim2:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Current Score", f"{score*100:.1f}%")
        with m2:
            delta = sim_score - score
            st.metric("Simulated Score", f"{sim_score*100:.1f}%", delta=f"{delta*100:.1f}% boost")
            
        if sim_status == "APPROVED" and status == "DECLINED":
            st.success(f"Paying **₦{sim_payment:,.2f}** changes status to APPROVED.")
        elif sim_status == "DECLINED":
            st.warning("Still Declined. Try a higher amount.")

# --- 8. BATCH PROCESSING ---
st.divider()
with st.expander("📂 Batch Portfolio Upload"):
    st.write("Upload CSV for bulk analysis.")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())