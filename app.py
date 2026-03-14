import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_flag_invoice

# --------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligent Portal",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------------------
# Header Section
# --------------------------------------------------------------

st.markdown("""
# Vendor Invoice Intelligent Portal
### AI-Driven Freight Cost Prediction and Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast freight cost accurately**
- **Detect risky and abnormal vendor invoices**
""")

st.divider()

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------

st.sidebar.title("🔍 Model Selection")
selected_model = st.sidebar.radio(
    "Choose Predicting Model",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**
- 📈 Improve Cost Forecasting
- 📝 Reduce Invoice Fraud and Anomalies
- ⚙️ Faster Finance Operations
""")

# --------------------------------------------------------------
# Freight Cost Prediction
# --------------------------------------------------------------

if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")
    
    
    st.subheader("""
    **Objectives:**
    Predit freight cost for a vendor invoice using **Quantity** and **Dollars**
    to support budgetting, forecasting, and vendor negotiations.
    """)
    
    with st.form("freight_cost"):
        col1, col2 = st.columns(2)
        
        with col1:
            quantity = st.number_input(
                "📦Quantity",
                min_value=1,
                value=1200
            )
        with col2:
            dollars = st.number_input(
                "💰Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )
        submit_freight= st.form_submit_button("🔮Predict Freight Cost")
    
    if submit_freight:
        input_data = {
            "Dollars":[dollars],
            "Quantity":[quantity]
        }
        
        prediction = predict_freight_cost(input_data)['Predicted_Freight']
        
        st.success("Prediction Completed Succesfully")
        
        st.metric(
            label = "📊 Estimated Freight Cost",
            value=f"${prediction[0]:,.2f}"
        )

# --------------------------------------------------------------
# Invoice Flag Prediction
# --------------------------------------------------------------
else:
    st.subheader("🚨Invoice Manual Approval Prediction")
    
    st.markdown("""
    **OBjective:**
    Predict whether a vendor invoice should be **flagged for manual approval**
    based on abnormal cost, freight or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )
            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )
            
        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1,
                value=162
            )
            total_item_quantity = st.number_input(
                "total Item Quantity",
                min_value=1,
                value=162
            )
        
        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )
        submit_flag = st.form_submit_button("🧠 Evaluate Invoice Risk")
    
    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }
        
        flag_predicted = predict_flag_invoice(input_data)['Predict_Flag']
        
        is_flagged = bool(flag_predicted[0])
        
        if is_flagged:
            st.error("🚨 Invoice requires **MANUAL APPROVAL**")
        else:
            st.success("✅Invoice is **SAFE for Auto-Approval**")