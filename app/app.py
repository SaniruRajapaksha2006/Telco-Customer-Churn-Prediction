"""
Streamlit web app for customer churn prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Telco Customer Churn Predictor")

st.markdown("""
This app predicts whether a telecom customer is likely to churn using machine learning.
""")

# Placeholder for now
st.info("App coming soon! After training the models, this app will allow you to:")
st.markdown("- Predict churn for individual customers")
st.markdown("- Upload CSV files for batch predictions")
st.markdown("- View model insights and feature importance")
