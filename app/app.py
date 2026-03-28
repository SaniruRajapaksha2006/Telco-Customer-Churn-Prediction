"""
Streamlit web app for customer churn prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.features.build_features import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📱",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }

    /* Headers */
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    h2, h3 {
        color: #262730;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }

    /* Prediction boxes */
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        animation: pulse 2s infinite;
    }

    .low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    /* Risk factor list */
    .risk-factor {
        background-color: #FFF3E0;
        border-left: 4px solid #FF4B4B;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    /* Info boxes */
    .info-box {
        background-color: #E8F4FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00f2fe;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F0F2F6;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    /* Metrics */
    .stMetric {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    /* Success/Error/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }

    /* Sidebar  */
    .css-1d391kg {
        background-color: #F0F2F6;
    }
</style>
""", unsafe_allow_html=True)


# Custom header with logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="font-size: 3rem;">📱 Telco Churn Predictor</h1>
        <p style="font-size: 1.2rem; color: #666;">AI-Powered Customer Retention Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# Load model and preprocessor
# Update the load_models function
@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    model_path = Path(__file__).parent.parent / "models"

    # Load Decision Tree
    dt_model = joblib.load(model_path / "decision_tree.pkl")

    # Load preprocessor
    engineer = joblib.load(model_path / "preprocessor.pkl")

    st.success(
        f"Models loaded! Preprocessor has {len(engineer.training_columns) if hasattr(engineer, 'training_columns') else '?'} features")

    return dt_model, engineer

def main():
    st.title("Telco Customer Churn Predictor")

    st.markdown("""
    ### Predict if a customer will leave the service provider
    This app uses a **Decision Tree model** trained on historical telecom data.
    
    **Model Performance:**
    - Accuracy: **75.5%**
    - Recall: **74.6%** (identifies 75% of actual churners)
    - ROC-AUC: **82.3%**
    """)

    # Load models
    try:
        model, engineer = load_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run `python train.py` first to train the models.")
        return

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Insights"])

    with tab1:
        st.header("Predict for a Single Customer")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with col2:
            st.subheader("Account Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                 "Credit card (automatic)"]
            )

        with col3:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        # Charges
        st.subheader("Charges")
        col4, col5 = st.columns(2)
        with col4:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0, max_value=200.0, value=50.0, step=5.0
            )
        with col5:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0, max_value=10000.0, value=500.0, step=50.0
            )

        # Create DataFrame
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': ['No' if phone_service == "No" else "Yes"],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': ['No'],  # Default
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        if st.button("Predict Churn Risk", type="primary"):
            try:
                # Preprocess input using the prediction method
                X_processed = engineer.predict_preprocess(input_data, fit=False)

                # Make prediction
                prediction = model.predict(X_processed)
                probability = model.predict_proba(X_processed)[0]

                # Display results with custom styling
                st.markdown("---")
                st.subheader("Prediction Results")

                col_result1, col_result2 = st.columns(2)

                with col_result1:
                    if prediction[0] == 1:
                        st.markdown(f"""
                        <div class="high-risk">
                            <h2 style="color: white; margin: 0;">HIGH RISK</h2>
                            <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">Customer is likely to churn</p>
                            <p style="color: white; font-size: 0.9rem;">Recommended Action: Immediate Retention Offer</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="low-risk">
                            <h2 style="color: white; margin: 0;">LOW RISK</h2>
                            <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">Customer is likely to stay</p>
                            <p style="color: white; font-size: 0.9rem;">Recommended Action: Standard Engagement</p>
                        </div>
                        """, unsafe_allow_html=True)

                with col_result2:
                    # Create a gauge-like display for probability
                    prob_color = "#f5576c" if probability[1] > 0.5 else "#4facfe"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background-color: #F0F2F6; border-radius: 10px;">
                        <p style="font-size: 1rem; margin: 0;">Churn Probability</p>
                        <div style="position: relative; margin: 1rem 0;">
                            <div style="width: 100%; height: 20px; background-color: #E0E0E0; border-radius: 10px; overflow: hidden;">
                                <div style="width: {probability[1] * 100}%; height: 100%; background: linear-gradient(90deg, {prob_color}, #ff6b6b); border-radius: 10px;"></div>
                            </div>
                            <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: {prob_color};">{probability[1]:.1%}</p>
                        </div>
                        <p style="font-size: 1rem; margin: 0;">Retention Probability: <strong>{probability[0]:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk factors
                st.subheader("Key Risk Factors")
                risk_factors = []

                if contract == "Month-to-month":
                    risk_factors.append("• **Month-to-month contract** - 42.7% churn rate (highest risk)")
                if internet_service == "Fiber optic":
                    risk_factors.append("• **Fiber optic internet** - 41.9% churn rate")
                if payment_method == "Electronic check":
                    risk_factors.append("• **Electronic check payment** - 45.3% churn rate")
                if tenure < 12:
                    risk_factors.append(f"• **New customer** ({tenure} months) - 40% of churn happens in first year")
                if online_security != "Yes" and internet_service != "No":
                    risk_factors.append("• **No online security** - increases churn risk")
                if tech_support != "Yes" and internet_service != "No":
                    risk_factors.append("• **No tech support** - 2.7x higher churn risk")
                if senior_citizen == "Yes":
                    risk_factors.append("• **Senior citizen** - 41.7% churn rate (1.8x higher)")
                if monthly_charges > 80:
                    risk_factors.append(f"• **High monthly charges** (${monthly_charges:.0f}) - premium customers churn more")

                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"""
                        <div class="risk-factor">
                            📌 {factor}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                        No major risk factors detected! This customer has good retention characteristics.
                    </div>
                    """, unsafe_allow_html=True)

                # Recommendation
                st.subheader("Recommendations")
                if prediction[0] == 1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white;">
                        <strong>Immediate Actions:</strong>
                        <ul>
                            <li>Offer contract upgrade incentive (2-year contract with 10% discount)</li>
                            <li>Bundle security features (free online security + tech support trial)</li>
                            <li>Schedule retention call within 24 hours</li>
                            <li>Send personalized offer via email</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white;">
                        <strong>Retention Strategies:</strong>
                        <ul>
                            <li>Send loyalty reward email</li>
                            <li>Offer service upgrade recommendations</li>
                            <li>Maintain current service quality</li>
                            <li>Monitor for any service issues</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Please check your input values and try again.")

    with tab2:
        st.header("Batch Prediction")
        st.markdown("""
        Upload a CSV file with multiple customers to get predictions in bulk.
        
        **Required columns:** gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
        TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, 
        MonthlyCharges, TotalCharges
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.write("**Preview of uploaded data:**")
                st.dataframe(batch_data.head())
                st.write(f"Total rows: {len(batch_data)}")

                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing predictions..."):
                        # Preprocess all data
                        X_processed, _ = engineer.prepare_data(batch_data, fit=False)

                        # Predict
                        predictions = model.predict(X_processed)
                        probabilities = model.predict_proba(X_processed)

                        # Add to dataframe
                        batch_data['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                        batch_data['Churn_Probability'] = probabilities[:, 1]
                        batch_data['Retention_Probability'] = probabilities[:, 0]

                        st.success("Predictions complete!")
                        st.dataframe(batch_data)

                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        churn_count = (batch_data['Churn_Prediction'] == 'Yes').sum()
                        churn_rate = churn_count / len(batch_data) * 100

                        with col1:
                            st.metric("Predicted Churners", churn_count)
                        with col2:
                            st.metric("Predicted Retained", len(batch_data) - churn_count)
                        with col3:
                            st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")

                        # Download button
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.write("Please check that your CSV has the required columns.")

    with tab3:
        st.header("Model Insights")

        st.subheader("Top Features Influencing Churn")
        st.markdown("""
        | Rank | Feature | Importance | Insight |
        |------|---------|------------|---------|
        | 1 | **Contract_Two year** | 30.9% | 2-year contracts have only 2.8% churn |
        | 2 | **Contract_One year** | 18.5% | Annual contracts show strong retention |
        | 3 | **Tenure** | 16.1% | 40% of churn happens in first 12 months |
        | 4 | **InternetService_Fiber optic** | 9.7% | Fiber optic: 41.9% churn rate |
        | 5 | **MonthlyCharges** | 7.3% | Higher charges = higher churn risk |
        """)

        st.subheader("High Risk Profiles")
        st.markdown("""
        - **Month-to-month contract + Fiber optic + Electronic check**: 78% churn rate
        - **New customer (<6 months) + High monthly charges (>$80)**: 67% churn rate
        - **Senior citizen + No tech support**: 54% churn rate
        """)

        st.subheader("Protective Factors")
        st.markdown("""
        - **Two-year contract**: 97% retention rate
        - **Credit card payment**: 85% retention rate
        - **Multiple service bundles**: 25% lower churn
        - **Tech support + Online security**: 15% churn rate
        """)

        st.subheader("Business Recommendations")
        st.info("""
        **Based on model insights:**
        
        1. **Convert month-to-month contracts** - Offer incentives for 1-2 year commitments
        2. **Improve fiber optic service** - Highest churn segment, needs investigation
        3. **Encourage automatic payments** - Credit card users churn 30% less
        4. **Target new customers** - Enhanced onboarding for first 12 months
        5. **Bundle security features** - Free tech support trial reduces churn
        """)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p>Built using Streamlit | Model: Decision Tree (75.5% accuracy) | Data: Telco Customer Churn Dataset</p>
            <p>© 2025 R. S. P. S. Uthsara | BSc (Hons) AI & Data Science</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()