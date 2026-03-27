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

# Load model and preprocessor
@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    model_path = Path(__file__).parent.parent / "models"

    # Load Decision Tree (best model)
    dt_model = joblib.load(model_path / "decision_tree.pkl")

    # Load preprocessor
    engineer = FeatureEngineer()
    engineer.load_artifacts(model_path / "preprocessor.pkl")

    return dt_model, engineer

def main():
    st.title("📱 Telco Customer Churn Predictor")

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
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
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

        if st.button("🔍 Predict Churn Risk", type="primary"):
            try:
                # Preprocess input
                X_processed, _ = engineer.prepare_data(input_data, fit=False)

                # Make prediction
                prediction = model.predict(X_processed)
                probability = model.predict_proba(X_processed)[0]

                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")

                col_result1, col_result2, col_result3 = st.columns(3)

                with col_result1:
                    if prediction[0] == 1:
                        st.error("### ⚠️ HIGH RISK")
                        st.write("**Customer is likely to churn**")
                    else:
                        st.success("### ✅ LOW RISK")
                        st.write("**Customer is likely to stay**")

                with col_result2:
                    st.metric("Churn Probability", f"{probability[1]:.1%}")

                with col_result3:
                    st.metric("Retention Probability", f"{probability[0]:.1%}")

                # Risk factors
                st.subheader("⚠️ Key Risk Factors")
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
                        st.write(factor)
                else:
                    st.success("No major risk factors detected! This customer has good retention characteristics.")

                # Recommendation
                st.subheader("💡 Recommendation")
                if prediction[0] == 1:
                    st.warning("""
                    **Suggested Actions:**
                    - Offer contract upgrade incentive (2-year contract)
                    - Bundle security features
                    - Provide tech support trial
                    - Send retention offer
                    """)
                else:
                    st.info("""
                    **Maintain Good Practices:**
                    - Continue current service quality
                    - Consider loyalty rewards
                    - Monitor for any service issues
                    """)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Please check your input values and try again.")

    with tab2:
        st.header("📁 Batch Prediction")
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

                if st.button("🚀 Run Batch Prediction"):
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

                        st.success("✅ Predictions complete!")
                        st.dataframe(batch_data)

                        # Summary statistics
                        st.subheader("📊 Summary Statistics")
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
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.write("Please check that your CSV has the required columns.")

    with tab3:
        st.header("📈 Model Insights")

        st.subheader("🎯 Top Features Influencing Churn")
        st.markdown("""
        | Rank | Feature | Importance | Insight |
        |------|---------|------------|---------|
        | 1 | **Contract_Two year** | 30.9% | 2-year contracts have only 2.8% churn |
        | 2 | **Contract_One year** | 18.5% | Annual contracts show strong retention |
        | 3 | **Tenure** | 16.1% | 40% of churn happens in first 12 months |
        | 4 | **InternetService_Fiber optic** | 9.7% | Fiber optic: 41.9% churn rate |
        | 5 | **MonthlyCharges** | 7.3% | Higher charges = higher churn risk |
        """)

        st.subheader("🔴 High Risk Profiles")
        st.markdown("""
        - **Month-to-month contract + Fiber optic + Electronic check**: 78% churn rate
        - **New customer (<6 months) + High monthly charges (>$80)**: 67% churn rate
        - **Senior citizen + No tech support**: 54% churn rate
        """)

        st.subheader("🟢 Protective Factors")
        st.markdown("""
        - **Two-year contract**: 97% retention rate
        - **Credit card payment**: 85% retention rate
        - **Multiple service bundles**: 25% lower churn
        - **Tech support + Online security**: 15% churn rate
        """)

        st.subheader("💼 Business Recommendations")
        st.info("""
        **Based on model insights:**
        
        1. **Convert month-to-month contracts** - Offer incentives for 1-2 year commitments
        2. **Improve fiber optic service** - Highest churn segment, needs investigation
        3. **Encourage automatic payments** - Credit card users churn 30% less
        4. **Target new customers** - Enhanced onboarding for first 12 months
        5. **Bundle security features** - Free tech support trial reduces churn
        """)

if __name__ == "__main__":
    main()