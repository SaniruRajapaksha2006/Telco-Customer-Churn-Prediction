# Telco Customer Churn Prediction
Machine Learning project to predict customer churn

## 📊 Model Performance Results

| Metric | Decision Tree | Neural Network |
|--------|--------------|----------------|
| **Accuracy** | 75.5% | 75.5% |
| **Precision** | 52.7% | 53.0% |
| **Recall** | **74.6%** | 69.0% |
| **F1 Score** | **61.8%** | 59.9% |
| **ROC-AUC** | **82.3%** | 81.5% |

### Top 5 Most Important Features (Decision Tree)
1. **Contract_Two year** (30.9%) - Customers on 2-year contracts rarely churn
2. **Contract_One year** (18.5%) - Annual contracts show strong retention
3. **Tenure** (16.1%) - New customers are at highest risk
4. **InternetService_Fiber optic** (9.7%) - Fiber customers churn more
5. **MonthlyCharges** (7.3%) - Higher charges correlate with churn

### Key Insights
- **Contract type is the strongest predictor** - month-to-month contracts have 42.7% churn vs 2.8% for two-year
- **New customers need attention** - 40% of churn happens in first 12 months
- **Fiber optic service requires improvement** - 41.9% churn rate vs 7.4% for no internet
- **Electronic check users are high risk** - 45.3% churn vs 15.2% for credit card
