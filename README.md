# 📊 Telco Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

This project develops a machine learning system to predict customer churn in the telecommunications industry. By analyzing historical customer data including demographics, account information, and service usage patterns, we can identify customers at risk of leaving and enable proactive retention strategies.

### Business Impact

- Acquiring new customers costs **5–10x more** than retaining existing ones
- Reducing churn by just 5% can increase profits by **25–95%**
- Early identification of at-risk customers saves millions in revenue

---

## 📊 Model Performance Results

### Decision Tree (Best Model)

| Metric | Score |
|--------|-------|
| **Accuracy** | 75.5% |
| **Precision** | 52.7% |
| **Recall** | **74.6%** |
| **F1 Score** | **61.8%** |
| **ROC-AUC** | **82.3%** |

### Neural Network

| Metric | Score |
|--------|-------|
| **Accuracy** | 75.5% |
| **Precision** | 53.0% |
| **Recall** | 69.0% |
| **F1 Score** | 59.9% |
| **ROC-AUC** | 81.5% |

### Top 5 Features (Decision Tree)

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | **Contract_Two year** | 30.9% | 2-year contracts have only 2.8% churn |
| 2 | **Contract_One year** | 18.5% | Annual contracts show strong retention |
| 3 | **Tenure** | 16.1% | 40% of churn happens in first 12 months |
| 4 | **InternetService_Fiber optic** | 9.7% | Fiber optic: 41.9% churn rate |
| 5 | **MonthlyCharges** | 7.3% | Higher charges = higher churn risk |

---

## 🔍 Key Insights

### High Risk Factors

- **Month-to-month contracts**: 42.7% churn vs 2.8% for two-year
- **Electronic check payment**: 45.3% churn vs 15.2% for credit card
- **Fiber optic internet**: 41.9% churn vs 7.4% for no internet
- **No tech support**: 41.6% churn vs 15.3% with support
- **New customers (<12 months)**: 40% of churn happens in first year

### Protective Factors

- **Two-year contracts**: 97% retention rate
- **Credit card payments**: 85% retention rate
- **Multiple service bundles**: 25% lower churn
- **Tech support + online security**: 15% churn rate
- **Senior citizens with dependents**: 30% lower churn

---

## 🏗️ Project Structure

```
Telco-Customer-Churn-Prediction/
├── data/                   # Dataset storage
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned and processed data
├── notebooks/              # Jupyter notebooks for EDA
├── src/                    # Source code
│   ├── data/               # Data loading utilities
│   │   └── make_dataset.py # Data cleaning and loading
│   ├── features/           # Feature engineering
│   │   └── build_features.py  # Feature preprocessing with SMOTE
│   └── models/             # Model training
│       └── train_model.py  # Decision Tree & Neural Network
├── app/                    # Streamlit web app
│   └── app.py              # Main application
├── models/                 # Trained models (generated locally)
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Installation

```bash
# Clone the repository
git clone https://github.com/Saniru2006/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset and place it in data/raw/

# Train models
python train.py

# Run the web app
streamlit run app/app.py
```

---

## 🎨 Features

### Single Customer Prediction

- Input customer demographics and service details
- Real-time churn probability prediction
- Identify specific risk factors
- Get actionable retention recommendations

### Batch Prediction

- Upload CSV files with multiple customers
- Bulk predictions with probability scores
- Download results as CSV
- Summary statistics of predictions

### Model Insights

- Feature importance analysis
- High-risk customer profiles
- Protective factors
- Business recommendations

---

## 💡 Business Recommendations

Based on model insights:

1. **Convert month-to-month contracts** — Offer incentives for 1–2 year commitments to reduce churn by up to 40%
2. **Improve fiber optic service** — Highest churn segment (41.9%), requires immediate investigation
3. **Encourage automatic payments** — Credit card users churn 30% less than electronic check users
4. **Target new customers** — Enhanced onboarding and engagement for first 12 months
5. **Bundle security features** — Free tech support trial reduces churn by 26%
6. **Senior citizen program** — Specialized support for seniors who have 1.8x higher churn risk

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, TensorFlow, Keras |
| Imbalance Handling | SMOTE (Synthetic Minority Over-sampling Technique) |
| Hyperparameter Tuning | KerasTuner, GridSearchCV |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## 📈 Model Architecture

### Decision Tree

- Pruned using cost-complexity pruning (`ccp_alpha = 0.000629`)
- Gini impurity criterion
- Maximum depth automatically determined by pruning
- Feature importance derived from tree structure

### Neural Network

- Architecture: `128 → 64 → 32 → 1`
- Batch Normalization after each hidden layer
- Dropout rates: 0.3, 0.2, 0.1
- Early stopping with `patience = 15`
- Learning rate reduction on plateau (`factor = 0.5`)
- Adam optimizer with initial `learning rate = 0.001`

---

## 🔬 Data Preprocessing

The preprocessing pipeline includes:

**1. Data Cleaning**
- Handling missing values in `TotalCharges` (filled with 0)
- Removing `customerID` for privacy

**2. Feature Engineering**
- Binary encoding for two-category features
- One-hot encoding for multi-category features
- Feature scaling with `StandardScaler`

**3. Class Imbalance Handling**
- SMOTE applied to training data
- Balanced dataset from 26.5% to 50% churn rate

---

## 📊 Dataset Statistics

| Property | Value |
|----------|-------|
| Total Records | 7,043 customers |
| Features | 21 attributes |
| Target | Churn (Yes/No) |
| Class Distribution | 73.5% Non-churn, 26.5% Churn |
| Data Source | [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

---

## 🔮 Future Improvements

- [ ] Deploy to Streamlit Cloud for live demo
- [ ] Add SHAP explanations for model interpretability
- [ ] Implement ensemble methods (Random Forest, XGBoost)
- [ ] Add real-time monitoring for model drift
- [ ] Create REST API with FastAPI
- [ ] Containerize with Docker
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Include A/B testing framework

---

## 📝 Ethical Considerations

- **Privacy Protection**: CustomerID removed during preprocessing
- **Fairness Audits**: Analysis across demographic subgroups
- **Transparency**: Feature importance and model decisions are explainable
- **Accountability**: Full documentation of all modeling decisions
- **Reproducibility**: Random seeds set for consistent results

---

## 👨‍💻 Author

**R. S. P. S. Uthsara**
- Student ID: 2425606
- BSc (Hons) Artificial Intelligence and Data Science
- IIT Sri Lanka / Robert Gordon University

---

## 🙏 Acknowledgments

- **IIT Sri Lanka** — for providing the learning environment and resources
- **IBM** — for providing the Telco Customer Churn dataset
- **Open Source Community** — for the amazing tools and libraries

---

## 📄 License


---

## 📧 Contact

- **GitHub**: [@SaniruRajapaksha2006](https://github.com/SaniruRajapaksha2006)
- **Project Link**: [https://github.com/SaniruRajapaksha2006/Telco-Customer-Churn-Prediction](https://github.com/SaniruRajapaksha2006/Telco-Customer-Churn-Prediction)

---

⭐ **Star the Project**

If you found this project helpful, please consider giving it a star on GitHub! It helps others discover the project.