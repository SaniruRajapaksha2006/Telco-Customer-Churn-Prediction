"""
Feature engineering module
Based on CM2604 coursework preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handle all feature engineering operations"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def encode_binary_features(self, df, fit=True):
        """Encode binary categorical features"""
        df_encoded = df.copy()
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        
        for col in binary_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                    print(f"  {col}: {self.label_encoders[col].classes_} → {self.label_encoders[col].transform(self.label_encoders[col].classes_)}")
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def one_hot_encode(self, df, fit=True):
        """One-hot encode multi-category features"""
        df_encoded = df.copy()
        multi_cat_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        
        # Filter to columns that exist
        cols_to_encode = [col for col in multi_cat_cols if col in df_encoded.columns]
        
        if cols_to_encode:
            df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True)
            logger.info(f"One-hot encoded {len(cols_to_encode)} columns")
        
        return df_encoded
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        df_scaled = df.copy()
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        numerical_cols = [col for col in numerical_cols if col in df_scaled.columns]
        
        if numerical_cols:
            if fit:
                df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
                logger.info(f"Scaled {len(numerical_cols)} numerical features")
            else:
                df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        return df_scaled

    def prepare_data(self, df, target_col='Churn', fit=True):
        """Complete preprocessing pipeline"""
        # Don't clean again - assume data is already cleaned
        # df is already cleaned from train.py

        # Encode binary features
        df = self.encode_binary_features(df, fit=fit)

        # One-hot encode multi-category features
        df = self.one_hot_encode(df, fit=fit)

        # Encode target
        if target_col in df.columns:
            le_target = LabelEncoder()
            y = le_target.fit_transform(df[target_col])
            df = df.drop(target_col, axis=1)
        else:
            y = None

        # Scale features
        df = self.scale_features(df, fit=fit)

        return df, y
    
    def apply_smote(self, X, y):
        """Apply SMOTE for class imbalance"""
        logger.info(f"Before SMOTE - Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"After SMOTE - Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, path):
        """Save preprocessing artifacts"""
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        joblib.dump(artifacts, path)
        logger.info(f"Saved artifacts to {path}")
    
    def load_artifacts(self, path):
        """Load preprocessing artifacts"""
        artifacts = joblib.load(path)
        self.label_encoders = artifacts['label_encoders']
        self.scaler = artifacts['scaler']
        logger.info(f"Loaded artifacts from {path}")
