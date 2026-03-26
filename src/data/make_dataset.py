"""
Data loading and preprocessing module
Based on CM2604 coursework
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load Telco Customer Churn dataset"""
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """Clean the dataset"""
    logger.info("Starting data cleaning")
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0
    missing_after = df_clean['TotalCharges'].isnull().sum()
    if missing_after > 0:
        logger.info(f"Filling {missing_after} missing values with 0")
        df_clean['TotalCharges'].fillna(0, inplace=True)
    
    # Remove customerID for privacy
    df_clean = df_clean.drop('customerID', axis=1)
    logger.info("Removed customerID column")
    
    return df_clean

def get_data_info(df):
    """Get dataset information"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'churn_distribution': df['Churn'].value_counts().to_dict(),
        'churn_percentage': (df['Churn'].value_counts(normalize=True) * 100).to_dict()
    }
    return info

if __name__ == "__main__":
    # Test the module
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if data_path.exists():
        df = load_data(data_path)
        df = clean_data(df)
        info = get_data_info(df)
        print("\n=== Dataset Information ===")
        print(f"Shape: {info['shape']}")
        print(f"Churn Distribution: {info['churn_distribution']}")
    else:
        print(f"Data file not found at {data_path}")
        print("Please download the dataset from Kaggle and place it in data/raw/")
