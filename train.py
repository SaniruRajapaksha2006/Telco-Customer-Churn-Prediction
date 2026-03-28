"""
Main training script
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle

from src.data.make_dataset import load_data, clean_data
from src.features.build_features import FeatureEngineer
from src.models.train_model import ModelTrainer

def main():
    # Paths
    data_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("TELCO CUSTOMER CHURN PREDICTION")
    print("=" * 60)
    
    # Load and clean data
    print("\n1. Loading and cleaning data...")
    df = load_data(data_path)
    df = clean_data(df)
    
    # Prepare features
    print("\n2. Preparing features...")
    engineer = FeatureEngineer()
    X, y = engineer.prepare_data(df, fit=True)

    # Store training columns for later prediction
    training_columns = list(X.columns)
    print(f"Training columns: {len(training_columns)} features")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = engineer.split_data(X, y)
    
    # Apply SMOTE
    print("\n4. Applying SMOTE for class balance...")
    X_train_resampled, y_train_resampled = engineer.apply_smote(X_train, y_train)
    
    # Split validation set
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
    )
    
    # Train models
    trainer = ModelTrainer()
    
    print("\n5. Training Decision Tree...")
    dt_model = trainer.train_decision_tree(X_train_final, y_train_final, ccp_alpha=0.000629)
    
    print("\n6. Training Neural Network...")
    nn_model, history = trainer.train_neural_network(
        X_train_final, y_train_final, X_val, y_val,
        input_shape=(X_train_final.shape[1],)
    )
    
    # Evaluate
    print("\n7. Evaluating models...")
    print("\n" + "=" * 60)
    print("DECISION TREE RESULTS")
    print("=" * 60)
    dt_metrics, dt_pred, dt_prob = trainer.evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK RESULTS")
    print("=" * 60)
    nn_metrics, nn_pred, nn_prob = trainer.evaluate_model(nn_model, X_test, y_test, "Neural Network")

    # Save models and preprocessor
    print("\n8. Saving models...")

    # Save Decision Tree
    trainer.save_model(dt_model, "Decision Tree", models_dir / "decision_tree.pkl")

    # Save Neural Network
    trainer.save_model(nn_model, "Neural Network", models_dir / "neural_network.h5")

    # Save preprocessor as FULL OBJECT (not dict)
    joblib.dump(engineer, models_dir / "preprocessor.pkl")
    print("Saved preprocessor (FeatureEngineer object)")

    # Save training columns separately for verification
    if hasattr(engineer, 'training_columns'):
        joblib.dump(engineer.training_columns, models_dir / "training_columns.pkl")
        print(f"Saved {len(engineer.training_columns)} training columns")
    
    # Feature importance
    print("\n9. Feature Importance (Decision Tree):")
    importance = trainer.get_feature_importance(dt_model, X.columns)
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))
    
    print("\nTraining complete! Models saved to models/")
    print("\nTo run the web app: streamlit run app/app.py")

if __name__ == "__main__":
    main()
