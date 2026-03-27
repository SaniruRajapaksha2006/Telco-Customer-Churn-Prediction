"""
Model training module
Based on CM2604 coursework models
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report,
                           confusion_matrix)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def train_decision_tree(self, X_train, y_train, ccp_alpha=0.000629, random_state=42):
        """Train Decision Tree with pruning"""
        logger.info(f"Training Decision Tree with ccp_alpha={ccp_alpha}")
        
        dt = DecisionTreeClassifier(
            random_state=random_state,
            ccp_alpha=ccp_alpha
        )
        dt.fit(X_train, y_train)
        
        self.models['Decision Tree'] = dt
        logger.info("Decision Tree training complete")
        
        return dt
    
    def find_optimal_alpha(self, X_train, y_train, X_val, y_val):
        """Find optimal ccp_alpha for pruning"""
        from sklearn.tree import DecisionTreeClassifier
        
        # Train initial tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        
        # Get pruning path
        path = dt.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas
        
        # Evaluate each alpha
        results = []
        for alpha in ccp_alphas:
            pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
            pruned_tree.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, pruned_tree.predict(X_train))
            val_acc = accuracy_score(y_val, pruned_tree.predict(X_val))
            
            results.append({
                'alpha': alpha,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            })
        
        # Find best alpha
        best_result = max(results, key=lambda x: x['val_accuracy'])
        logger.info(f"Optimal alpha: {best_result['alpha']:.6f}")
        logger.info(f"Validation accuracy: {best_result['val_accuracy']:.4f}")
        
        return best_result['alpha']

    def build_neural_network(self, input_shape):
        """Build Neural Network architecture"""
        # Use legacy Adam for Mac M1/M2 compatibility
        from tensorflow.keras.optimizers import legacy

        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=legacy.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    def train_neural_network(self, X_train, y_train, X_val, y_val,
                             input_shape, epochs=100, batch_size=32):
        """Train Neural Network with callbacks"""
        logger.info("Training Neural Network...")

        # Convert to float32 explicitly for TensorFlow
        X_train = np.array(X_train).astype(np.float32)
        X_val = np.array(X_val).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)
        y_val = np.array(y_val).astype(np.float32)

        model = self.build_neural_network(input_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.models['Neural Network'] = model
        logger.info(f"Neural Network training complete - Stopped at epoch {len(history.history['loss'])}")

        return model, history

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        # Convert to float32 for Neural Network
        if model_name == "Neural Network":
            X_test = np.array(X_test).astype(np.float32)
            y_test = np.array(y_test).astype(np.float32)

        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test).flatten()
            y_pred = (y_prob > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }

        self.results[model_name] = metrics

        logger.info(f"{model_name} - Acc: {metrics['accuracy']:.4f}, "
                    f"Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, "
                    f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")

        # Print classification report
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

        return metrics, y_pred, y_prob
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance from Decision Tree"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            logger.warning("Model doesn't have feature_importances_ attribute")
            return None
    
    def save_model(self, model, model_name, path):
        """Save trained model"""
        if model_name == 'Decision Tree':
            joblib.dump(model, path)
        else:
            model.save(path)
        logger.info(f"Saved {model_name} to {path}")
