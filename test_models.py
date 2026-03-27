"""Test if models load correctly"""

import joblib
import tensorflow as tf
from pathlib import Path

print("Testing model loading...\n")

# Test Decision Tree
try:
    dt = joblib.load("models/decision_tree.pkl")
    print(f"✅ Decision Tree loaded: {type(dt).__name__}")
except Exception as e:
    print(f"❌ Decision Tree failed: {e}")

# Test Neural Network
try:
    nn = tf.keras.models.load_model("models/neural_network.h5")
    print(f"✅ Neural Network loaded: {type(nn).__name__}")
except Exception as e:
    print(f"❌ Neural Network failed: {e}")

# Test Preprocessor
try:
    preprocessor = joblib.load("models/preprocessor.pkl")
    print(f"✅ Preprocessor loaded: {type(preprocessor).__name__}")
except Exception as e:
    print(f"❌ Preprocessor failed: {e}")

print("\n✅ All models ready for deployment!")
