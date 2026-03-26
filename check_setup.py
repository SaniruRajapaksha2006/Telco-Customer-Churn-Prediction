#!/usr/bin/env python3
"""Quick diagnostic script to verify project setup"""

import os
import sys
from pathlib import Path

def check_file(filepath, required_strings=None):
    """Check if file exists and has required content"""
    if not filepath.exists():
        return f"❌ Missing: {filepath}"
    
    size = filepath.stat().st_size
    if size == 0:
        return f"⚠️ Empty: {filepath}"
    
    if required_strings:
        content = filepath.read_text()
        missing = [s for s in required_strings if s not in content]
        if missing:
            return f"⚠️ {filepath} missing: {missing}"
    
    return f"✅ OK: {filepath} ({size} bytes)"

def main():
    print("=" * 60)
    print("PROJECT SETUP DIAGNOSTIC")
    print("=" * 60)
    
    base = Path.cwd()
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check structure
    print("\n📁 Checking file structure:")
    
    files_to_check = [
        ("src/data/make_dataset.py", ["def load_data", "def clean_data"]),
        ("src/features/build_features.py", ["class FeatureEngineer", "def apply_smote"]),
        ("src/models/train_model.py", ["class ModelTrainer", "def train_decision_tree"]),
        ("train.py", ["def main"]),
        ("app/app.py", ["import streamlit", "st.set_page_config"]),
        ("src/__init__.py", None),
        ("src/data/__init__.py", None),
        ("src/features/__init__.py", None),
        ("src/models/__init__.py", None),
        ("requirements.txt", ["pandas", "tensorflow"]),
        (".gitignore", ["venv", "__pycache__"]),
        ("README.md", ["# Telco"]),
    ]
    
    for filepath, required in files_to_check:
        result = check_file(base / filepath, required)
        print(f"  {result}")
    
    # Check imports
    print("\n📦 Checking imports:")
    sys.path.insert(0, str(base))
    
    try:
        from src.data import make_dataset
        print("  ✅ src.data.make_dataset imports OK")
    except Exception as e:
        print(f"  ❌ src.data.make_dataset: {e}")
    
    try:
        from src.features.build_features import FeatureEngineer
        print("  ✅ src.features.build_features imports OK")
    except Exception as e:
        print(f"  ❌ src.features.build_features: {e}")
    
    try:
        from src.models.train_model import ModelTrainer
        print("  ✅ src.models.train_model imports OK")
    except Exception as e:
        print(f"  ❌ src.models.train_model: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Diagnostic complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
