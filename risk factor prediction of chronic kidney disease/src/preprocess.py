"""
Chronic Kidney Disease (CKD) Preprocessing Pipeline
Handles UCI Dataset 857 ingestion, interval parsing, numeric standardization,
and PyTorch tensor conversion.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch

FEATURE_COLUMNS = [
    'bp (Diastolic)', 'bp limit', 'sg', 'al', 'rbc', 'su', 'pc', 'pcc',
    'ba', 'bgr', 'bu', 'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc',
    'wbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'grf', 'age'
]

FEATURE_LABELS = {
    'bp (Diastolic)': 'Diastolic Blood Pressure',
    'bp limit': 'Blood Pressure Limit Status',
    'sg': 'Specific Gravity',
    'al': 'Albumin Level',
    'rbc': 'Red Blood Cells (Abnormal/Normal)',
    'su': 'Sugar Level',
    'pc': 'Pus Cell (Abnormal/Normal)',
    'pcc': 'Pus Cell Clumps (Present/Not)',
    'ba': 'Bacteria (Present/Not)',
    'bgr': 'Blood Glucose Random (mg/dL)',
    'bu': 'Blood Urea (mg/dL)',
    'sod': 'Sodium (mEq/L)',
    'sc': 'Serum Creatinine (mg/dL)',
    'pot': 'Potassium (mEq/L)',
    'hemo': 'Hemoglobin (g/dL)',
    'pcv': 'Packed Cell Volume (%)',
    'rbcc': 'Red Blood Cell Count (mcmL)',
    'wbcc': 'White Blood Cell Count (cells/cumm)',
    'htn': 'Hypertension History',
    'dm': 'Diabetes Mellitus History',
    'cad': 'Coronary Artery Disease',
    'appet': 'Appetite (Good/Poor)',
    'pe': 'Pedal Edema',
    'ane': 'Anemia History',
    'grf': 'Glomerular Filtration Rate (eGFR)',
    'age': 'Patient Age'
}

def parse_interval_to_numeric(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if not val_str or val_str.lower() == 'p' or val_str == '?':
        return np.nan
    if val_str.startswith('<') or val_str.startswith('≤'):
        clean = re.sub(r'[^\d.]', '', val_str)
        if clean:
            num = float(clean)
            return num * 0.75 if num > 1 else num * 0.95
    if val_str.startswith('≥') or val_str.startswith('>='):
        clean = re.sub(r'[^\d.]', '', val_str)
        if clean:
            num = float(clean)
            return num * 1.15
    if '-' in val_str:
        parts = val_str.split('-')
        clean0 = re.sub(r'[^\d.]', '', parts[0])
        clean1 = re.sub(r'[^\d.]', '', parts[1])
        if clean0 and clean1:
            return (float(clean0) + float(clean1)) / 2.0
    try:
        clean = re.sub(r'[^\d.]', '', val_str)
        return float(clean) if clean else np.nan
    except Exception:
        return np.nan

def load_and_preprocess_data(csv_path="data/ckd-dataset-v2.csv"):
    df_raw = pd.read_csv(csv_path, encoding='utf-8')
    df = df_raw.iloc[2:].reset_index(drop=True)
    
    y = df['affected'].astype(int).values
    
    X_raw = df[FEATURE_COLUMNS].copy()
    X_num = pd.DataFrame()
    
    for col in FEATURE_COLUMNS:
        X_num[col] = X_raw[col].apply(parse_interval_to_numeric)
        median_val = X_num[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_num[col] = X_num[col].fillna(median_val)
        
    means = X_num.mean().to_dict()
    stds = X_num.std().replace(0, 1.0).to_dict()
    mins = X_num.min().to_dict()
    maxs = X_num.max().to_dict()
    
    X_scaled = (X_num - pd.Series(means)) / pd.Series(stds)
    
    preprocessor_meta = {
        'feature_columns': FEATURE_COLUMNS,
        'feature_labels': FEATURE_LABELS,
        'means': means,
        'stds': stds,
        'mins': mins,
        'maxs': maxs,
        'num_features': len(FEATURE_COLUMNS)
    }
    
    return X_num, X_scaled, y, preprocessor_meta

def get_train_val_test_split(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
    n = len(y)
    indices = np.random.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = np.array(X)
        
    X_train, y_train = torch.tensor(X_np[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val, y_val = torch.tensor(X_np[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)
    X_test, y_test = torch.tensor(X_np[test_idx], dtype=torch.float32), torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)