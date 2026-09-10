"""
Training & Evaluation Engine for Chronic Kidney Disease Prediction
Low-level PyTorch training loop, evaluation metrics, and JS weight export.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.preprocess import load_and_preprocess_data, get_train_val_test_split
from src.model import CKDRiskNet, LowLevelLogisticRegression

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_prob_np = y_prob.detach().cpu().numpy().flatten()
    y_pred_np = (y_prob_np >= threshold).astype(int)
    
    acc = accuracy_score(y_true_np, y_pred_np)
    prec = precision_score(y_true_np, y_pred_np, zero_division=0)
    rec = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_true_np, y_pred_np).tolist()
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm
    }

def train_model(model, train_data, val_data, epochs=200, lr=0.005, weight_decay=1e-4, batch_size=32):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_weights = None
    patience = 30
    patience_counter = 0
    
    n_samples = X_train.shape[0]
    
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for b in range(n_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_idx)
            
        train_loss = epoch_loss / n_samples
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            
            train_probs = torch.sigmoid(model(X_train))
            val_probs = torch.sigmoid(val_logits)
            
            train_acc = ((train_probs >= 0.5) == y_train).float().mean().item()
            val_acc = ((val_probs >= 0.5) == y_val).float().mean().item()
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch >= 60:
            print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break
            
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history

def export_model_for_js(model, preprocessor_meta, export_dir="model"):
    os.makedirs(export_dir, exist_ok=True)
    
    # Save PyTorch state dict
    torch.save(model.state_dict(), os.path.join(export_dir, "ckd_risk_net.pt"))
    
    # Extract weights for clean JS evaluation
    state_dict = model.state_dict()
    weights_json = {}
    
    if isinstance(model, CKDRiskNet):
        weights_json = {
            'type': 'CKDRiskNet',
            'fc1_w': state_dict['fc1.weight'].cpu().numpy().tolist(),
            'fc1_b': state_dict['fc1.bias'].cpu().numpy().tolist(),
            'bn1_mean': state_dict['bn1.running_mean'].cpu().numpy().tolist(),
            'bn1_var': state_dict['bn1.running_var'].cpu().numpy().tolist(),
            'bn1_w': state_dict['bn1.weight'].cpu().numpy().tolist(),
            'bn1_b': state_dict['bn1.bias'].cpu().numpy().tolist(),
            'fc2_w': state_dict['fc2.weight'].cpu().numpy().tolist(),
            'fc2_b': state_dict['fc2.bias'].cpu().numpy().tolist(),
            'bn2_mean': state_dict['bn2.running_mean'].cpu().numpy().tolist(),
            'bn2_var': state_dict['bn2.running_var'].cpu().numpy().tolist(),
            'bn2_w': state_dict['bn2.weight'].cpu().numpy().tolist(),
            'bn2_b': state_dict['bn2.bias'].cpu().numpy().tolist(),
            'fc3_w': state_dict['fc3.weight'].cpu().numpy().tolist(),
            'fc3_b': state_dict['fc3.bias'].cpu().numpy().tolist(),
            'fc_out_w': state_dict['fc_out.weight'].cpu().numpy().tolist(),
            'fc_out_b': state_dict['fc_out.bias'].cpu().numpy().tolist()
        }
    elif isinstance(model, LowLevelLogisticRegression):
        weights_json = {
            'type': 'LowLevelLogisticRegression',
            'weight': state_dict['weight'].cpu().numpy().flatten().tolist(),
            'bias': state_dict['bias'].cpu().numpy().flatten().tolist()
        }
        
    with open(os.path.join(export_dir, "model_weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights_json, f, indent=2)
        
    with open(os.path.join(export_dir, "preprocessor.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessor_meta, f, indent=2)
        
    # Also save a copy in web/assets for seamless frontend loading
    os.makedirs("web/assets", exist_ok=True)
    with open("web/assets/model_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_json, f, indent=2)
    with open("web/assets/preprocessor.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor_meta, f, indent=2)
        
    print(f"Model and Preprocessor exported successfully to {export_dir} and web/assets/")

def main():
    print("=== Training Low-Level PyTorch CKD Risk Model ===")
    X_num, X_scaled, y, preprocessor_meta = load_and_preprocess_data()
    train_data, val_data, test_data = get_train_val_test_split(X_scaled, y, seed=42)
    
    print(f"Dataset Split: Train={train_data[0].shape[0]}, Val={val_data[0].shape[0]}, Test={test_data[0].shape[0]}")
    
    model = CKDRiskNet(input_dim=preprocessor_meta['num_features'], hidden1=32, hidden2=16, hidden3=8)
    trained_model, history = train_model(model, train_data, val_data, epochs=150, lr=0.008)
    
    X_test, y_test = test_data
    test_probs = trained_model.predict_proba(X_test)
    test_metrics = compute_metrics(y_test, test_probs)
    
    print("\n=== Test Set Performance Metrics ===")
    print(f"Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall   : {test_metrics['recall']*100:.2f}%")
    print(f"F1-Score : {test_metrics['f1']*100:.2f}%")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")
    
    export_model_for_js(trained_model, preprocessor_meta)

if __name__ == "__main__":
    main()