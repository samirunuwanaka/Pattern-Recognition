"""
Low-Level PyTorch Neural Network Architectures for Chronic Kidney Disease Prediction
Implements Multi-Layer Perceptron (CKDRiskNet) and low-level Logistic Regression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CKDRiskNet(nn.Module):
    """
    Multi-Layer Perceptron for Chronic Kidney Disease Risk Factor Prediction.
    Architecture:
      Input (d_in=26) -> Linear(d_in, 32) -> BatchNorm1d -> ReLU -> Dropout(0.2)
                      -> Linear(32, 16) -> BatchNorm1d -> ReLU -> Dropout(0.1)
                      -> Linear(16, 8) -> ReLU
                      -> Linear(8, 1) -> Output Logit
    """
    def __init__(self, input_dim=26, hidden1=32, hidden2=16, hidden3=8, dropout_rate=0.2):
        super(CKDRiskNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc_out = nn.Linear(hidden3, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, x):
        # Layer 1
        h1 = self.fc1(x)
        if h1.shape[0] > 1:
            h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        
        # Layer 2
        h2 = self.fc2(h1)
        if h2.shape[0] > 1:
            h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        
        # Layer 3
        h3 = F.relu(self.fc3(h2))
        
        # Output Logits
        logits = self.fc_out(h3)
        return logits
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

class LowLevelLogisticRegression(nn.Module):
    """
    Linear Risk Model in PyTorch using explicit low-level tensor operations.
    Computes logit = X * W + b, probability = sigmoid(logit).
    """
    def __init__(self, input_dim=26):
        super(LowLevelLogisticRegression, self).__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        # Low-level matrix multiplication
        logits = torch.matmul(x, self.weight) + self.bias
        return logits
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs