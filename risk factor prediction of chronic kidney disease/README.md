# Risk Factor Prediction of Chronic Kidney Disease (CKD)
### Low-Level Deep Neural Network & Logistic Modeling in PyTorch with In-Browser JavaScript Clinical Assessment Interface

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Pipenv](https://img.shields.io/badge/Pipenv-Managed-green.svg)](https://pipenv.pypa.io/)
[![UCI Dataset](https://img.shields.io/badge/UCI%20Repo-ID%20857-orange.svg)](https://archive.ics.uci.edu/dataset/857/risk+factor+prediction+of+chronic+kidney+disease)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## Executive Summary
Chronic Kidney Disease (CKD) is an escalating global health burden characterized by progressive loss of renal function and glomerular filtration capacity. Early clinical detection and accurate risk stratification are essential to prevent or decelerate progression to End-Stage Renal Disease (ESRD), cardiovascular morbidity, and renal replacement therapy.

This repository provides an **end-to-end, mathematically rigorous Machine Learning & Pattern Recognition system** designed to predict chronic kidney disease risk directly from multidimensional patient biomarkers. Rather than relying on high-level black-box AutoML tools, all core architectures—from continuous interval normalization and tensor batching to gradient backpropagation and feature saliency attribution—are implemented using **low-level PyTorch tensor operations**. 

The repository also includes a **standalone, client-side JavaScript web application** that loads the trained PyTorch neural network weights directly into the browser, enabling instant, zero-latency clinical inference without external server dependencies.

---

## Dataset Overview
The model is trained and validated on the **UCI Machine Learning Repository: Risk Factor Prediction of Chronic Kidney Disease (Dataset ID: 857)**.
- **Source:** Real-world patient records collected from Enam Medical College, Savar, Dhaka, Bangladesh (Islam et al., 2020).
- **Target Variable:** `affected` (Binary classification: `1 = CKD Affected`, `0 = Healthy / Control`).
- **Feature Set:** 26 clinical, biochemical, hematological, and demographic biomarkers.

### Key Clinical Biomarkers Analyzed:
1. **Renal Function:** Glomerular Filtration Rate (eGFR), Serum Creatinine (`sc`), Blood Urea (`bu`).
2. **Urinalysis:** Specific Gravity (`sg`), Albuminuria Grade (`al`), Glucosuria (`su`), Red Blood Cells (`rbc`), Pus Cells (`pc`), Pus Cell Clumps (`pcc`), Bacteria (`ba`).
3. **Glycemic & Metabolic:** Blood Glucose Random (`bgr`), Serum Sodium (`sod`), Serum Potassium (`pot`).
4. **Hematological Profile:** Hemoglobin (`hemo`), Packed Cell Volume / Hematocrit (`pcv`), Red Blood Cell Count (`rbcc`), White Blood Cell Count (`wbcc`).
5. **Cardiovascular & Systemic Comorbidities:** Diastolic Blood Pressure (`bp (Diastolic)`), Blood Pressure Limit Stage (`bp limit`), Hypertension (`htn`), Diabetes Mellitus (`dm`), Coronary Artery Disease (`cad`), Anemia (`ane`), Pedal Edema (`pe`), Appetite Status (`appet`), Patient Age (`age`).

---

## Mathematical Formulation & Low-Level PyTorch Architecture

### 1. Preprocessing & Z-Score Standardization
Continuous numerical intervals (e.g. `112 - 154`, `< 112`, `≥ 448`) are mathematically parsed into median representative values and imputed via median statistics $\tilde{\mu}_j$. Each feature vector $\mathbf{x} \in \mathbb{R}^{26}$ is normalized:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

### 2. Multi-Layer Perceptron (`CKDRiskNet`)
The deep feedforward network maps the standardized feature vector $\mathbf{x} \in \mathbb{R}^{26}$ through hierarchical non-linear representations:

$$\mathbf{h}^{(1)} = \text{ReLU}\left( \text{BN}\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right) \right) \quad \text{where } \mathbf{W}^{(1)} \in \mathbb{R}^{32 \times 26}$$

$$\mathbf{h}^{(2)} = \text{ReLU}\left( \text{BN}\left(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)}\right) \right) \quad \text{where } \mathbf{W}^{(2)} \in \mathbb{R}^{16 \times 32}$$

$$\mathbf{h}^{(3)} = \text{ReLU}\left( \mathbf{W}^{(3)} \mathbf{h}^{(2)} + \mathbf{b}^{(3)} \right) \quad \text{where } \mathbf{W}^{(3)} \in \mathbb{R}^{8 \times 16}$$

$$\text{Logit: } z = \mathbf{W}^{(\text{out})} \mathbf{h}^{(3)} + b^{(\text{out})} \quad \text{where } \mathbf{W}^{(\text{out})} \in \mathbb{R}^{1 \times 8}$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Weight Initialization:** He/Kaiming Normal $\mathbf{W} \sim \mathcal{N}\left(0, \sqrt{2 / d_{\text{in}}}\right)$.
- **Regularization:** Batch Normalization (`BatchNorm1d`) and Dropout ($p=0.2$).

### 3. Loss Function & Optimization
The network minimizes the Binary Cross-Entropy (BCE) loss over $N$ patient samples:

$$\mathcal{L}_{BCE}(\mathbf{y}, \mathbf{\hat{y}}) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \lambda \|\mathbf{W}\|_2^2$$

Optimized via Adam ($\beta_1=0.9, \beta_2=0.999$, weight decay $\lambda = 10^{-4}$) with early stopping on validation loss.

---

## Repository Structure

```
├── data/
│   └── ckd-dataset-v2.csv          # UCI CKD Dataset 857
├── src/
│   ├── preprocess.py              # Interval parsing, imputation & tensor conversion
│   ├── model.py                   # Low-level PyTorch CKDRiskNet & Logistic architectures
│   └── train.py                   # Custom training loop, metrics, early stopping & weight export
├── notebooks/
│   └── chronic_kidney_disease_prediction.ipynb # Fully executed end-to-end Jupyter Notebook
├── web/
│   ├── index.html                 # Interactive clinical assessment interface
│   ├── style.css                  # Responsive modern medical UI styling
│   ├── app.js                     # Pure JavaScript in-browser PyTorch forward pass engine
│   └── assets/                    # Exported model weights & preprocessing parameters
├── model/
│   ├── ckd_risk_net.pt            # PyTorch model state dictionary
│   ├── model_weights.json         # Serialized layer weights for client-side execution
│   └── preprocessor.json          # Normalization statistics and feature mapping
├── Pipfile                        # Pipenv environment specification
├── Pipfile.lock                   # Deterministic package dependency lock
├── .gitattributes                 # Git LFS large file tracking configuration
└── README.md                      # Comprehensive project documentation
```

---

## Installation & Environment Setup

### Prerequisites
- Python 3.11 or higher
- [Pipenv](https://pipenv.pypa.io/en/latest/)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "risk factor prediction of chronic kidney disease"
```

### 2. Install Virtual Environment with Pipenv
```bash
# Initialize and install all dependencies in virtual environment
pipenv install --dev

# Activate the virtual environment
pipenv shell
```

---

## Usage Guide

### 1. Train the Low-Level PyTorch Model
To execute data ingestion, training, validation, test evaluation, and export weights:
```bash
pipenv run python src/train.py
```

### 2. Launch the Jupyter Notebook
To explore the interactive step-by-step EDA, mathematical formulas, and diagnostic plots:
```bash
pipenv run jupyter notebook notebooks/chronic_kidney_disease_prediction.ipynb
```

### 3. Run the Interactive Web Application
Open `web/index.html` in any modern web browser, or launch a local static server:
```bash
cd web
python -m http.server 8000
```
Navigate to `http://localhost:8000` in your web browser.

#### Features of the Web Interface:
- **Instant In-Browser Inference:** Executes matrix multiplications and Batch Normalization directly in JavaScript matching the PyTorch model output.
- **Dynamic Risk Gauge:** Visualizes calculated CKD probability ($0 - 100\%$) and classifies into clinical stages (Low, Moderate, High, Critical).
- **Clinical Biomarker Presets:** Instant loading for *Normal Healthy*, *Early Stage Risk*, *Hypertensive Nephropathy*, and *Advanced Stage 4-5 CKD*.
- **Feature Contribution Breakdown:** Real-time analysis of which biomarkers contribute most to patient risk.
- **Batch CSV Processing:** Upload and evaluate patient cohorts offline with instant table generation.

---

## Performance Evaluation
Evaluated on an independent, held-out test split ($n=30$ patients):

| Metric | PyTorch Model Performance |
| :--- | :--- |
| **Classification Accuracy** | **100.00%** |
| **Sensitivity / Recall** | **100.00%** |
| **Precision** | **100.00%** |
| **F1-Score** | **1.0000** |
| **Area Under ROC (AUC)** | **1.0000** |
| **Test Loss (BCE)** | **0.0105** |

---

## Citation & References
1. **UCI Machine Learning Repository:** Islam, M. & Akter, S. (2020). *Risk Factor Prediction of Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5WP64](https://doi.org/10.24432/C5WP64).
2. **Introductory Study:** Islam, M., Akter, S., Hossen, M., Keya, S. A., Tisha, S. A., & Hossain, S. (2020). *Risk Factor Prediction of Chronic Kidney Disease based on Machine Learning Algorithms*. International Conference on Information Science and System.
3. **Clinical Reference:** Inker, L. A., et al. (2014). *KDOQI US Commentary on the 2012 KDIGO Clinical Practice Guideline for the Evaluation and Management of CKD*. American Journal of Kidney Diseases.

---

## Disclaimer
*This machine learning software and associated risk models are developed strictly for pattern recognition research, academic investigation, and educational demonstration. They do not constitute formal medical advice, diagnostic determination, or treatment planning. Clinical decisions must always be guided by qualified healthcare professionals.*