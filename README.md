# Fraud Detection using Machine Learning and Deep Learning

This repository contains a complete workflow for analysing fairness in detecting fraudulent bank transactions using Machine Learning (ML) and Deep Learning (DL). It includes exploratory data analysis, preprocessing, class imbalance handling, model training, evaluation, and saved trained models.

---

## 📁 Repository Structure

```text
├── base-models.ipynb              # EDA, feature engineering, ML baseline models
├── ml_soln_fraud_det.ipynb        # Classical ML model training and evaluation
├── dl_soln_fraud_dect.ipynb       # Deep learning model training
├── trained_models/                # Saved trained models (.pkl, .h5, .pth)
├── dataset/                       # Training/testing dataset
├── requirements.txt               # Dependencies
└── README.md
```

---

## 📌 1. Project Description

The goal of this project is to analyze fairness in fraud detection models capable of identifying rare fraudulent transactions in highly imbalanced datasets.

This project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning & preprocessing (one-hot encoding, scaling)
- Feature engineering & selection
- Oversampling and undersampling techniques for imbalance handling
- Training Classical ML models
- Training Neural Network models
- Comparing performance using Recall, AUC, Precision, F1-score
- Saving trained models for inference

---

## ⚙️ 2. Steps to Run the Code

### **Step 1 — Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 2 — Launch Jupyter Notebook**

```bash
jupyter notebook
```

### **Step 3 — Open and Run the Notebooks**

1. `base-models.ipynb`
2. `ml_soln_fraud_det.ipynb`
3. `dl_soln_fraud_dect.ipynb`

### **Step 4 — Ensure Dataset is Available**

Place your dataset inside:

```text
dataset/
```
Download the dataset from:

[Bank Account Fraud Dataset (NeurIPS 2022) on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

Most notebooks load data using:


```python
pd.read_csv("dataset/data.csv")
```

### **Step 5 — Trained Models**

Saved models are stored in:

```text
trained_models/
```

---

## 📦 3. Required Python Libraries

All dependencies are listed in `requirements.txt`.

Key libraries include:

- Python ≥ 3.8  
- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- xgboost  
- lightgbm  
- imbalanced-learn  
- tensorflow (entire list is in `requirements.txt`)

---

## 📥 4. Input Format

The input dataset must be in CSV format containing:

- Numerical features  
- Categorical features  
- Target column:  

```
fraud  (0 = non-fraud, 1 = fraud)
```

**Example row:**

| feature_1 | feature_2 | feature_3 | fraud |
|-----------|-----------|-----------|--------|
| 0.51      | 22        | DE        | 0      |
| 0.13      | 45        | US        | 1      |

(there are 31 input features)

---

## 📤 5. Output Format

Models generate:

### **Binary Predictions**
```python
model.predict(X_test)
```

### **Evaluation Metrics**
- Accuracy  
- Precision  
- Recall (important for fraud detection)  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

### **Saved Model Formats**
- `.pkl` (scikit-learn models)  
- `.joblib`  (other ML models)  
- `.pt` (PyTorch models)

---

## 🚀 6. Example Inference Code

### **Load a Saved ML Model**
```python
import joblib
model = joblib.load("trained_models/random_forest.pkl")
pred = model.predict(X_test)
```

### **Load a Saved Deep Learning Model**
```python
import torch

model = torch.load("trained_models/model.pt", map_location="cpu")
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
```

---

## ⭐ 7. Features

- Complete fairness Analysis of fraud detection models  
- Extensive EDA and feature engineering  
- Multiple ML and DL approaches  
- Class imbalance solutions (SMOTE, undersampling)  
- Model performance and fairness comparison  
- Reusable trained models  
- Well-documented notebooks  

---

## 📄 License

This project is intended for academic and research use.

