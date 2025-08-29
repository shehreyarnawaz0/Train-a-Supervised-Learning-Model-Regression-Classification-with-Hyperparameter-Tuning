# 🔥 Supervised Learning with Hyperparameter Tuning (SaaS Churn Data)

## 📌 Project Overview
This project demonstrates how to train and optimize a **supervised learning model** (both **classification** and **regression**) using the SaaS Churn dataset.  

The main focus is on:
- Dataset preparation  
- Model training  
- Hyperparameter tuning with GridSearchCV  
- Model evaluation with appropriate metrics  
- Saving the trained model for later use  

---

## 🎯 Objective
- Build a supervised learning pipeline.  
- Optimize the model using **GridSearchCV**.  
- Compare **baseline** vs. **tuned model** performance.  
- Evaluate with metrics and visualizations.  

---

## 📂 Dataset
- File: `/content/drive/MyDrive/saas_churn_data.csv`  
- **Target Variable**: `churn` (binary classification → Yes/No or 1/0).  
- Features: Various customer and subscription-related columns.  

> ⚠️ If your dataset uses a different target column name, update the variable `target_col` in the code.

---

## 🛠️ Models Used
### Classification
- **Random Forest Classifier**  
- Evaluation Metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

### Regression (Optional)
- **Decision Tree Regressor**  
- Evaluation Metrics:
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - R² Score  

---

## ⚡ Hyperparameter Tuning
Performed using **GridSearchCV** with 5-fold cross-validation.  

### Random Forest Classifier Grid:
- `n_estimators`: [50, 100, 200]  
- `max_depth`: [None, 5, 10]  
- `min_samples_split`: [2, 5, 10]  

### Decision Tree Regressor Grid:
- `max_depth`: [None, 5, 10, 20]  
- `min_samples_split`: [2, 5, 10]  
- `min_samples_leaf`: [1, 2, 5]  

---

## 📊 Results
- **Baseline Model**: Trained with default parameters.  
- **Tuned Model**: Best parameters selected via GridSearchCV.  
- Compared performance before & after tuning.  

Example (for classification):
Baseline Accuracy: 0.82
Tuned Accuracy: 0.88


---

## 🚀 How to Run (Google Colab)
1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Update dataset path in code:


file_path = "/content/drive/MyDrive/saas_churn_data.csv"
Run all cells.

Choose task type:


task_type = "classification"   # or "regression"
Evaluate performance and visualize results.

💾 Model Saving
Final tuned model is saved as:

classification_model.pkl (for classification)

regression_model.pkl (for regression)

Use joblib.load("classification_model.pkl") to load it later.

📁 Deliverables
Python/Colab Notebook (.ipynb)

Trained model file (.pkl)

README.md (this file)

requirements.txt

🔮 Bonus Features
Confusion matrix heatmap for classification

Scatter plot of predicted vs actual for regression

Easy switching between classification and regression with task_type variable

