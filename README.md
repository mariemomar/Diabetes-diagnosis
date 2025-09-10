# 🩺 Diabetes Diagnosis using Machine Learning

This project predicts whether a person has diabetes based on medical features (e.g., glucose, BMI, age).  
It uses **multiple machine learning models** (Logistic Regression, SVM with Linear Kernel, and SVM with RBF Kernel), and applies a **majority voting** system to decide the final outcome.  

---

## 🚀 Features
- Preprocesses the dataset with **StandardScaler** for normalization.  
- Trains three ML models:  
  - Logistic Regression  
  - Support Vector Machine (Linear Kernel)  
  - Support Vector Machine (RBF Kernel)  
- Saves models and scaler with **joblib** for reuse.  
- Accepts user input for prediction.  
- Applies **majority voting** to combine model outputs.  

---

## 📂 Project Structure
```
│── main.py # Main script (user input + prediction)
│── classifiers.ipynb (data loading & models training)
│── diabetes.csv (dataset)
│── logistic_regression_model.pkl
│── svm_model.pkl
│── svm_model_rbf.pkl
│── scaler.pkl
│── README.md