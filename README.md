# 🏢 Employee Attrition Intelligence AI

## Explainable Employee Attrition Prediction and Intelligent Retention Recommendation System Using 

---

## 📖 Project Overview

Employee attrition is a significant challenge for organizations as it leads to increased recruitment costs, loss of skilled employees, and reduced productivity. This project develops an Explainable Artificial Intelligence (XAI) based Employee Attrition Prediction System that predicts employee attrition risk and provides personalized retention recommendations for HR professionals.

The system combines machine learning, SHAP explainability, business impact analysis, and an interactive Streamlit dashboard to support data-driven HR decision-making.

---

## 🚀 Features

- Employee Attrition Prediction
- Multiple Machine Learning Models
- XGBoost Final Prediction Model
- SHAP Explainable AI
- Personalized Retention Recommendations
- Business Impact Analysis
- Interactive Streamlit Dashboard
- Real-time Prediction

---

## 🤖 Machine Learning Models

The following supervised learning algorithms were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost (Final Model)
- Stacked Ensembled Model

---

## 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 79.25% |
| Decision Tree | 78.57% |
| Random Forest | 84.01% |
| SVM | 81.63% |
| Gradient Boosting | 86.05% |
| **XGBoost** | **86.73%** |

XGBoost achieved the highest prediction accuracy and was selected as the final deployment model.

---

## 🔍 Explainable AI

The project integrates SHAP (SHapley Additive Explanations) to provide:

- Global Feature Importance
- Local Prediction Explanation
- SHAP Summary Plot
- SHAP Feature Importance Plot
- SHAP Waterfall Plot

---

## 💼 Dashboard Features

### Employee Profile
- Personal Information
- Department
- Job Role
- Education
- Experience

### Career Growth & Compensation
- Monthly Income
- Salary Hike
- Promotion
- Years at Company
- Years in Current Role

### Satisfaction & Engagement
- Job Satisfaction
- Environment Satisfaction
- Work-Life Balance
- Job Involvement
- OverTime
- Stock Option Level

### Prediction Output
- Attrition Prediction
- Attrition Probability
- Risk Level
- SHAP Explanation
- Personalized Retention Recommendations
- Business Impact Analysis

---

## 📂 Project Structure

```text
Employee-Attrition-Intelligence-AI/
│
├── Streamlit Dashboard/
│   └── app.py
│
├── Minor Project Final Code.ipynb
│
├── IBM HR Dataset.csv
│
├── xgb_attrition_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── shap_explainer.pkl
├── recommendation_mapping.pkl
│
├── catboost_info/
│
├── .ipynb_checkpoints/
│
└── README.md
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Plotly
- Matplotlib

---

## 📂 Dataset

**IBM HR Analytics Employee Attrition & Performance Dataset**

Source:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/drnimishadavis/Explainable-Employee-Attrition-Prediction-and-Intelligent-Retention-Recommendation-System.git
```

### Navigate to Project

```bash
cd Employee-Attrition-Intelligence-AI
```

### Run Application

```bash
streamlit run "Streamlit Dashboard/app.py"
```

---

## 📌 Dashboard Output

The Streamlit dashboard displays:

- Employee Attrition Prediction
- Prediction Probability
- Risk Level
- SHAP Explainability
- Feature Importance
- Personalized Retention Recommendations
- Business Impact Analysis
- Interactive Charts

---

## 🎓 Academic Information

**Project Title**

Explainable Employee Attrition Prediction and Intelligent Retention Recommendation System 

**Course**

Master of Computer Applications (Artificial Intelligence)

Amrita Vishwa Vidyapeetham

---

## 👩‍💻 Author

**Nimisha Davis**

GitHub: https://github.com/drnimishadavis

---

## 📜 License

This project is developed for academic and research purposes only.
