# **Propensity Analysis – Insurance Policy Renewal Prediction**

## **📌 Project Overview**

This project focuses on predicting the **likelihood of a customer renewing their insurance policy** using machine learning. The goal is to enable **targeted marketing campaigns**, improve **customer retention rates**, and optimize resource allocation.

Key Highlights:

* Built and compared multiple machine learning models.
* Implemented an **ensemble Voting Classifier** (XGBoost, Random Forest, Logistic Regression).
* Applied **explainable AI techniques** (LIME) to interpret predictions.
* Evaluated performance using **accuracy, precision, recall, F1-score, and confusion matrix**.

---

## **🚀 Features**

* **End-to-end ML pipeline** – data preprocessing, feature engineering, model training, and evaluation.
* **Ensemble model (Voting Classifier)** with cross-validation for robust performance.
* **Explainability** via **LIME** for local prediction insights.
* **Optimized metrics** (Mean F1-score ≈ **0.885** across folds).
* Business-oriented insights for **targeted policy renewal strategies**.

---

## **📂 Project Structure**

```
├── notebooks/
│   └── projectimplementation.ipynb   # Main notebook
├── data/
│   └── [Your dataset here]
├── models/
│   └── final_voting_classifier.pkl   # Saved model
├── README.md                         # Project description
├── requirements.txt                  # Python dependencies
```

---

## **⚙️ Tech Stack**

* **Languages:** Python (3.11+)
* **Libraries:** pandas, numpy, scikit-learn, xgboost, lime, matplotlib, seaborn
* **Tools:** Jupyter Notebook, GitHub
* **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)

---

## **📊 Model Performance**

* **Cross-validated F1 Score:** \~0.885
* **Best Features Identified:** IDV, Vehicle Age, Location, Policy Tenure
* Confusion Matrix and Precision-Recall plots confirm strong generalization.

---

## **📌 Key Insights**

* Customers with **higher IDV** and **longer tenure** have higher renewal probabilities.
* LIME visualizations reveal how specific features influence individual predictions.
* Ensemble learning provides **improved accuracy and stability** compared to single models.

---

## **🚀 How to Run**

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>.git
   cd <repo-folder>
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**

   ```bash
   jupyter notebook notebooks/projectimplementation.ipynb
   ```

4. **(Optional) Load Saved Model:**

   ```python
   import joblib
   model = joblib.load('models/final_voting_classifier.pkl')
   ```

---

## **📌 Future Work**

* Add **SHAP explainability** for global feature importance.
* Deploy the model with **Flask/Streamlit** for real-time predictions.
* Integrate with a **business dashboard (Power BI/Looker)**.

---

## **🧑‍💻 Author**

**Aarush Gupta**
*Third-year CSE (Data Science) student at VIT Vellore*
