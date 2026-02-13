# Machine Learning Classification Project

## a. Problem Statement

The objective of this project is to classify whether a tumor is 
Malignant (cancerous) or Benign (non-cancerous) using multiple 
Machine Learning classification algorithms.

The goal is to compare the performance of six different classification 
models using various evaluation metrics and determine the best-performing model.

---

## b. Dataset Description

Dataset Name: Breast Cancer Wisconsin Dataset  
Source: sklearn built-in dataset  

Number of Instances: 569  
Number of Features: 30 numerical features  
Type: Binary Classification  

Target Classes:
0 → Malignant  
1 → Benign  

The dataset contains features computed from digitized images of 
fine needle aspirate (FNA) of breast mass tissue.

Feature categories include:
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- And several other statistical measurements

This dataset satisfies assignment requirements:
✔ Minimum 12 features (30 available)  
✔ Minimum 500 instances (569 available)

---

## c. Models Used

The following 6 Machine Learning models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

---

## Evaluation Metrics Used

For each model, the following metrics were calculated:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name        | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression  |          |     |           |        |    |     |
| Decision Tree        |          |     |           |        |    |     |
| KNN                  |          |     |           |        |    |     |
| Naive Bayes          |          |     |           |        |    |     |
| Random Forest        |          |     |           |        |    |     |
| XGBoost              |          |     |           |        |    |     |


---

## Observations

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Performs very well due to clear linear separation in dataset. |
| Decision Tree        | Easy to interpret but may overfit the data. |
| KNN                  | Performs well after scaling but is computationally expensive. |
| Naive Bayes          | Fast model but assumes feature independence. |
| Random Forest        | High performance due to ensemble learning and reduced overfitting. |
| XGBoost              | Generally achieves the best performance due to boosting technique. |

---

