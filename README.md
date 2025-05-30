## Breast Cancer Prediction using Logistic Regression

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset** to build a **binary classification model** that predicts whether a tumor is malignant or benign using **Logistic Regression**.

---

### Dataset

* Source: [Kaggle - UCI Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* Features: 30 numeric features (mean, standard error, worst values of radius, texture, area, etc.)
* Target:

  * `M` (Malignant) → 0
  * `B` (Benign) → 1

---

### Objectives

* Build a binary classifier using Logistic Regression.
* Evaluate performance using accuracy, precision, recall, F1, and ROC-AUC.
* Demonstrate the impact of decision thresholds.
* Visualize the ROC curve.

---

### Tools & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

### Evaluation Metrics

* **Confusion Matrix**
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**
* **ROC Curve Visualization**
* **Threshold tuning example (e.g., 0.3)**

---

### 📈 Logistic Regression & Sigmoid Function

The model uses the **sigmoid function** to output probabilities:

```
σ(z) = 1 / (1 + e^(-z))
```

By default, a threshold of `0.5` is used for classification. You can manually adjust the threshold to observe how **precision** and **recall** trade off.
