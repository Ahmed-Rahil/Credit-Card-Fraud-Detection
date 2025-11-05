# Credit Card Fraud Detection: A Comparative Methodological Analysis

## 1. Executive Summary

This project presents a robust machine learning model for detecting fraudulent credit card transactions from a dataset with a severe class imbalance (0.17% fraud cases). The primary objective is to optimize for **Recall** (maximizing fraud detection) and **Precision** (minimizing false positives), as standard `Accuracy` is a misleading metric in this context.

A comparative analysis was conducted between a baseline `RandomForestClassifier` and an advanced `imblearn` pipeline utilizing `SMOTE` (Synthetic Minority Over-sampling Technique). The findings conclusively demonstrate that the baseline model, despite its simplicity, is superior in performance, achieving a higher F1-Score and a better Precision-Recall (P-R) Curve. This project highlights the non-necessity of complex rebalancing techniques when a robust ensemble model is already in use.

## 2. Technologies Used

- **Python 3**
- **Pandas:** For data manipulation and loading.
- **Scikit-learn:** For machine learning (RandomForest, StandardScaler, train_test_split, metrics).
- **Imbalanced-learn:** For the SMOTE pipeline and advanced imbalance handling.
- **Matplotlib & Seaborn:** For data visualization and plotting.
- **Google Colab:** As the development environment.

## 3. Dataset

- **Source:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (0.172%)
- **Features:**
  - `V1-V28`: 28 anonymized, PCA-transformed features.
  - `Time` & `Amount`: Raw transactional data; require feature scaling.
  - `Class`: Target variable (1 for Fraud, 0 for Normal).

## 4. Methodology

A scientific comparison was conducted between two distinct approaches. All data was first preprocessed by applying `StandardScaler` to the `Time` and `Amount` columns. The data was then split into training and testing sets using `stratify=y` to ensure the class imbalance was preserved.

### Model 1: Baseline (Control Group)

- **Algorithm:** `RandomForestClassifier`
- **Method:** Trained directly on the original, imbalanced training data. This acts as our control to measure the out-of-the-box performance of a robust ensemble method.

### Model 2: Tuned Pipeline (Experimental Group)

- **Algorithm:** `RandomForestClassifier` combined with `SMOTE`.
- **Method:** An `imblearn.pipeline.Pipeline` was constructed to prevent data leakage during cross-validation.
  1.  **SMOTE:** This step oversamples the minority class _only_ on the training portion of each CV fold.
  2.  **GridSearchCV:** The pipeline was optimized for the highest `f1-score`, tuning key hyperparameters.

The hypothesis was that Model 2 would significantly outperform Model 1 by correcting the class imbalance.

## 5. Results & Analysis

Contrary to the initial hypothesis, the **Baseline Model (Model 1) demonstrated superior and more reliable performance.** The evidence from the unseen test set is as follows.

### Evidence 1: Confusion Matrices

The confusion matrix shows the model's performance in absolute numbers. The key objective is to maximize the "Caught Fraud" (True Positive) and minimize the "False Fraud" (False Positive) cells.

| Metric                             | Model 1 (Baseline RF) | Model 2 (SMOTE + Tuned RF) |    Winner    |
| :--------------------------------- | :-------------------: | :------------------------: | :----------: |
| **Fraud Caught (Recall)**          | **80 / 98** (81.63%)  |      79 / 98 (80.61%)      | **Baseline** |
| **False Alarms (False Positives)** |         **5**         |             26             | **Baseline** |

The baseline model **caught more fraud** while **making over 5 times _fewer_ false alarms** than the "improved" SMOTE model.

**Model 1: Baseline Random Forest**
![Baseline Model Confusion Matrix](icon\ConfusionMatrix_RF.png)

**Model 2: Tuned Pipeline (SMOTE + RF)**
![Tuned Model Confusion Matrix](icon\ConfusionMatrix_finetuned.png)

### Evidence 2: Precision-Recall Curve

The P-R Curve provides a holistic summary of performance across all thresholds. A higher "Area Under the Curve" (AUC-PR) is better.

- **Baseline RF (AUC-PR): 0.8647**
- **Tuned Pipeline (AUC-PR): 0.8591**

The P-R curve comparison plot confirms this finding. The blue dashed line (Baseline) is slightly but consistently above the orange line (Tuned Pipeline), resulting in a higher overall AUC-PR score.

![Precision-Recall Curve Comparison](icon\Precision_Recall.png)

### Analysis of Findings

This project provides a critical insight: **algorithmic complexity does not guarantee superior performance.** The analysis suggests two primary reasons for this outcome:

1.  **Ensemble Robustness:** `RandomForestClassifier`, as an ensemble model, is naturally robust to class imbalance. By bootstrapping, some trees in the forest become "experts" at finding the rare fraud cases, and their combined vote is highly effective without artificial rebalancing.
2.  **SMOTE-Induced Noise:** The synthetic data created by `SMOTE` was likely unrepresentative of _real_ fraud. This introduced noise and caused the Tuned Pipeline to **overfit** to these unrealistic synthetic patterns, thereby _worsening_ its ability to generalize and find genuine fraud in the unseen test set.

## 6. Conclusion

The final selected model is the **Baseline `RandomForestClassifier`**.

This project successfully developed a high-performance fraud detection model (81.6% Recall with very high Precision). More importantly, it demonstrated the critical value of a rigorous, data-driven methodology. By validating a complex balancing technique against a simpler, well-chosen baseline, we proved _through evidence_ that the simpler model was more effective, more efficient, and more reliable.

## 7. How to Run

1.  Click the "Open in Colab" badge at the top of this README.
2.  Ensure your `creditcard.csv` file is accessible in your Google Drive.
3.  Update the `file_path` variable in **Cell 4** of the notebook to point to the correct location of your CSV file.
4.  Run all cells sequentially.
