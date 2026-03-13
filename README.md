# Credit Risk Model

## Project Overview
This project develops and compares several machine learning models for credit risk prediction, focusing on interpretability for tree-based models. The goal is to identify the best-performing model and extract key insights for effective credit risk management. The models evaluated include Logistic Regression, Random Forest, LightGBM, and XGBoost.

## Data Generation
The dataset is synthetically generated to simulate 500,000 customer records with various financial and demographic attributes such as age, income, education level, credit limit, utilization, past late payments, debt-to-income ratio, and home ownership. A `default` target variable is engineered based on a calculated `risk_score` to create an imbalanced dataset, mimicking real-world credit risk scenarios.


## Models
Four distinct machine learning models are trained and evaluated on the prepared dataset:
1.  **Logistic Regression**: A baseline linear model, trained with `penalty='l2'`, `C=0.1`, and `class_weight='balanced'` to handle class imbalance.
2.  **Random Forest Classifier**: An ensemble tree-based model (`n_estimators=100`, `max_depth=10`, `n_jobs=-1`).
3.  **LightGBM Classifier**: A gradient boosting framework that uses tree-based learning algorithms (`objective='binary'`, `metric='auc'`, `learning_rate=0.05`, `feature_fraction=0.8`, `is_unbalance=True`, `num_boost_round=200`).
4.  **XGBoost Classifier**: Another powerful gradient boosting model (`objective='binary:logistic'`, `eval_metric='logloss'`, `n_estimators=200`, `learning_rate=0.05`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=0.1`, `scale_pos_weight` adjusted for imbalance).

## Model Evaluation Metrics
Each model's performance is assessed using a comprehensive set of metrics crucial for imbalanced classification problems:
- **Gini Coefficient**: Derived from AUC, measures model discrimination.
- **AUC (Area Under the Receiver Operating Characteristic Curve)**: Measures the ability of the model to distinguish between positive and negative classes.
- **PR_AUC (Area Under the Precision-Recall Curve)**: Particularly informative for imbalanced datasets, as it focuses on the positive class prediction accuracy.
- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics.
- **KS Statistic (Kolmogorov-Smirnov)**: Measures the maximum difference between the cumulative true positive and cumulative false positive rates, indicating separation between good and bad customers.

### Model Results Summary
The `report` dataframe provides a clear comparison of all models across the defined metrics. For example:

```
                       Gini     AUC  PR_AUC      F1      KS
Logistic Regression  0.9475  0.9737  0.3802  0.1860  0.8439
Random Forest        0.9191  0.9596  0.2700  0.0187  0.8137
LightGBM             0.9216  0.9608  0.2599  0.3206  0.8234
XGBoost              0.9189  0.9595  0.2646  0.2961  0.8080
```

### Visualizations for Interpretability and Comparison

1.  **Precision-Recall Curves**
![Risk Profile Plot](P1.png)
2.  **SHAP (SHapley Additive exPlanations) Summary Plots**

3.  **ROC AUC Curves**

4.  **3D Prediction Surfaces**

