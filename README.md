# Multiclass Classification Project

## Overview

Problem Statement: 
 Due to inconsistent classification of identical products across Otto Group's diverse global infrastructure, 
 this project aims to build a predictive model using 200,000+ products with 93 features to accurately distinguish main product categories and enable reliable performance analysis.

 The workflow covers data preprocessing, outlier analysis, feature reduction, model training, model comparison, hyperparameter tuning, and final evaluation.

The main objective was to build a reliable classification model while accounting for class imbalance and reducing unnecessary features.

---

## Dataset

 - Dataset Source : https://www.kaggle.com/c/otto-group-product-classification-challenge
 - The Data Consists of 93 columns & 200,000 rows .


The target variable contains **9 classes**:
- Class_1
- Class_2
- Class_3
- Class_4
- Class_5
- Class_6
- Class_7
- Class_8
- Class_9

### Class Distribution

| Class | Percentage |
|---|---:|
| Class_1 | 3.12% |
| Class_2 | 26.05% |
| Class_3 | 12.94% |
| Class_4 | 4.35% |
| Class_5 | 4.43% |
| Class_6 | 22.84% |
| Class_7 | 4.59% |
| Class_8 | 13.68% |
| Class_9 | 8.01% |

The dataset is moderately imbalanced. The largest class (`Class_2`) represents 26.05% of the data, while the smallest class (`Class_1`) represents 3.12%.

Because of this imbalance, **Macro F1** was used as the primary evaluation metric rather than relying only on accuracy.

---

## Preprocessing

### 1. Train / Validation 

The dataset was divided into training, validation

All preprocessing and feature-selection decisions were learned from the training data and then applied to validation/test data to avoid data leakage.

### 2. Outlier Analysis

Outliers were investigated using the **IQR method**.

The analysis showed that there were no significant outliers requiring removal or capping under the selected criterion. Therefore, no aggressive outlier removal was applied.

### 3. Scaling

Scaling was applied only to the linear baseline model.

A `StandardScaler` was fitted on the training data and then applied to the validation/test data.

Tree-based models were trained without scaling because algorithms such as Random Forest and XGBoost do not require standardized input features.

---

## Feature Reduction

### Correlation-Based Reduction

A correlation matrix was calculated using the training data.

A correlation threshold of **0.90** was used to identify highly correlated features.

In this dataset, the correlation filter did not remove any features. This indicates that there were no pairs of features with sufficiently high linear correlation under the selected threshold.

### Tree-Based Feature Selection

Since correlation analysis did not reduce the feature space, a **Random Forest-based feature selection** method was applied.

Feature importance was calculated using a Random Forest classifier, and `SelectFromModel` with a median importance threshold was used to retain the more informative features.

The selected feature set was then used consistently for the subsequent tree-based models.

---

## Models

Three tree-based classification models were evaluated:

1. **Random Forest**
2. **XGBoost**
3. **CatBoost**

A **Logistic Regression** model was also considered as a linear baseline, using:

- Correlation-based feature reduction
- StandardScaler
- Multiclass classification

For XGBoost, the target labels were converted to numerical labels using `LabelEncoder`. Random Forest and Logistic Regression were able to use the original class labels directly.

---

## Class Imbalance

The class distribution was examined before model training.

Because the imbalance was noticeable but not extreme, class weighting was considered for tree-based models. `class_weight="balanced"` was used in the Random Forest experiments, while XGBoost can handle class weighting through sample weights for multiclass classification.

The main comparison metric remained **Macro F1**, since it gives equal importance to all classes.

---

### Results Analysis

Because  XGBoost was the strongest initial model, this was selected for further hyperparameter tuning.

---

## Hyperparameter Tuning

Optuna was selected for hyperparameter optimization.

The optimization objective was:

**Maximize Macro F1**

This was chosen because the target contains nine imbalanced classes and Macro F1 provides a more balanced evaluation across minority and majority classes.

The main hyperparameters considered included:


### XGBoost

- `n_estimators`
- `max_depth`
- `learning_rate`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `gamma`
- `reg_alpha`
- `reg_lambda`

After tuning, the best parameters were used to retrain the selected models before final test evaluation.

---

## Evaluation Metrics

The following metrics were used:

### Accuracy

Measures the overall proportion of correctly classified samples.

### Macro F1

Calculates the F1 score independently for each class and then averages them equally.

This was the **primary metric** because of the class imbalance.

### Weighted F1

Calculates the F1 score for each class while weighting each class according to its number of samples.

### Classification Report

Precision, recall, and F1 score were also examined for each individual class.

---

## Final Evaluation Strategy

The test set was kept separate from feature selection and model development.

The final workflow was:

```text
Raw Data
   ↓
Train / Validation / Test Split
   ↓
Outlier Analysis
   ↓
Correlation Analysis
   ↓
Tree-Based Feature Selection
   ↓
Model Training
   ↓
Initial Model Comparison
   ↓
Random Forest & XGBoost Selection
   ↓
Optuna Hyperparameter Tuning
   ↓
Retraining with Best Parameters
   ↓
Final Test Evaluation
```

The same selected feature set was applied to the test data. No new feature-selection decisions were made using the test set.

---

## Conclusion

The experiments showed that **Random Forest was the strongest initial model**, achieving an Accuracy of approximately **80.15%** and a Macro F1 of approximately **75.17%**.

XGBoost performed similarly and remained a strong candidate for further optimization. CatBoost showed lower performance under the initial configuration.

The final model selection should be based primarily on **Macro F1**, with Accuracy and Weighted F1 used as complementary metrics.

The next stage of the project is to use the tuned model with the best validation performance and report its final performance on the unseen test set.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- XGBoost
- CatBoost
- Optuna
- Matplotlib

---

## Project Status

**Status:** Model development and hyperparameter tuning

**Task:** 9-class multiclass classification

**Primary metric:** Macro F1
