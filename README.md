# Telecom Customer Churn Prediction

A comprehensive end-to-end machine learning solution for predicting customer churn in a telecom setting. This repository contains all data preprocessing, feature engineering, model pipelines, hyperparameter tuning experiments, and a Streamlit application for interactive single-customer predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Science & Technical Approach](#science--technical-approach)
- [Feature Engineering](#feature-engineering)
- [Preprocessing & Pipelines](#preprocessing--pipelines)
- [Decision Tree & Random Forest](#decision-tree--random-forest)
- [Logistic Regression](#logistic-regression)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
## Project Overview
Telecom operators lose significant revenue when customers churn. This project builds a robust churn-prediction system that:
- Analyzes customer behavior, demographics, and billing history
- Engineers predictive features such as contract-remaining value and service-usage counts
- Trains multiple ML pipelines (Decision Tree, Random Forest, Logistic Regression)
- Optimizes them via `RandomizedSearchCV`
- Deploys a dark-themed Streamlit app for real-time, single-customer predictions

## Data
Input: Customer billing, demographic, and service-usage data
Target: Binary Churn flag (1 = churned, 0 = stayed)

External Mappings:

zip_cv_map.pkl: Cross-validated ZIP-to-churn map

global_zip_mean.pkl: Fallback churn rate

All custom transformers and preprocessing logic live in preprocessing.py.

## Exploratory Data Analysis (EDA)
Distribution Analysis
Heavy right-skew in billing features (Total Charges, Avg Monthly GB Download)

Bimodal tenure distribution distinguishing new vs. long-term customers

Correlation Study
Monthly Charge shows positive correlation with churn

Initial CLV proxy (monthly_charge × tenure) redundant with Total Charges

## Science & Technical Approach
Statistical Transformations
Apply log1p to reduce skew

Standard scaling to bring features to comparable variance

Predictive CLV
Compute "months remaining in current contract cycle" via:

```python
remaining = contract_length - (tenure % contract_length)
CLV = monthly_charge * remaining
```

Captures forward-looking customer value rather than pure historical spend.

Cross-Validated Target Encoding
Out-of-fold mean churn rate per ZIP using KFold, avoiding leakage

Stored as a dict plus global fallback for unseen codes


Rare grouping to reduce noise

Imbalance Correction
SMOTE oversampling vs. class_weight='balanced' to boost recall on churners

## Feature Engineering
Imputation
Offer → "No Offer"

Phone add-ons → "No Phone Service" + zero fill

Internet add-ons → "No Internet Service" + zero fill

Flags & Counts
HadRefunds, HadExtraDataCharges (binary)

AddOnCount: Count of "Yes" across all add-ons

StreamCount: Robust count of streaming services (fills missing columns)

Contract & Tenure
Bucket tenure into [0–6, 6–12, 12–24, 24–36, 36–60, 60+]

Ordinal encode Contract (Month-to-Month < One Year < Two Year)

Compute predictive CLV on remaining months

ZIP-Code Encoding
Cross-validated target mean encoding for robust ZIP signal

## Preprocessing & Pipelines
## Decision Tree & Random Forest
```python
ImbPipeline([
  ('impute_offer',     fill_offer_tf),
  ('impute_internet',  fill_internet_tf),
  ('add_addons',       add_addons_tf),
  ('tenure_eng',       tenure_eng_tf),
  ('calc_clv',         calc_clv_tf),
  ('refund_flag',      refund_flag_tf),
  ('extra_flag',       extra_flag_tf),
  ('stream_feats',     stream_feats_tf),
  ('zip_cv',           zip_cv_tf),
  ('drop_raw',         drop_raw_tf),
  ('encode',           ColumnTransformer([
      ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_feats),
      ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_feats),
      ('pass', 'passthrough', numeric_feats)
  ])),
  ('smote',            SMOTE(random_state=42)),
  ('clf',              DecisionTreeClassifier(random_state=42))
])
```

No numeric scaling

One-Hot + Ordinal encoding

SMOTE for imbalance

## Logistic Regression
```python
ImbPipeline([
  ...same initial steps...,
  ('encode_scale', ColumnTransformer([
      ('skewed', Pipeline([('log1p', log1p_tf), ('scale', StandardScaler())]), skewed_feats),
      ('num',    StandardScaler(), other_numeric_feats),
      ('ohe',    OneHotEncoder(handle_unknown='ignore'), ohe_feats),
      ('ord',    OrdinalEncoder(categories=ordinal_categories), ordinal_feats),
  ])),
  ('smote',           SMOTE(random_state=42)),       
  ('poly',            PolynomialFeatures(degree=d)),
  ('clf',             LogisticRegression(
                        solver='liblinear', 
                        penalty='l1' or 'l2', 
                        class_weight='balanced',
                        max_iter=5000))
])
```
Log-transform + scaling before polynomial expansion

PolynomialFeatures for interactions

Class weighting or SMOTE

## Hyperparameter Tuning
RandomizedSearchCV (30–40 trials, 5-fold CV)

Tree models: max_depth, min_samples_split, min_samples_leaf, max_features, (n_estimators for RF)

Logistic: C, penalty, solver, poly__degree

## Model Evaluation
used accuracy, precision, recall, f beta score -parameter = 2 to give recall a higher advantage- and ROC.

## Deployment
Saved Pipelines in models/ (.joblib)
Streamlit App (app.py):

Dark theme, sidebar model selector

Single-customer form, input validation, default injection for missing columns

Displays churn prediction & probability

```bash
streamlit run app.py
```
