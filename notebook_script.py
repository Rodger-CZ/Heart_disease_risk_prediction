# %% [markdown]
# # Predicting Heart Disease — End-to-End ML Pipeline
# 
# This notebook presents a complete machine learning workflow for the Kaggle Playground Series S6E2 competition.
# 
# ## Workflow
# 
# Data inspection and quality checks
# 
# Robust preprocessing using ColumnTransformer
# 
# Stratified cross-validation using ROC AUC
# 
# Baseline model: Logistic Regression
# 
# Improved model: HistGradientBoostingClassifier
# 
# Model ensembling (probability averaging)
# 
# Kaggle submission generation
# 
# ## Results
# 
# Logistic Regression CV AUC: ~0.950
# 
# HistGradientBoosting CV AUC: ~0.955
# 
# Public Leaderboard AUC: 0.95284
# 
# The close alignment between cross-validation and leaderboard scores indicates a stable and well-validated modeling approach.
# 
# This notebook demonstrates practical tabular machine learning skills, proper validation strategy, and competition workflow.

# %% [markdown]
# # Predicting Heart Disease (Kaggle Playground Series S6E2)
# 
# ## Project Overview
# This notebook builds a complete, reproducible machine learning pipeline for predicting
# the probability of heart disease using structured tabular data. The workflow covers:
# data inspection, cleaning, exploratory analysis, robust preprocessing, model training
# with cross-validation, and Kaggle submission generation.
# 
# ## Evaluation
# Submissions are evaluated using ROC AUC (higher is better).
# 
# ## Tools
# Python, pandas, numpy, matplotlib/seaborn, scikit-learn
# (Optionally: CatBoost/LightGBM)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:17:50.818419Z","iopub.execute_input":"2026-02-07T23:17:50.818747Z","iopub.status.idle":"2026-02-07T23:17:51.180290Z","shell.execute_reply.started":"2026-02-07T23:17:50.818721Z","shell.execute_reply":"2026-02-07T23:17:51.179318Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:17:51.181715Z","iopub.execute_input":"2026-02-07T23:17:51.182271Z","iopub.status.idle":"2026-02-07T23:17:52.772692Z","shell.execute_reply.started":"2026-02-07T23:17:51.182240Z","shell.execute_reply":"2026-02-07T23:17:52.771631Z"}}
# WE start by importing the necessary Packages and Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

plt.style.use("seaborn-v0_8")
RANDOM_STATE = 42

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:17:52.773966Z","iopub.execute_input":"2026-02-07T23:17:52.774512Z","iopub.status.idle":"2026-02-07T23:17:54.364397Z","shell.execute_reply.started":"2026-02-07T23:17:52.774470Z","shell.execute_reply":"2026-02-07T23:17:54.363634Z"}}
# We load the data into our notebook
train = pd.read_csv("/kaggle/input/playground-series-s6e2/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s6e2/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s6e2/sample_submission.csv")

# We inspect the number of the fields and records in each dataset 
train.shape, test.shape, sub.shape


# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:17:54.366312Z","iopub.execute_input":"2026-02-07T23:17:54.366572Z","iopub.status.idle":"2026-02-07T23:17:54.373111Z","shell.execute_reply.started":"2026-02-07T23:17:54.366550Z","shell.execute_reply":"2026-02-07T23:17:54.372318Z"}}
# Identifying the Target columns 
TARGET = list(set(train.columns) - set(test.columns))[0]
TARGET

# %% [markdown]
# Our Target columns is Heart Disease

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:17:54.374212Z","iopub.execute_input":"2026-02-07T23:17:54.374556Z","iopub.status.idle":"2026-02-07T23:17:54.493860Z","shell.execute_reply.started":"2026-02-07T23:17:54.374516Z","shell.execute_reply":"2026-02-07T23:17:54.493030Z"}}
X = train.drop(columns=[TARGET])
y = train[TARGET]

X.head(), y.value_counts(normalize=True)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:17:54.494847Z","iopub.execute_input":"2026-02-07T23:17:54.495489Z","iopub.status.idle":"2026-02-07T23:17:54.533181Z","shell.execute_reply.started":"2026-02-07T23:17:54.495462Z","shell.execute_reply":"2026-02-07T23:17:54.532097Z"}}
# Feature engineering and encoding
# We separate the numerical eatures from the categorical features
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:17:54.534264Z","iopub.execute_input":"2026-02-07T23:17:54.535247Z","iopub.status.idle":"2026-02-07T23:18:10.792066Z","shell.execute_reply.started":"2026-02-07T23:17:54.535218Z","shell.execute_reply":"2026-02-07T23:18:10.791214Z"}}
# We proceed to logistic regression to generate the model
# We will transform and fit our model in this cell
model = LogisticRegression(max_iter=5000, solver="lbfgs")

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
auc_scores = []

for tr_idx, va_idx in skf.split(X, y):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_va)[:, 1]
    auc_scores.append(roc_auc_score(y_va, proba))

print("Baseline CV AUC mean:", np.mean(auc_scores))
print("Baseline CV AUC std :", np.std(auc_scores))

# %% [markdown]
# Note: Logistic Regression was fitted with standardized numeric features to improve
# optimization stability and avoid convergence warnings when combining scaled numeric
# features with one-hot encoded categorical features.

# %% [markdown]
# ## 4. Improved Model: Gradient Boosting
# 
# To capture non-linear relationships between features, a gradient boosting
# classifier was trained and evaluated using the same stratified cross-validation
# strategy.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:18:10.793185Z","iopub.execute_input":"2026-02-07T23:18:10.793587Z","iopub.status.idle":"2026-02-07T23:18:11.075473Z","shell.execute_reply.started":"2026-02-07T23:18:10.793543Z","shell.execute_reply":"2026-02-07T23:18:11.074551Z"}}
from sklearn.ensemble import HistGradientBoostingClassifier

hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=8,
    max_iter=500,
    random_state=RANDOM_STATE
)

hgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", hgb_model)
])

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:18:11.076760Z","iopub.execute_input":"2026-02-07T23:18:11.077708Z","iopub.status.idle":"2026-02-07T23:19:48.688862Z","shell.execute_reply.started":"2026-02-07T23:18:11.077609Z","shell.execute_reply":"2026-02-07T23:19:48.687894Z"}}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
auc_scores_hgb = []

for tr_idx, va_idx in skf.split(X, y):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    hgb_pipeline.fit(X_tr, y_tr)
    proba = hgb_pipeline.predict_proba(X_va)[:, 1]
    auc_scores_hgb.append(roc_auc_score(y_va, proba))

print("HGB CV AUC mean:", np.mean(auc_scores_hgb))
print("HGB CV AUC std :", np.std(auc_scores_hgb))


# %% [markdown]
# ### Model Comparison
# 
# | Model | CV AUC |
# |------|-------|
# | Logistic Regression (Baseline) | ~0.9505 |
# | HistGradientBoostingClassifier | ~0.9550 |
# 
# The gradient boosting model provides improved performance by modeling
# non-linear feature interactions while maintaining stable cross-validation
# results.

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:19:48.691184Z","iopub.execute_input":"2026-02-07T23:19:48.691558Z","iopub.status.idle":"2026-02-07T23:20:13.979834Z","shell.execute_reply.started":"2026-02-07T23:19:48.691531Z","shell.execute_reply":"2026-02-07T23:20:13.979086Z"}}
hgb_pipeline.fit(X, y)

test_proba = hgb_pipeline.predict_proba(test)[:, 1]

submission = sub.copy()
submission[TARGET] = test_proba
submission.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:20:13.981126Z","iopub.execute_input":"2026-02-07T23:20:13.981855Z","iopub.status.idle":"2026-02-07T23:20:14.593351Z","shell.execute_reply.started":"2026-02-07T23:20:13.981828Z","shell.execute_reply":"2026-02-07T23:20:14.592504Z"}}
submission.to_csv("submission.csv", index=False)
print("submission.csv saved with shape:", submission.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:20:14.594396Z","iopub.execute_input":"2026-02-07T23:20:14.594643Z","iopub.status.idle":"2026-02-07T23:20:14.600461Z","shell.execute_reply.started":"2026-02-07T23:20:14.594622Z","shell.execute_reply":"2026-02-07T23:20:14.599384Z"}}
assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == list(sub.columns)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:20:14.602006Z","iopub.execute_input":"2026-02-07T23:20:14.602382Z","iopub.status.idle":"2026-02-07T23:20:43.605816Z","shell.execute_reply.started":"2026-02-07T23:20:14.602357Z","shell.execute_reply":"2026-02-07T23:20:43.604642Z"}}
# Fit baseline
pipeline.fit(X, y)
lr_test_proba = pipeline.predict_proba(test)[:, 1]

# Fit HGB
hgb_pipeline.fit(X, y)
hgb_test_proba = hgb_pipeline.predict_proba(test)[:, 1]


# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:20:43.607164Z","iopub.execute_input":"2026-02-07T23:20:43.607540Z","iopub.status.idle":"2026-02-07T23:20:43.612387Z","shell.execute_reply.started":"2026-02-07T23:20:43.607512Z","shell.execute_reply":"2026-02-07T23:20:43.611452Z"}}
ensemble_proba = (lr_test_proba + hgb_test_proba) / 2

# %% [code] {"execution":{"iopub.status.busy":"2026-02-07T23:20:43.613725Z","iopub.execute_input":"2026-02-07T23:20:43.614488Z","iopub.status.idle":"2026-02-07T23:20:44.227090Z","shell.execute_reply.started":"2026-02-07T23:20:43.614460Z","shell.execute_reply":"2026-02-07T23:20:44.226124Z"}}
submission_ens = sub.copy()
submission_ens[TARGET] = ensemble_proba

submission_ens.to_csv("submission_ensemble.csv", index=False)
print("submission_ensemble.csv saved")

# %% [markdown]
# ### Model Ensembling
# 
# To improve prediction stability, probabilities from Logistic Regression and
# HistGradientBoosting models were averaged. Ensembling combines linear and
# non-linear model strengths and often improves generalization performance.
