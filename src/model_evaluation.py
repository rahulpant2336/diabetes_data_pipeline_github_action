from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import joblib
import pickle
from sklearn.model_selection import KFold
import json

#determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent
print(base_dir)

model_X_test_data_path = base_dir / "data/testing_data" / "X_test.csv"
model_y_test_data_path = base_dir / "data/testing_data" / "y_test.csv"

X_test = pd.read_csv(model_X_test_data_path)
y_test = pd.read_csv(model_y_test_data_path)

model_X_train_data_path = base_dir / "data/testing_data" / "X_train.csv"
model_y_train_data_path = base_dir / "data/testing_data" / "y_train.csv"

X_train = pd.read_csv(model_X_train_data_path)
y_train = pd.read_csv(model_y_train_data_path)

y_train = y_train.values.ravel()

params_path = base_dir / "params.yaml"

#load parameters from params.yaml
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["model_evaluation"]

# Load the xgb model
with open(base_dir / "models/xgb_model.pkl", "rb") as f:
    xgb_loaded_model = joblib.load(f)

# Load the lgbm model
with open(base_dir / "models/lgbm_model.pkl", "rb") as f:
    lgbm_loaded_model = joblib.load(f)

# Load the rf model
with open(base_dir / "models/rf_model.pkl", "rb") as f:
    rf_loaded_model = joblib.load(f)

models = []

models.append(('RF', rf_loaded_model))
models.append(('XGB', xgb_loaded_model))
models.append(("LightGBM", lgbm_loaded_model))

print(models)

# evaluate each model in turn
results = []
names = []
output_json = {}

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)

    # Save into JSON-friendly dict
    output_json[name] = {
        "mean_accuracy": cv_results.mean(),
        "std_accuracy": cv_results.std()
    }

# Write to file
results_dir = base_dir / "results" 
results_dir.mkdir(parents=True, exist_ok=True)   # create folder if missing

with open(results_dir / "model_results.json", "w") as f:
    json.dump(output_json, f, indent=4)

print(f"Results saved to {results_dir / 'model_results.json'} file")

# After choosing the best model, evaluate once on X_test, y_test:
# Find best model by mean accuracy
best_name, best_model = max(output_json.items(), key=lambda x: x[1]["mean_accuracy"])
print(f"Best model from CV: {best_name}")

# Retrain on full training set
best_model_obj = [m for n, m in models if n == best_name][0]
best_model_obj.fit(X_train, y_train)

# Evaluate on test set
print("Final test accuracy:", best_model_obj.score(X_test, y_test))
