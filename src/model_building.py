from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import yaml
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import RobustScaler

import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Allow local file store for MLflow tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent
print(base_dir)

fe_data_path = base_dir / "data/feature_engineering" / "feature_engineering_data.csv"
params_path = base_dir / "params.yaml"

# load parameters from params.yaml
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["model_building"]

# Extract model hyperparameters
lgbm_params = params["lgbm"]
rf_params = params["random_forest"]
xgb_params = params["xgb"]

# load feature engineered dataset
df = pd.read_csv(fe_data_path)
print("Feature engineered Data loaded...")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# separate features X and binary target y
feature_cols = list(X.columns)
target_col = y.name

# train test split
data_test_size = params["train_test_split"]["test_size"]
data_random_state = params["train_test_split"]["random_state"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_test_size, random_state=data_random_state)

# output testing directory path
testing_dir = base_dir / "data" / "testing_data"
testing_dir.mkdir(parents=True, exist_ok=True)

# save model testing data
X_test.to_csv(testing_dir / "X_test.csv", index=False)
y_test.to_csv(testing_dir / "y_test.csv", index=False)
X_train.to_csv(testing_dir / "X_train.csv", index=False)
y_train.to_csv(testing_dir / "y_train.csv", index=False)

print(f"Model testing data saved! Saved data to: {testing_dir}")

# output model directory path
models_dir = base_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# === Save scaler and feature columns ===
numeric_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
scaler = RobustScaler().fit(X_train[numeric_cols])
joblib.dump(scaler, models_dir / "scaler.pkl")
joblib.dump(feature_cols, models_dir / "feature_cols.pkl")
print("Scaler and feature columns saved!")

# Models dictionary
models = {
    "Light GBM": (
        LGBMClassifier(
            learning_rate=lgbm_params["learning_rate"],
            max_depth=lgbm_params["max_depth"],
            n_estimators=lgbm_params["n_estimators"],
            random_state=lgbm_params["random_state"],
            verbose=-1
        ),
        lgbm_params
    ),
    "Random Forest": (
        RandomForestClassifier(
            max_depth=rf_params["max_depth"],
            max_features=rf_params["max_features"],
            min_samples_split=rf_params["min_samples_split"],
            n_estimators=rf_params["n_estimators"],
            random_state=rf_params["random_state"]
        ),
        rf_params
    ),
    "XGBoost": (
        XGBClassifier(
            learning_rate=xgb_params["learning_rate"],
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            random_state=xgb_params["random_state"]
        ),
        xgb_params
    )
}

# configure MLflow tracking URI (SQLite Database) and experiment
mlflow_db_path = base_dir / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.resolve().as_posix()}")
mlflow.set_experiment("Diabetes Predictions")

metrics_summary = []

print("Starting Model Building & Training with MLflow tracking")
for model_name, (model, model_hparams) in models.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model_hparams)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("target_column", target_col)

        # Fit model
        model_tuned = model.fit(X_train, y_train)

        # Calculate Metrics
        cvs = cross_val_score(model_tuned, X_train, y_train, cv=5, scoring="accuracy").mean()

        # Log evaluation metrics to MLflow
        mlflow.log_metric("cross val score", cvs)

        metrics_summary.append({
            "Model": model_name,
            "Cross Val Score": cvs
        })

        # Save model binary locally
        model_filename = model_name.lower().replace(" ", "_") + ".pkl"
        model_filepath = models_dir / model_filename
        with open(model_filepath, "wb") as f:
            joblib.dump(model, f)

        # Log artifacts (no artifact_path to avoid /C: issue)
        mlflow.log_artifact(model_filepath.resolve().as_posix())
        mlflow.log_artifact((models_dir / "scaler.pkl").resolve().as_posix())
        mlflow.log_artifact((models_dir / "feature_cols.pkl").resolve().as_posix())

        # Flavor-based logging
        if "Light GBM" in model_name:
            mlflow.lightgbm.log_model(lgb_model=model, name="LightGBM_Model")
        elif "Random Forest" in model_name:
            mlflow.sklearn.log_model(sk_model=model, name="RandomForest_Model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        elif "XGBoost" in model_name:
            model_info = mlflow.xgboost.log_model(xgb_model=model, name="XGBoost_Model", registered_model_name="XGBoost_Model")
            mlflow.set_tag("stage", "production")
            mlflow.set_tag("registered_model_name", "XGBoost_Model")
            client = MlflowClient()
            if model_info.registered_model_version:
                version = str(model_info.registered_model_version)
                client.set_registered_model_tag("XGBoost_Model", "stage", "production")
                client.set_model_version_tag("XGBoost_Model", version, "stage", "production")
                client.set_model_version_tag("XGBoost_Model", version, "status", "production")
                client.set_registered_model_tag("XGBoost_Model", "production", version)
        else:
            mlflow.sklearn.log_model(model, name="model")

        print(f"{model_name} Metrics logged to MLflow")
        print(f"Cross val score {cvs}")

# summary dataframe
summary_df = pd.DataFrame(metrics_summary)
print("\n == Final Model Comparison Summary ==")
print(summary_df.to_string(index=False))
