from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="Diabetes Prediction API")

# Input schema (match raw dataset columns)
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Locate project base directory
base_dir = Path(__file__).resolve().parent.parent

# Load trained best model
best_model_path = base_dir / "models" / "xgboost.pkl"
best_model = joblib.load(best_model_path)

# Load fitted scaler and feature columns from training
scaler = joblib.load(base_dir / "models" / "scaler.pkl")
feature_cols = joblib.load(base_dir / "models" / "feature_cols.pkl")

# --- Feature engineering helpers ---
def set_insulin(val):
    return "Normal" if 16 <= val <= 166 else "Abnormal"

def transform_input(data: dict):
    df = pd.DataFrame([data])

    # Replace 0 with NaN for selected features
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
        df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)

    # Fill NaN with median (global median for inference)
    df = df.fillna(df.median())

    # BMI categories
    if df["BMI"].iloc[0] < 18.5:
        df["NewBMI"] = "Underweight"
    elif df["BMI"].iloc[0] <= 24.9:
        df["NewBMI"] = "Normal"
    elif df["BMI"].iloc[0] <= 29.9:
        df["NewBMI"] = "Overweight"
    elif df["BMI"].iloc[0] <= 34.9:
        df["NewBMI"] = "Obesity 1"
    elif df["BMI"].iloc[0] <= 39.9:
        df["NewBMI"] = "Obesity 2"
    else:
        df["NewBMI"] = "Obesity 3"

    # Insulin category
    df["NewInsulinScore"] = df["Insulin"].apply(set_insulin)

    # Glucose categories
    if df["Glucose"].iloc[0] <= 70:
        df["NewGlucose"] = "Low"
    elif df["Glucose"].iloc[0] <= 99:
        df["NewGlucose"] = "Normal"
    elif df["Glucose"].iloc[0] <= 126:
        df["NewGlucose"] = "Overweight"
    else:
        df["NewGlucose"] = "Secret"

    # One-hot encode
    df = pd.get_dummies(df, columns=["NewBMI","NewInsulinScore","NewGlucose"], drop_first=True, dtype=int)

    # Align with training feature set
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Apply scaler to numeric features
    numeric_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

@app.post("/predict")
def predict(data: PatientData):
    # Apply preprocessing + feature engineering
    processed_df = transform_input(data.dict())

    # Predict
    prediction = best_model.predict(processed_df)[0]
    probability = best_model.predict_proba(processed_df)[0].tolist()

    return {
        "prediction": int(prediction),
        "probabilities": probability,
        "features_used": list(processed_df.columns)
    }
