from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import joblib

#determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent
print(base_dir)

fe_data_path = base_dir / "data/feature_engineering" / "feature_engineering_data.csv"
params_path = base_dir / "params.yaml"

#load parameters from params.yaml
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["model_building"]


#load raw dataset
df = pd.read_csv(fe_data_path)
# print(df.head(3))
print("feature engineered Data loaded...")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

#train test split
data_test_size = params["train_test_split"]["test_size"]
data_random_state = params["train_test_split"]["random_state"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_test_size, random_state=data_random_state)

#output testing directory path
testing_dir = base_dir / "data" / "testing_data"
testing_dir.mkdir(parents=True, exist_ok=True)

#save model testing data
X_test.to_csv(testing_dir / "X_test.csv", index=False)
y_test.to_csv(testing_dir / "y_test.csv", index=False)

X_train.to_csv(testing_dir / "X_train.csv", index=False)
y_train.to_csv(testing_dir / "y_train.csv", index=False)

print(f"model Testing data saved! Saved data to: {testing_dir}")


#output model directory path
models_dir = base_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

#Random Forests
max_depth = params["random_forest"]["max_depth"]
max_features = params["random_forest"]["max_features"]
min_samples_split = params["random_forest"]["min_samples_split"]
n_estimators = params["random_forest"]["n_estimators"]
random_state = params["random_forest"]["random_state"]

rf_tuned = RandomForestClassifier(
    max_depth= max_depth,
    max_features= max_features,
    min_samples_split= min_samples_split,
    n_estimators= n_estimators,
    random_state= random_state   
)

rf_tuned = rf_tuned.fit(X_train,y_train)

print("Random forest output: ", cross_val_score(rf_tuned, X_train, y_train, cv = 5, scoring="accuracy").mean())

# Save the model
joblib.dump(rf_tuned, models_dir / "rf_model.pkl")

#LightGBM 
learning_rate = params["lgbm"]["learning_rate"]
max_depth = params["lgbm"]["max_depth"]
n_estimators = params["lgbm"]["n_estimators"]
random_state = params["lgbm"]["random_state"]

lgbm_tuned = LGBMClassifier(
    learning_rate= learning_rate,
    max_depth= max_depth,
    n_estimators= n_estimators,
    random_state = random_state,
    verbose=-1 # suppress all LightGBM warnings/logs
).fit(X_train, y_train)

print("Light GBM output: ",cross_val_score(lgbm_tuned, X_train, y_train, cv = 5, scoring="accuracy").mean())

# Save the model
joblib.dump(lgbm_tuned, models_dir / "lgbm_model.pkl")

# XGBoost
learning_rate = params["xgb"]["learning_rate"]
n_estimators = params["xgb"]["n_estimators"]
max_depth = params["xgb"]["max_depth"]
min_samples_split = params["xgb"]["min_samples_split"]
random_state = params["xgb"]["random_state"]

xgb_tuned = GradientBoostingClassifier(
    learning_rate= learning_rate,
    n_estimators= n_estimators,
    max_depth= max_depth,            
    min_samples_split= min_samples_split,
    random_state= random_state
).fit(X_train, y_train)

print("XGB output: ",cross_val_score(xgb_tuned, X_train, y_train, cv = 5, scoring="accuracy").mean())

# Save the model
joblib.dump(xgb_tuned, models_dir / "xgb_model.pkl")


print("All the models pkl file has been saved!")