from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

#determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent
print(base_dir) 

sample_data_path = base_dir / "data" / "diabetes.csv"
params_path = base_dir / "params.yaml"

#load parameters from params.yaml
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["data_ingestion"]

n_neighbors = params["n_neighbors"]
# random_state = params["random_state"]

#load raw dataset
df = pd.read_csv(sample_data_path)
# print(df.head(3))
print("Data loaded...")

# #stratified train/test split using parameters from params.yaml
# train_data, test_data = train_test_split(
#     df,
#     test_size= test_size,
#     random_state= random_state
# )

# #output raw directory path
# raw_dir = base_dir / "data" / "raw"
# raw_dir.mkdir(parents=True, exist_ok=True)

# #save raw train and test split datasets
# train_data.to_csv(raw_dir / "train.csv", index=False)
# test_data.to_csv(raw_dir / "test.csv", index=False)

# print(f"Data ingestion complete(test_size={test_size}, random_state={random_state}). Saved raw split data to: {raw_dir}")

#We saw on df.head() that some features contain 0, it doesn't make sense here and this indicates missing value Below we replace 0 value by NaN:
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

# The missing values ​​will be filled with the median values ​​of each variable.
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

# The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick.
columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]


#Local Outlier Factor (LOF)
# We determine outliers between all variables with the LOF method
from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors)
lof.fit_predict(df)

df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:30]

#We choose the threshold value according to lof scores
threshold = np.sort(df_scores)[7]

#We delete those that are higher than the threshold
outlier = df_scores > threshold
preprocessed_df = df[outlier]
print(preprocessed_df.shape)

#output preprocessed directory path
preprocessed_dir = base_dir / "data" / "preprocessed"
preprocessed_dir.mkdir(parents=True, exist_ok=True)

#save preprocessed data
preprocessed_df.to_csv(preprocessed_dir / "preprocessed_data.csv", index=False)

print(f"Data preprocessing completed! Saved data to: {preprocessed_dir}")


