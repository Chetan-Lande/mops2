import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("../data/dataset.csv")


X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# Separate numeric + categorical columns
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = RandomForestRegressor(n_estimators=100, max_depth=10)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)


    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(pipeline, "model")

print("RMSE:", rmse)
print("R2 Score:", r2)
