import pandas as pd
import yaml
import json
import pickle
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Подключение к MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("HW5_Iris")

with mlflow.start_run():
    mlflow.log_param("model_type", params['train']['model_type'])
    mlflow.log_param("n_estimators", params['train']['n_estimators'])

    model = RandomForestClassifier(
        n_estimators=params['train']['n_estimators'],
        random_state=params['prepare']['random_state']
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("models/model.pkl")

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    print(f"Accuracy: {acc:.4f}")
