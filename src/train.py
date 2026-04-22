import pandas as pd
import yaml
import json
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Загрузка параметров
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Загрузка данных
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Путь к БД
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("HW5_Iris")

with mlflow.start_run(run_name="baseline_model"):

    # Логирование параметров
    mlflow.log_param("model_type", params['train']['model_type'])
    mlflow.log_param("n_estimators", params['train']['n_estimators'])
    mlflow.log_param("split_ratio", params['prepare']['split_ratio'])
    mlflow.log_param("random_state", params['prepare']['random_state'])

    # Обучение
    model = RandomForestClassifier(
        n_estimators=params['train']['n_estimators'],
        random_state=params['prepare']['random_state']
    )
    model.fit(X_train, y_train)

    # Предсказание и метрики
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    # Сохранение модели
    Path("models").mkdir(exist_ok=True)
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(model, "model")

    # Сохранение метрик для DVC
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    print(f"Accuracy: {acc:.4f}")
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
