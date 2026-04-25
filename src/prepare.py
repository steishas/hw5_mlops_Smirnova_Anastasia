import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

# Загрузка параметров
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Загрузка данных из parquet
df = pd.read_parquet('data/raw/iris.parquet')

# Указываем нужные колонки для признаков
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[feature_columns]
y = df['target']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=1 - params['prepare']['split_ratio'],
    random_state=params['prepare']['random_state'],
    stratify=y
)

# Сохранение
Path('data/processed').mkdir(parents=True, exist_ok=True)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv('data/processed/train.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
print(f"Features: {feature_columns}")
