import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/raw/iris.csv")

train, test = train_test_split(
    df,
    test_size=1 - params['prepare']['split_ratio'],
    random_state=params['prepare']['random_state']
)

Path("data/processed").mkdir(parents=True, exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"Train: {len(train)}, Test: {len(test)}")
