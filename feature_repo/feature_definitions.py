from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, ValueType

# Сущность
iris_sample = Entity(
    name="iris_sample",
    join_keys=["sample_id"],
    value_type=ValueType.INT64,
)

# Источник
iris_source = FileSource(
    name="iris_data_source",
    path="../data/raw/iris.parquet",
    timestamp_field="event_timestamp",
)

# feature view
iris_features = FeatureView(
    name="iris_features",
    entities=[iris_sample],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width", dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width", dtype=Float64),
        Field(name="target", dtype=Int64),
    ],
    source=iris_source,
    online=True,
)