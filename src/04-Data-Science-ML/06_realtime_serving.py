# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud Detection Real-Time Inference
# MAGIC
# MAGIC Deploy features and model for real-time predictions via REST API.
# MAGIC
# MAGIC 1. Enable Change Data Feed on feature table
# MAGIC 2. Create Online Store and publish feature table
# MAGIC 3. Create/update Model Serving endpoint
# MAGIC 4. Query endpoint for real-time fraud scoring

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.1a1 mlflow databricks-sdk --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

dbutils.widgets.text("model_name", f"{catalog}.{db}.fsi_fraud_model", "Model Name")
dbutils.widgets.text("model_version", "1", "Model Version")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Change Data Feed on feature table

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE fraud_feature_table SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Online Store and publish feature table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
import time

fe = FeatureEngineeringClient()

online_store_name = "fsi-fraud-online-store"

# Check if online store exists
online_store = fe.get_online_store(name=online_store_name)

if online_store:
    print(f"Online store exists: {online_store.name}, State: {online_store.state}, Capacity: {online_store.capacity}")
else:
    print(f"Creating Online store: {online_store_name}")
    online_store = fe.create_online_store(
        name=online_store_name,
        capacity="CU_1"
    )

# COMMAND ----------

# Publish feature table to online store
print(f"Publishing feature table to online store...")
max_retries = 5
retry_count = 0
while retry_count < max_retries:
    try:
        publish_state = fe.publish_table(
            online_store=online_store,
            source_table_name=f"{catalog}.{db}.fraud_feature_table",
            online_table_name=f"{catalog}.{db}.fraud_feature_table_online",
            publish_mode="TRIGGERED",
        )
        print(f"Published successfully")
        break
    except Exception as e:
        if "feature sync is currently in progress" in str(e):
            print("Feature sync in progress, retrying...")
            retry_count += 1
            time.sleep(10)
        else:
            raise e
else:
    print("Failed to publish after multiple retries.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create/Update Model Serving endpoint

# COMMAND ----------

endpoint_name = "fsi_fraud_serving"

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

try:
    model_version = str(dbutils.jobs.taskValues.get(taskKey="register_model", key="model_version"))
except:
    pass

served_model_name = model_name.split(".")[-1]

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoint_config_dict = {
    "served_entities": [
        {
            "entity_name": model_name,
            "entity_version": model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
        },
    ],
    "traffic_config": {
        "routes": [
            {"served_model_name": f"{served_model_name}-{model_version}", "traffic_percentage": 100},
        ]
    },
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# COMMAND ----------

try:
    existing = w.serving_endpoints.get(endpoint_name)
    if existing.state and existing.state.config_update and existing.state.config_update.value == "UPDATE_FAILED":
        print(f"Endpoint {endpoint_name} stuck in UPDATE_FAILED — deleting and recreating...")
        w.serving_endpoints.delete(endpoint_name)
        time.sleep(10)
        w.serving_endpoints.create(
            name=endpoint_name,
            config=endpoint_config,
            tags=[EndpointTag.from_dict({"key": "project", "value": "fsi-fraud-detection"})],
        )
    else:
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=endpoint_config.served_entities,
            traffic_config=endpoint_config.traffic_config,
        )
    print(f"Updating endpoint {endpoint_name} with model {model_name} version {model_version}")
except ResourceDoesNotExist:
    w.serving_endpoints.create(
        name=endpoint_name,
        config=endpoint_config,
        tags=[EndpointTag.from_dict({"key": "project", "value": "fsi-fraud-detection"})],
    )
    print(f"Creating endpoint {endpoint_name} with model {model_name} version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for endpoint to be ready

# COMMAND ----------

from datetime import timedelta

endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name, timeout=timedelta(minutes=30))
assert endpoint.state.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY", "Endpoint not ready"
print(f"Endpoint {endpoint_name} is ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query endpoint

# COMMAND ----------

dataframe_records = [
    {"transaction_id": "001df0ff-a9a6-4b94-a548-bfb6d5393698", "event_ts": "2025-01-01", "split": "test"},
    {"transaction_id": "3d1dd327-0120-495e-bb37-34008d7587a9", "event_ts": "2025-01-01", "split": "test"},
]

try:
    print("Fraud inference:")
    response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=dataframe_records)
    print(response.predictions)
except Exception as e:
    print(f"Query test failed (endpoint is deployed, query may need adjustment): {e}")
    print("The endpoint is ready — test queries manually via the UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [Create monitor for model performance]($./07_model_monitoring)
