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

# Publish feature table to online store.
#
# We use publish_mode="SNAPSHOT" — a one-shot full copy of the current Delta
# state. No ongoing sync, no streaming costs. The serving endpoint queries
# this snapshot at request time.
#
# IMPORTANT: SNAPSHOT does NOT auto-update when the source Delta table is
# rewritten. So we drop-if-exists before re-publishing — each deploy_serving
# run creates a FRESH snapshot of whatever's currently in fraud_feature_table.
# Without this, a stale snapshot from a prior run would silently keep
# serving old data.
#
# Also covers: the orphan-table-after-drop+recreate bug we hit when
# 01_feature_engineering used to drop the source. That's now fixed in 01
# (idempotent get-or-create) so the source_table_id is stable.
from databricks.sdk import WorkspaceClient as _WC
_w = _WC()
online_table_full = f"{catalog}.{db}.fraud_feature_table_online"

# Drop the existing synced table (and let the underlying Postgres table
# get cleaned up) so the publish below creates a fresh snapshot. Catch
# both UC-level and synced-table-level "not found" so first-run is fine.
print(f"Dropping any existing synced table {online_table_full} for a fresh snapshot...")
try:
    _w.api_client.do("DELETE", f"/api/2.0/database/synced_tables/{online_table_full}")
    print(f"  deleted synced table {online_table_full}")
    time.sleep(10)  # give Lakebase a moment to clean up the Postgres-side table
except Exception as e:
    print(f"  no existing synced table to drop ({type(e).__name__}: {str(e)[:120]})")

# Take a fresh SNAPSHOT of the current source Delta state.
print(f"Publishing feature table to online store (SNAPSHOT mode)...")
max_retries = 5
retry_count = 0
while retry_count < max_retries:
    try:
        publish_state = fe.publish_table(
            online_store=online_store,
            source_table_name=f"{catalog}.{db}.fraud_feature_table",
            online_table_name=online_table_full,
            publish_mode="SNAPSHOT",
        )
        print(f"Published successfully: {publish_state}")
        break
    except Exception as e:
        msg = str(e)
        if "feature sync is currently in progress" in msg or "already exists" in msg.lower():
            print(f"Transient publish error, retrying... ({msg[:120]})")
            retry_count += 1
            time.sleep(15)
        else:
            raise e
else:
    raise Exception("Failed to publish after multiple retries.")

# Wait for the SNAPSHOT to actually finish copying before updating serving
# endpoint. The serving endpoint validates online-store readiness; if we
# update it before the snapshot is done we'd get
#   "No suitable online store found for feature table ..."
print("Waiting for SNAPSHOT to complete...")
source_delta_version = spark.sql(
    f"DESCRIBE HISTORY {catalog}.{db}.fraud_feature_table LIMIT 1"
).first()["version"]
print(f"Source Delta version captured by snapshot: {source_delta_version}")

deadline = time.time() + 30 * 60  # 30 min cap
while time.time() < deadline:
    try:
        st = _w.api_client.do(
            "GET",
            f"/api/2.0/database/synced_tables/{online_table_full}",
        )
    except Exception as e:
        print(f"  synced-table not yet readable ({type(e).__name__}); retrying...")
        time.sleep(15)
        continue
    sync_status = st.get("data_synchronization_status") or {}
    last = sync_status.get("last_sync") or {}
    synced_version = (last.get("delta_table_sync_info") or {}).get("delta_commit_version", -1)
    detailed = sync_status.get("detailed_state", "")
    print(f"  synced_version={synced_version}  detailed_state={detailed}")
    if synced_version >= source_delta_version and "ONLINE_NO_PENDING_UPDATE" in detailed:
        print("SNAPSHOT is complete.")
        break
    time.sleep(15)
else:
    raise Exception(
        f"SNAPSHOT did not reach source version {source_delta_version} within 30 min. "
        f"Serving endpoint would fail with 'no suitable online store found'."
    )

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
