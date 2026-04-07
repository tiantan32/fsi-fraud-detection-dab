# Databricks notebook source
# MAGIC %md
# MAGIC # Monitor Model using Lakehouse Monitoring
# MAGIC
# MAGIC Attach a monitor to the inference table to track data drift, prediction drift, and model quality.
# MAGIC
# MAGIC Lakehouse Monitoring generates:
# MAGIC - **Profile metrics table** (`_profile_metrics`): accuracy, F1, custom metrics
# MAGIC - **Drift metrics table** (`_drift_metrics`): statistical tests for data drift

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk mlflow-skinny --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create/Update Inference Table

# COMMAND ----------

# DBTITLE 1,Create inference table (first time or full overwrite)
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE fraud_inference_table AS
# MAGIC   SELECT i.*, CAST(l.is_fraud AS DOUBLE) AS label
# MAGIC   FROM fraud_offline_inference i
# MAGIC   LEFT JOIN fraud_label_table l ON i.transaction_id = l.transaction_id AND i.event_ts = l.event_ts
# MAGIC   ORDER BY i.inference_timestamp;
# MAGIC
# MAGIC ALTER TABLE fraud_inference_table SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# DBTITLE 1,Subsequent calls: Update labels when available
# MAGIC %sql
# MAGIC MERGE INTO fraud_inference_table AS i
# MAGIC   USING fraud_label_table AS l
# MAGIC   ON i.transaction_id = l.transaction_id AND i.event_ts = l.event_ts
# MAGIC   WHEN MATCHED THEN UPDATE SET i.label = CAST(l.is_fraud AS DOUBLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create baseline table

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE fraud_baseline AS
# MAGIC   SELECT i.* EXCEPT (transaction_id, event_ts, inference_timestamp), CAST(l.is_fraud AS DOUBLE) AS label
# MAGIC   FROM fraud_offline_inference i
# MAGIC   LEFT JOIN fraud_label_table l ON i.transaction_id = l.transaction_id AND i.event_ts = l.event_ts;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define custom metric
# MAGIC
# MAGIC Expected fraud loss: average dollar loss from missed fraud (false negatives).

# COMMAND ----------

from pyspark.sql.types import DoubleType, StructField
from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType

expected_loss_metric = [
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="expected_fraud_loss",
        input_columns=[":table"],
        definition="""avg(CASE
            WHEN {{prediction_col}} != {{label_col}} AND CAST({{label_col}} AS INT) = 1 THEN -200000
            ELSE 0 END
        )""",
        output_data_type=StructField("output", DoubleType()).json(),
    )
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create monitor

# COMMAND ----------

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorCronSchedule,
    MonitorInfoStatus, MonitorRefreshInfoState,
)

w = WorkspaceClient()
inference_table_fqn = f"{catalog}.{db}.fraud_inference_table"

print(f"Creating monitor for {inference_table_fqn}")

try:
    info = w.quality_monitors.create(
        table_name=inference_table_fqn,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            timestamp_col="inference_timestamp",
            granularities=["1 day"],
            model_id_col="model_version",
            label_col="label",
        ),
        schedule=MonitorCronSchedule(
            quartz_cron_expression="0 0 12 * * ?",
            timezone_id="PST",
        ),
        assets_dir=f"{os.getcwd()}/monitoring",
        output_schema_name=f"{catalog}.{db}",
        baseline_table_name=f"{catalog}.{db}.fraud_baseline",
        slicing_exprs=["is_cross_border = 1.0"],
        custom_metrics=expected_loss_metric,
    )
except Exception as e:
    if "already exist" in str(e).lower():
        print(f"Monitor already exists, retrieving info:")
        info = w.quality_monitors.get(table_name=inference_table_fqn)
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for monitor to be active

# COMMAND ----------

while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
    info = w.quality_monitors.get(table_name=inference_table_fqn)
    time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"
print(f"Monitor is active: {info.status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trigger initial refresh and wait

# COMMAND ----------

def get_refreshes():
    return w.quality_monitors.list_refreshes(table_name=inference_table_fqn).refreshes

refreshes = get_refreshes()
if len(refreshes) == 0:
    w.quality_monitors.run_refresh(table_name=inference_table_fqn)
    time.sleep(5)
    refreshes = get_refreshes()

run_info = refreshes[0]
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
    run_info = w.quality_monitors.get_refresh(table_name=inference_table_fqn, refresh_id=run_info.refresh_id)
    print(f"Waiting for refresh to complete: {run_info.state}...")
    time.sleep(180)

assert run_info.state == MonitorRefreshInfoState.SUCCESS, "Monitor refresh failed"
print("Monitor refresh complete!")

# COMMAND ----------

w.quality_monitors.get(table_name=inference_table_fqn)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect the monitoring dashboard
# MAGIC
# MAGIC Navigate to `fraud_inference_table` in the Catalog Explorer, go to the **Quality** tab and click **View dashboard**.
# MAGIC
# MAGIC Next: [Detect drift and trigger model retrain]($./08_drift_detection)
