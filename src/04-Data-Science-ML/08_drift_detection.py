# Databricks notebook source
# MAGIC %md
# MAGIC # Step 8: Drift Detection & Automated Retraining Trigger
# MAGIC
# MAGIC Query the Lakehouse Monitor's drift and profile metrics to detect:
# MAGIC 1. **Performance degradation** — F1 drops below threshold or fraud catch rate declines
# MAGIC 2. **Data drift** — distribution shift in predictions or key features
# MAGIC 3. **Business impact** — expected fraud loss exceeds acceptable threshold
# MAGIC
# MAGIC If violations exceed the threshold, signal the workflow to trigger retraining.

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

w = WorkspaceClient()

inference_table_name = f"{catalog}.{db}.fraud_inference_table"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Get monitor table names

# COMMAND ----------

monitor_info = w.quality_monitors.get(table_name=inference_table_name)
drift_table = monitor_info.drift_metrics_table_name
profile_table = monitor_info.profile_metrics_table_name

print(f"Profile table: {profile_table}")
print(f"Drift table: {drift_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check performance metrics

# COMMAND ----------

performance_df = spark.sql(f"""
    SELECT
        window.start AS time_window,
        f1_score.macro AS f1_macro,
        fraud_catch_rate,
        false_alarm_rate,
        expected_fraud_loss,
        model_version
    FROM {profile_table}
    WHERE log_type = 'INPUT'
      AND column_name = ':table'
      AND slice_key IS NULL
    ORDER BY window.start DESC
    LIMIT 30
""")

display(performance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check data drift

# COMMAND ----------

drift_df = spark.sql(f"""
    SELECT
        window.start AS time_window,
        column_name,
        js_distance AS drift_metric,
        model_version
    FROM {drift_table}
    WHERE column_name IN ('prediction', 'label', 'amount', 'balance_orig_ratio', 'is_cross_border')
      AND slice_key IS NULL
      AND drift_type = 'CONSECUTIVE'
    ORDER BY window.start DESC
    LIMIT 50
""")

display(drift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define violation rules and count breaches

# COMMAND ----------

# Thresholds
F1_THRESHOLD = 0.5                 # F1 should not drop below this
FRAUD_CATCH_THRESHOLD = 0.4        # Must catch at least 40% of fraud
EXPECTED_LOSS_THRESHOLD = -50000   # Average loss per window must not exceed this
DRIFT_THRESHOLD = 0.2              # Jensen-Shannon distance threshold

# Performance violations
perf_violations = 0
if not performance_df.isEmpty():
    perf_violations = performance_df.filter(
        (F.col("f1_macro") < F1_THRESHOLD) |
        (F.col("fraud_catch_rate") < FRAUD_CATCH_THRESHOLD) |
        (F.col("expected_fraud_loss") < EXPECTED_LOSS_THRESHOLD)
    ).count()

print(f"Performance violations: {perf_violations}")

# Drift violations
drift_violations = 0
if not drift_df.isEmpty():
    drift_violations = drift_df.filter(
        F.col("drift_metric") > DRIFT_THRESHOLD
    ).count()

print(f"Drift violations: {drift_violations}")

total_violations = perf_violations + drift_violations
print(f"\nTotal violations: {total_violations}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Signal retraining decision
# MAGIC
# MAGIC Set task value so the DABs workflow can use a **condition task** to branch:
# MAGIC - If `total_violations > 0` → trigger retraining pipeline
# MAGIC - Otherwise → no action needed

# COMMAND ----------

# Set as task value for workflow conditional branching
dbutils.jobs.taskValues.set(key="total_violations", value=total_violations)

if total_violations > 0:
    print(f"\n{'='*60}")
    print(f"  DRIFT DETECTED — {total_violations} violation(s)")
    print(f"  Retraining will be triggered by the workflow.")
    print(f"{'='*60}")
else:
    print(f"\n{'='*60}")
    print(f"  NO DRIFT — Model is performing within acceptable bounds.")
    print(f"  No retraining needed.")
    print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retraining Workflow
# MAGIC
# MAGIC This notebook is designed to run as part of a DABs workflow defined in `resources/fsi_fraud_job.yml`.
# MAGIC The workflow uses a **condition task** after this notebook:
# MAGIC
# MAGIC ```yaml
# MAGIC - task_key: check_violations
# MAGIC   condition_task:
# MAGIC     op: GREATER_THAN
# MAGIC     left: "{{tasks.drift_detection.values.total_violations}}"
# MAGIC     right: "0"
# MAGIC
# MAGIC - task_key: retrain_model
# MAGIC   depends_on:
# MAGIC     - task_key: check_violations
# MAGIC       outcome: "true"
# MAGIC   notebook_task:
# MAGIC     notebook_path: ./02_model_training_hpo.py
# MAGIC ```
# MAGIC
# MAGIC This creates a **closed-loop MLOps system**:
# MAGIC
# MAGIC ```
# MAGIC Monitor → Detect Drift → Retrain → Validate → Promote → Serve → Monitor
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC [Go back to the introduction]($../00-FSI-fraud-detection-introduction-lakehouse)
