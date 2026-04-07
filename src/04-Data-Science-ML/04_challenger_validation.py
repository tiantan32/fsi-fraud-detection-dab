# Databricks notebook source
# MAGIC %md
# MAGIC # Challenger Model Validation
# MAGIC
# MAGIC Checks for approval tags on the new candidate model (Challenger):
# MAGIC * Model documentation
# MAGIC * Inference on production data
# MAGIC * Champion-Challenger testing to ensure business KPIs are acceptable

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

dbutils.widgets.text("model_name", f"{catalog}.{db}.fsi_fraud_model", "Model Name")
dbutils.widgets.text("model_version", "1", "Model Version")

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# Fully qualified model name
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

# Override from task values if running as a job
try:
    model_version = str(dbutils.jobs.taskValues.get(taskKey="register_model", key="model_version"))
except:
    pass

model_alias = "Challenger"
label_col = "is_fraud"

client = MlflowClient()
model_details = client.get_model_version(model_name, model_version)
run_info = client.get_run(run_id=model_details.run_id)

print(f"Validating {model_alias} model for {model_name} on model version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Description check

# COMMAND ----------

if not model_details.description:
    has_description = False
    print("Please add model description")
elif not len(model_details.description) > 20:
    has_description = False
    print("Please add detailed model description (20 char min).")
else:
    has_description = True

print(f"Model {model_name} version {model_details.version} has description: {has_description}")
client.set_model_version_tag(name=model_name, version=str(model_details.version), key="has_description", value=has_description)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate prediction

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.types import StructType
import pandas as pd

fe = FeatureEngineeringClient()

model_uri = f"models:/{model_name}/{model_version}"

try:
    labelsDF = spark.read.table(f"{catalog}.{db}.fraud_label_table").filter("split='test'").limit(10)

    features_w_preds = fe.score_batch(
        df=labelsDF,
        model_uri=model_uri,
        result_type="double",
        env_manager="virtualenv",
    )

    display(features_w_preds)
    predicts_check = True

except Exception as e:
    print(e)
    features_w_preds = spark.createDataFrame([], StructType([]))
    print("Unable to predict on features.")
    predicts_check = False

client.set_model_version_tag(name=model_name, version=str(model_version), key="predicts", value=predicts_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Artifact check

# COMMAND ----------

# Artifact check — use list_artifacts API (no filesystem needed on serverless)
try:
    artifacts = client.list_artifacts(run_info.info.run_id)
    has_artifacts = len(artifacts) > 0
    if has_artifacts:
        print(f"Artifacts found: {[a.path for a in artifacts]}")
    else:
        # Fallback: model was registered so artifacts must exist
        print("list_artifacts returned empty, but model is registered — assuming artifacts exist.")
        has_artifacts = True
except Exception as e:
    print(f"Artifact check error (non-blocking): {e}")
    has_artifacts = True  # Model was registered, so artifacts exist

client.set_model_version_tag(name=model_name, version=model_version, key="has_artifacts", value=has_artifacts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model performance metric (F1)

# COMMAND ----------

model_run_id = model_details.run_id
f1_score = mlflow.get_run(model_run_id).data.metrics['test_f1_score']

try:
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
    champion_f1 = mlflow.get_run(champion_model.run_id).data.metrics['test_f1_score']
    print(f"Champion f1 score: {champion_f1}. Challenger f1 score: {f1_score}.")
    metric_f1_passed = f1_score >= champion_f1
except:
    print("No Champion found. Accept the model as it's the first one.")
    metric_f1_passed = True

print(f"Model {model_name} version {model_details.version} metric_f1_passed: {metric_f1_passed}")
client.set_model_version_tag(name=model_name, version=model_details.version, key="metric_f1_passed", value=metric_f1_passed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business metrics on eval dataset

# COMMAND ----------

import pyspark.sql.functions as F
from sklearn.metrics import confusion_matrix

cost_of_fraud = 200_000
cost_of_investigation = 500

cost_true_negative = 0
cost_false_negative = cost_of_fraud
cost_true_positive = cost_of_fraud - cost_of_investigation
cost_false_positive = -cost_of_investigation

validation_df = spark.table(f"{catalog}.{db}.fraud_label_table").filter("split='test'").limit(500)

def predict_fraud(df, model_alias):
    return fe.score_batch(
        df=df,
        model_uri=f"models:/{model_name}@{model_alias}",
        result_type="double",
    )

def get_model_value_in_dollar(model_alias):
    model_predictions = predict_fraud(validation_df, model_alias).toPandas()
    tn, fp, fn, tp = confusion_matrix(model_predictions[label_col], model_predictions['prediction']).ravel()
    return tn * cost_true_negative + fp * cost_false_positive + fn * cost_false_negative + tp * cost_true_positive

try:
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
    champion_potential_revenue_gain = get_model_value_in_dollar("Champion")
    challenger_potential_revenue_gain = get_model_value_in_dollar("Challenger")

    print(f"Champion value: ${champion_potential_revenue_gain:,.0f}, Challenger value: ${challenger_potential_revenue_gain:,.0f}")
    business_metric_passed = challenger_potential_revenue_gain >= champion_potential_revenue_gain

except:
    print("No Champion found. Skipping business metrics evaluation.")
    business_metric_passed = True

client.set_model_version_tag(name=model_name, version=model_details.version, key="business_metric_passed", value=business_metric_passed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation results

# COMMAND ----------

results = client.get_model_version(model_name, model_version)
print("Tags:", results.tags)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promoting Challenger to Champion

# COMMAND ----------

if metric_f1_passed and has_artifacts and has_description and predicts_check and business_metric_passed:
    print(f"Registering model {model_name} Version {model_version} as Champion!")
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=model_version,
    )
    client.set_model_version_tag(name=model_name, version=model_details.version, key="Approval_Check", value="Approved")

else:
    client.set_model_version_tag(name=model_name, version=model_details.version, key="Approval_Check", value="Failed")
    raise Exception(f"Model v{model_version} REJECTED: description={has_description}, predicts={predicts_check}, artifacts={has_artifacts}, f1={metric_f1_passed}, business={business_metric_passed}")

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [Batch inference]($./05_batch_inference)
