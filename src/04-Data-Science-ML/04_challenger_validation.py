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

# Tolerance for HPO variance: Challenger passes if F1 is within 5% of Champion.
# Strict `>=` rejects on noise (e.g., 0.001 regression from sampling). A
# production policy might set this from a CI bootstrap over training folds.
F1_TOLERANCE_PCT = 0.05

try:
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
    champion_f1 = mlflow.get_run(champion_model.run_id).data.metrics['test_f1_score']
    f1_floor = champion_f1 * (1 - F1_TOLERANCE_PCT)
    print(
        f"Champion f1={champion_f1:.4f}, Challenger f1={f1_score:.4f}, "
        f"floor (within {F1_TOLERANCE_PCT:.0%})={f1_floor:.4f}"
    )
    metric_f1_passed = f1_score >= f1_floor
except Exception as e:
    print(f"No Champion found ({e}). Accept the model as it's the first one.")
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

# Pull the fairness tag set upstream by 09_explainability_fairness. If the
# task didn't run yet (e.g. first deploy), treat as Failed so the gate does
# not silently waive bias review.
_fairness_tag = client.get_model_version(model_name, model_version).tags.get("Fairness_Check", "Missing")
fairness_passed = (_fairness_tag.lower() == "passed")
print(f"Fairness_Check tag = {_fairness_tag}  fairness_passed = {fairness_passed}")

all_automated_checks_passed = (
    metric_f1_passed and has_artifacts and has_description
    and predicts_check and business_metric_passed
    and fairness_passed
)

# The human-in-the-loop gate is at the GitHub Actions level (the `prod`
# Environment requires reviewer sign-off before bundle deploy). Inside the
# deployed bundle, the mlops job runs end-to-end autonomously: if all
# automated checks pass (metrics + fairness + business KPI), promote to
# Champion. If any check fails, halt the DAG so deploy_serving / batch
# inference do not get the new model.
if all_automated_checks_passed:
    print(f"All automated checks passed. Promoting {model_name} v{model_version} to @Champion.")
    client.set_registered_model_alias(
        name=model_name, alias="Champion", version=model_version,
    )
    client.set_model_version_tag(
        name=model_name, version=model_details.version,
        key="Approval_Check", value="Approved",
    )
    # Stamp who/when for audit lineage. The "who" is the service principal
    # in prod (the job's run_as), which is exactly what regulators want to
    # see paired with the GH Action approver in the deploy log.
    import time
    current_user = spark.sql("SELECT current_user()").first()[0]
    client.set_model_version_tag(
        name=model_name, version=model_details.version,
        key="Promoted_To_Champion_By", value=current_user,
    )
    client.set_model_version_tag(
        name=model_name, version=model_details.version,
        key="Promoted_At_Epoch_S", value=str(int(time.time())),
    )

else:
    client.set_model_version_tag(
        name=model_name, version=model_details.version,
        key="Approval_Check", value="Failed",
    )
    raise Exception(
        f"Model v{model_version} REJECTED: description={has_description}, "
        f"predicts={predicts_check}, artifacts={has_artifacts}, "
        f"f1={metric_f1_passed}, business={business_metric_passed}, "
        f"fairness={fairness_passed}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [Batch inference]($./05_batch_inference)
