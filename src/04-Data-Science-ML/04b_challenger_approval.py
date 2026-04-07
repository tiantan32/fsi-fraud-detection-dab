# Databricks notebook source
# MAGIC %md
# MAGIC # Challenger Model Approval (Human-In-The-Loop)
# MAGIC
# MAGIC Checks for approval tags on the Challenger model. Enables a "Human-In-The-Loop" gate
# MAGIC as part of the model deployment workflow.

# COMMAND ----------

# MAGIC %pip install --quiet mlflow-skinny --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

dbutils.widgets.text("model_name", f"{catalog}.{db}.fsi_fraud_model", "Model Name")
dbutils.widgets.text("model_version", "1", "Model Version")
dbutils.widgets.text("approval_tag_name", "Approval_Check", "Approval Tag to check")

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
tag_name = dbutils.widgets.get("approval_tag_name")

# Fetch model version's UC tags
tags = client.get_model_version(model_name, model_version).tags

# Check if any tag matches the approval tag name
if not any(tag == tag_name for tag in tags.keys()):
    raise Exception("Model version not approved for deployment")
else:
    if tags.get(tag_name).lower() == "approved":
        print("Model version approved for deployment")

        client.set_registered_model_alias(
            name=model_name,
            alias="Champion",
            version=model_version,
        )
    else:
        raise Exception("Model version not approved for deployment")

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [Batch inference]($./05_batch_inference)
