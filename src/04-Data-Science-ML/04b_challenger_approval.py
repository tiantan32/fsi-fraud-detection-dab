# Databricks notebook source
# MAGIC %md
# MAGIC # Challenger Model Approval — NOT in default DAG
# MAGIC
# MAGIC **This notebook is no longer wired into the default mlops job DAG.**
# MAGIC The production approval gate is at the GitHub Actions / bundle CI level
# MAGIC (the `prod` Environment requires reviewer approval before deploy). Once
# MAGIC the bundle is deployed, the job runs autonomously with automated quality
# MAGIC gates (fairness + metric + business KPI).
# MAGIC
# MAGIC Kept here as an OPTIONAL belt-and-suspenders gate for regulated FSI use
# MAGIC cases that need a per-model-version human sign-off in addition to the
# MAGIC code-promotion gate. To re-enable, add a task referencing this notebook
# MAGIC between `validate_model` and `deploy_serving` / `batch_inference` in
# MAGIC `resources/fsi_fraud_job.yml`, and have the reviewer set
# MAGIC `Approval_Check=Approved` on the UC model version (Catalog Explorer ->
# MAGIC Models -> version -> Tags) before repairing this task.

# COMMAND ----------

# MAGIC %pip install --quiet mlflow-skinny --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

dbutils.widgets.text("model_name", f"{catalog}.{db}.fsi_fraud_model", "Model Name")
dbutils.widgets.text("model_version", "", "Model Version (blank = latest Challenger)")
dbutils.widgets.text("approval_tag_name", "Approval_Check", "Approval Tag to check")
dbutils.widgets.dropdown("auto_approve_in_dev", "false", ["true", "false"], "Auto-approve in dev")

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
model_version_widget = dbutils.widgets.get("model_version")
tag_name = dbutils.widgets.get("approval_tag_name")
auto_approve_in_dev = dbutils.widgets.get("auto_approve_in_dev").lower() == "true"

# Resolve model_version: if blank, take the @Challenger alias
if not model_version_widget:
    model_version = client.get_model_version_by_alias(model_name, "Challenger").version
    print(f"Resolved @Challenger alias on {model_name} -> v{model_version}")
else:
    model_version = model_version_widget

# COMMAND ----------

# Dev-only convenience: if auto_approve_in_dev=true, set the tag here so the
# gate clears without a human. MUST be false in staging/prod so a reviewer
# has to manually set Approval_Check=Approved in the UC UI
# (Catalog Explorer -> Models -> version -> Tags).
if auto_approve_in_dev:
    print(
        "auto_approve_in_dev=true. Setting Approval_Check=Approved automatically. "
        "This bypass is for dev only and MUST NOT be enabled in staging/prod."
    )
    client.set_model_version_tag(
        name=model_name, version=model_version,
        key=tag_name, value="Approved",
    )

# COMMAND ----------

# Gate: read the tag and FAIL the task (halting the DAG) until a human
# reviewer sets Approval_Check=Approved on the model version.
tags = client.get_model_version(model_name, model_version).tags
current_value = tags.get(tag_name, "<missing>")
print(f"Approval state: {tag_name}={current_value}")

if current_value.lower() != "approved":
    raise Exception(
        f"Model {model_name} v{model_version} NOT approved for deployment. "
        f"Tag {tag_name}={current_value}. A reviewer must set "
        f"{tag_name}=Approved on this UC model version, then 'Repair run' "
        f"this approval_gate task to release the downstream deploy."
    )

print(f"Approved. Promoting {model_name} v{model_version} to @Champion.")
client.set_registered_model_alias(
    name=model_name,
    alias="Champion",
    version=model_version,
)

# Stamp who/when for audit lineage
import time
current_user = spark.sql("SELECT current_user()").first()[0]
client.set_model_version_tag(
    name=model_name, version=model_version,
    key="Promoted_To_Champion_By", value=current_user,
)
client.set_model_version_tag(
    name=model_name, version=model_version,
    key="Promoted_At_Epoch_S", value=str(int(time.time())),
)

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [Batch inference]($./05_batch_inference)
