# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Model Registration — Champion/Challenger Pattern
# MAGIC
# MAGIC Register the best model from HPO to Unity Catalog and set it as the **Challenger**.
# MAGIC The existing production model (if any) remains as the **Champion** until validation passes.

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

model_name = f"{catalog}.{db}.fsi_fraud_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Find the best run from HPO

# COMMAND ----------

current_user = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{current_user}/fsi-fraud-detection/fraud_model_hpo"

# Try to get run_id from task values (when running as a job), otherwise search
try:
    best_run_id = dbutils.jobs.taskValues.get(taskKey="model_training", key="best_run_id")
except:
    # Fallback: find the best run from the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experiment {experiment_name} not found. Run model training first.")
    best_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED' AND run_name = 'fraud-hpo-best-run'",
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    )
    best_run_id = best_run.iloc[0]["run_id"]

run_info = mlflow.get_run(best_run_id)
best_f1 = run_info.data.metrics.get("test_f1_score", 0)
best_classifier = run_info.data.params.get("classifier", "unknown")

print(f"Best run: {best_run_id}")
print(f"Classifier: {best_classifier}, F1: {best_f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register model to Unity Catalog

# COMMAND ----------

model_details = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=model_name,
)

print(f"Registered model: {model_name}, version: {model_details.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set Challenger alias and add metadata

# COMMAND ----------

# Set as Challenger
client.set_registered_model_alias(
    name=model_name,
    alias="Challenger",
    version=model_details.version,
)

# Update model description
client.update_registered_model(
    name=model_name,
    description=(
        "FSI Fraud Detection model trained on banking transaction features. "
        "Uses Feature Store for offline/online feature serving and on-demand risk scoring. "
        "Champion/Challenger promotion via automated validation."
    ),
)

# Update version description with metrics
client.update_model_version(
    name=model_name,
    version=model_details.version,
    description=(
        f"Trained with {best_classifier} via Optuna HPO. "
        f"Test F1={best_f1:.4f}. "
        f"Features from fraud_feature_table + transaction_risk_score on-demand function."
    ),
)

# Tag with validation status
client.set_model_version_tag(model_name, model_details.version, "validation_status", "invalid")

# Copy fairness + SHAP outcomes from the upstream explainability_fairness task
# onto the model version, so reviewers see them in Catalog Explorer without
# having to open the MLflow run.
try:
    fairness_passed = dbutils.jobs.taskValues.get(
        taskKey="explainability_fairness", key="fairness_passed"
    )
    worst_di = dbutils.jobs.taskValues.get(
        taskKey="explainability_fairness", key="worst_disparate_impact_ratio"
    )
    client.set_model_version_tag(
        model_name, model_details.version,
        "Fairness_Check", "Passed" if fairness_passed == "true" else "Failed",
    )
    client.set_model_version_tag(
        model_name, model_details.version,
        "Worst_Disparate_Impact", str(worst_di),
    )
    client.set_model_version_tag(
        model_name, model_details.version, "Explainability", "SHAP",
    )
    print(f"Stamped fairness tags: Fairness_Check={fairness_passed}, Worst_DI={worst_di}")
except Exception as e:
    # If the explainability task didn't set values (e.g. interactive registration),
    # mark fairness as Missing so validate_model can refuse to promote.
    client.set_model_version_tag(
        model_name, model_details.version, "Fairness_Check", "Missing",
    )
    print(f"No upstream fairness task values found ({e}); tagged Fairness_Check=Missing.")

print(f"Model {model_name} v{model_details.version} set as @Challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Grant permissions for shared use

# COMMAND ----------

try:
    spark.sql(f"GRANT EXECUTE ON FUNCTION `{model_name}` TO `account users`")
except Exception as e:
    print(f"Could not set model permissions (may require admin): {e}")

# COMMAND ----------

# Pass model version downstream
dbutils.jobs.taskValues.set(key="model_version", value=model_details.version)
dbutils.jobs.taskValues.set(key="model_name", value=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: [04_challenger_validation]($./04_challenger_validation) — Validate the Challenger against the Champion
