# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud Detection Model Batch Inference
# MAGIC
# MAGIC Score transactions using the **@Champion** model via Feature Store.
# MAGIC Predictions saved with model version metadata for downstream monitoring.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering mlflow lightgbm xgboost scikit-learn --upgrade
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# Install exact model requirements to match training env (needed for env_manager="local")
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
model_name_full = f"{catalog}.{db}.fsi_fraud_model"
requirements_path = ModelsArtifactRepository(f"models:/{model_name_full}@Champion").download_artifacts(artifact_path="requirements.txt")
print(f"Installing model requirements from: {requirements_path}")

# COMMAND ----------

# MAGIC %pip install -r $requirements_path --quiet
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

env_manager = "local"  # For fe.score_batch() — use "local" if pip installing all model artifacts

# COMMAND ----------

# DBTITLE 1,Set model alias for batch inference
model_alias = "Champion"
model_name = f"{catalog}.{db}.fsi_fraud_model"
model_uri = f"models:/{model_name}@{model_alias}"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Load labels/IDs to be scored
inference_df = spark.read.table(f"{catalog}.{db}.fraud_label_table")
inference_df = inference_df.limit(100)  # Limit for demo speed

label_col = "is_fraud"

# Batch score
preds_df = fe.score_batch(df=inference_df, model_uri=model_uri, result_type="double", env_manager=env_manager)
display(preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save predictions for monitoring

# COMMAND ----------

from mlflow import MlflowClient
from datetime import datetime
from pyspark.sql import functions as F

client = MlflowClient()
model_version = client.get_model_version_by_alias(name=model_name, alias=model_alias).version

offline_inference_df = preds_df.drop("split") \
                              .withColumn("model_version", F.lit(model_version)) \
                              .withColumn("inference_timestamp", F.lit(datetime.now()))

offline_inference_df.write.mode("append") \
                    .option("overwriteSchema", True) \
                    .saveAsTable(f"{catalog}.{db}.fraud_offline_inference")

display(offline_inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [Serve the features and model in real-time]($./06_realtime_serving)
