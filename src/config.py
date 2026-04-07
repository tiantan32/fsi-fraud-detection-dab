# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration file
# MAGIC
# MAGIC Configuration for the FSI Fraud Detection demo.
# MAGIC When deployed via Databricks Asset Bundles, these values are overridden by job/task parameters.
# MAGIC You can also override them manually by changing the values below.

# COMMAND ----------

# Read from task/widget parameters if available, otherwise use defaults
try:
    catalog = dbutils.widgets.get("catalog")
except:
    catalog = "main"

try:
    schema = dbutils.widgets.get("schema")
except:
    schema = "dbdemos_fsi_fraud_detection"

db = dbName = schema

try:
    volume_name = dbutils.widgets.get("volume_name")
except:
    volume_name = "fraud_raw_data"
