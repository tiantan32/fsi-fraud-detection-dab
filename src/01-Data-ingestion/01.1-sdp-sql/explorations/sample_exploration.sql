-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Sample Data Exploration
-- MAGIC Explore raw data from volumes before pipeline processing.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Read from config — these values are set via DABs variables
-- MAGIC catalog = spark.conf.get("spark.databricks.unityCatalog.catalog", "main")
-- MAGIC schema = spark.conf.get("spark.databricks.unityCatalog.schema", "dbdemos_fsi_fraud_detection")
-- MAGIC volume_name = "fraud_raw_data"
-- MAGIC folder = f"/Volumes/{catalog}/{schema}/{volume_name}"
-- MAGIC
-- MAGIC # Explore raw transactions
-- MAGIC display(spark.read.format("json").load(f"{folder}/transactions").limit(10))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Explore raw customers
-- MAGIC display(spark.read.format("csv").option("header", "true").option("multiLine", "true").load(f"{folder}/customers").limit(10))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Explore country codes
-- MAGIC display(spark.read.format("csv").option("header", "true").load(f"{folder}/country_code"))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Explore fraud reports
-- MAGIC display(spark.read.format("csv").option("header", "true").load(f"{folder}/fraud_report").limit(10))
