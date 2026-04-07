-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC # FSI & Banking platform with Databricks Data Intelligence Platform - Fraud detection in real time
-- MAGIC
-- MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/lakehouse-fsi-fraud-0.png " style="float: left; margin-right: 30px; margin-bottom: 50px" width="600px" />
-- MAGIC
-- MAGIC <br/>
-- MAGIC
-- MAGIC ## What is The Databricks Data Intelligence Platform for Banking?
-- MAGIC
-- MAGIC It's the only enterprise data platform that allows you to leverage all your data, from any source, on any workload to optimize your business with real time data, at the lowest cost.
-- MAGIC
-- MAGIC The Lakehouse allows you to centralize all your data, from customer & retail banking data to real time fraud detection, providing operational speed and efficiency at a scale never before possible.
-- MAGIC
-- MAGIC ### Simple
-- MAGIC   One single platform and governance/security layer for your data warehousing and AI to **accelerate innovation** and **reduce risks**.
-- MAGIC
-- MAGIC ### Open
-- MAGIC   Built on open source and open standards. You own your data and prevent vendor lock-in.
-- MAGIC
-- MAGIC ### Multicloud
-- MAGIC   One consistent data platform across clouds. Process your data where you need.
-- MAGIC
-- MAGIC ## Deployed via Databricks Asset Bundles (DABs)
-- MAGIC
-- MAGIC This demo is packaged as a DAB for reproducible, multi-environment deployment:
-- MAGIC - **`databricks.yml`**: Bundle configuration with dev/staging/prod targets
-- MAGIC - **`resources/`**: Pipeline and job definitions as YAML
-- MAGIC - **`src/`**: All notebooks, parameterized via bundle variables
-- MAGIC
-- MAGIC Deploy with: `databricks bundle deploy -t dev`

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Reducing Fraud with the Lakehouse
-- MAGIC
-- MAGIC In this demo, we'll build an end-to-end Banking platform, collecting data from multiple sources in real time.
-- MAGIC
-- MAGIC Based on this information, we'll be able to proactively reduce Fraud by rating financial transaction risk in real-time.
-- MAGIC
-- MAGIC <img width="1000px" src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/lakehouse-fsi-fraud-overview-0.png" />
-- MAGIC
-- MAGIC 1. Ingest and create our Banking database, with tables easy to query in SQL
-- MAGIC 2. Secure data and grant read access to the Data Analyst and Data Science teams.
-- MAGIC 3. Run BI queries to analyze existing Fraud
-- MAGIC 4. Build ML models & deploy them to provide real-time fraud detection capabilities.
-- MAGIC 5. Leverage GenAI for automated fraud report generation

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1/ Data Ingestion (SDP Pipeline)
-- MAGIC Open the [SDP pipeline notebook]($./01-Data-ingestion/01.1-sdp-sql/01-SDP-fraud-detection-SQL)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2/ Data Governance (Unity Catalog)
-- MAGIC Open [Unity Catalog notebook]($./02-Data-governance/02-UC-data-governance-ACL-fsi-fraud)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3/ BI & Analytics (Databricks SQL)
-- MAGIC Open the [Data warehousing notebook]($./03-BI-data-warehousing/03-BI-Datawarehousing-fraud)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4/ End-to-End MLOps — Advanced
-- MAGIC
-- MAGIC The full MLOps lifecycle for fraud detection:
-- MAGIC
-- MAGIC | Step | Notebook | What it does |
-- MAGIC |------|----------|--------------|
-- MAGIC | 1 | [01_feature_engineering]($./04-Data-Science-ML/01_feature_engineering) | Feature Store + on-demand functions |
-- MAGIC | 2 | [02_model_training_hpo]($./04-Data-Science-ML/02_model_training_hpo) | Optuna HPO across LightGBM/XGBoost/LR |
-- MAGIC | 3 | [03_model_registration]($./04-Data-Science-ML/03_model_registration) | Register to UC as Challenger |
-- MAGIC | 4 | [04_challenger_validation]($./04-Data-Science-ML/04_challenger_validation) | 5 quality gates + auto-promote to Champion |
-- MAGIC | 5 | [05_batch_inference]($./04-Data-Science-ML/05_batch_inference) | Score transactions with Champion model |
-- MAGIC | 6 | [06_realtime_serving]($./04-Data-Science-ML/06_realtime_serving) | Model Serving endpoint + A/B testing |
-- MAGIC | 7 | [07_model_monitoring]($./04-Data-Science-ML/07_model_monitoring) | Lakehouse Monitor with custom fraud metrics |
-- MAGIC | 8 | [08_drift_detection]($./04-Data-Science-ML/08_drift_detection) | Drift detection + conditional retraining |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5/ Generative AI
-- MAGIC - [05.1-AI-Functions-Creation]($./05-Generative-AI/05.1-AI-Functions-Creation) — AI-powered fraud reports
-- MAGIC - [05.2-Agent-Creation-Guide]($./05-Generative-AI/05.2-Agent-Creation-Guide) — Deploy as AI Agent
