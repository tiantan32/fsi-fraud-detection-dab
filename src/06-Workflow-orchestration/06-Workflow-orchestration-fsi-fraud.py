# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Deploying and orchestrating the full workflow
# MAGIC
# MAGIC ## Databricks Asset Bundles (DABs) for orchestration
# MAGIC
# MAGIC This demo is packaged as a **Databricks Asset Bundle**, which means the workflow orchestration is defined declaratively in YAML:
# MAGIC
# MAGIC * **`resources/fsi_fraud_job.yml`** — Defines the full workflow with task dependencies
# MAGIC * **`resources/fsi_fraud_pipeline.yml`** — Defines the SDP pipeline
# MAGIC * **`databricks.yml`** — Bundle configuration with dev/staging/prod targets
# MAGIC
# MAGIC ## Deployment
# MAGIC
# MAGIC ```bash
# MAGIC # Validate the bundle
# MAGIC databricks bundle validate -t dev
# MAGIC
# MAGIC # Deploy to dev
# MAGIC databricks bundle deploy -t dev
# MAGIC
# MAGIC # Run the init job
# MAGIC databricks bundle run fsi_fraud_init -t dev
# MAGIC
# MAGIC # Deploy to production
# MAGIC databricks bundle deploy -t prod
# MAGIC ```
# MAGIC
# MAGIC ## Workflow Tasks
# MAGIC
# MAGIC ### Job 1: `fsi_fraud_init` — Full MLOps Setup
# MAGIC
# MAGIC 1. **init_data** — Download and prepare raw data in UC Volumes
# MAGIC 2. **start_sdp_pipeline** — Run the SDP pipeline (Bronze → Silver → Gold)
# MAGIC 3. **feature_engineering** — Create Feature Store table + on-demand functions
# MAGIC 4. **model_training** — Optuna HPO across LightGBM/XGBoost/LogisticRegression
# MAGIC 5. **register_model** — Register best model to UC as @Challenger
# MAGIC 6. **validate_model** — 5 automated quality gates → promote to @Champion
# MAGIC 7. **batch_inference** — Score transactions, save with model version
# MAGIC 8. **deploy_serving** — Model Serving endpoint with auto-capture
# MAGIC 9. **setup_monitoring** — Lakehouse Monitor with custom fraud metrics
# MAGIC 10. **create_ai_functions** — GenAI fraud report functions
# MAGIC
# MAGIC ### Job 2: `fsi_fraud_monitor_retrain` — Scheduled Retraining Loop
# MAGIC
# MAGIC 1. **refresh_pipeline** — Ingest new data via SDP
# MAGIC 2. **batch_inference** — Score new data with Champion
# MAGIC 3. **drift_detection** — Query Lakehouse Monitor for violations
# MAGIC 4. **check_violations** — Condition task: retrain if violations > 0
# MAGIC 5. **retrain_model** → **reregister_model** → **revalidate_model** (conditional)
# MAGIC
# MAGIC ## Scheduling
# MAGIC
# MAGIC To schedule the workflow, add a `schedule` block to `resources/fsi_fraud_job.yml`:
# MAGIC
# MAGIC ```yaml
# MAGIC schedule:
# MAGIC   quartz_cron_expression: "0 0 * * * ?"  # Every hour
# MAGIC   timezone_id: "UTC"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-environment deployment
# MAGIC
# MAGIC DABs supports different configurations per target:
# MAGIC
# MAGIC | Target | Catalog | Schema | Mode |
# MAGIC |--------|---------|--------|------|
# MAGIC | **dev** | main | `dbdemos_fsi_fraud_detection_dev_<user>` | development |
# MAGIC | **staging** | main | `dbdemos_fsi_fraud_detection_staging` | default |
# MAGIC | **prod** | main | `dbdemos_fsi_fraud_detection` | production |
# MAGIC
# MAGIC All notebooks read catalog/schema from parameters, so the same code runs correctly across all environments.

# COMMAND ----------

# MAGIC %md
# MAGIC [Go back to the introduction]($../00-FSI-fraud-detection-introduction-lakehouse)
