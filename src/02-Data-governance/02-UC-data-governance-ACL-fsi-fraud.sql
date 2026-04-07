-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC # Ensuring Governance and security for our Banking platform
-- MAGIC
-- MAGIC Data governance and security is hard when it comes to a complete Data Platform. SQL GRANT on tables isn't enough and security must be enforced for multiple data assets (dashboards, Models, files etc).
-- MAGIC
-- MAGIC Unity Catalog is key for data governance, including:
-- MAGIC * Fine-grained ACL
-- MAGIC * Audit log
-- MAGIC * Data lineage
-- MAGIC * Data exploration & discovery
-- MAGIC * Sharing data with external organizations (Delta Sharing)

-- COMMAND ----------

-- MAGIC %run ../_resources/00-setup $reset_all_data=false

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Exploring our Banking database
-- MAGIC
-- MAGIC Unity Catalog works with 3 layers:
-- MAGIC * CATALOG
-- MAGIC * SCHEMA (or DATABASE)
-- MAGIC * TABLE
-- MAGIC
-- MAGIC All unity catalog is available with SQL (`CREATE CATALOG IF NOT EXISTS my_catalog` ...)

-- COMMAND ----------

SELECT CURRENT_CATALOG();

-- COMMAND ----------

-- DBTITLE 1,Review available tables
SHOW TABLES;

-- COMMAND ----------

-- DBTITLE 1,Granting access to Analysts & Data Engineers
-- Let's grant our ANALYSTS a SELECT permission:
-- Note: make sure you created analysts and dataengineers groups first.
-- GRANT SELECT ON TABLE gold_transactions TO `analysts`;
-- GRANT SELECT, MODIFY ON SCHEMA TO `dataengineers`;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Going further with Data governance
-- MAGIC
-- MAGIC Unity Catalog provides:
-- MAGIC - **Fine-grained ACL**: Row-level and column-level security
-- MAGIC - **Data lineage**: Track how data flows between tables
-- MAGIC - **Audit logs**: Know who accessed what data and when
-- MAGIC - **Delta Sharing**: Share data with external organizations
-- MAGIC
-- MAGIC # Next: Start building analysis with Databricks SQL
-- MAGIC
-- MAGIC Jump to the [BI / Data warehousing notebook]($../03-BI-data-warehousing/03-BI-Datawarehousing-fraud) or [Go back to the introduction]($../00-FSI-fraud-detection-introduction-lakehouse)
