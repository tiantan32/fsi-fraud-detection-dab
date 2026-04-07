# Databricks notebook source
# MAGIC %md
# MAGIC # Deploying Your AI Functions with Databricks AI Agents
# MAGIC
# MAGIC In this notebook, you'll learn how to take the functions you defined in the previous notebook and integrate them into a **Databricks AI Agent**.
# MAGIC
# MAGIC ## Step 1: Prepare Your Workspace
# MAGIC * Duplicate this browser window
# MAGIC * Keep this guide open for reference
# MAGIC * Arrange windows side-by-side

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 2: Access the Databricks Playground
# MAGIC
# MAGIC Find the **Playground** under the **Machine Learning** section in your Databricks Workspace's left sidebar.
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/refs/heads/main/images/cross_demo_assets/AI_Agent_GIFs/AI_agent_open_playground.gif" alt="Opening the Playground" width="70%">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 3: Configure Your Agent Functions
# MAGIC
# MAGIC Your functions are organized in Unity Catalog:
# MAGIC `my_catalog.my_schema.get_customer_details`
# MAGIC `my_catalog.my_schema.fraud_report_generator`
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/refs/heads/main/images/cross_demo_assets/AI_Agent_GIFs/AI_agent_function_selection.gif" alt="Function Selection" width="70%">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 4: Export Your Agent
# MAGIC
# MAGIC * Verify all functions are selected in the Playground
# MAGIC * Click "Export"
# MAGIC * Save the exported notebooks to this directory (`05-Generative-AI/`)
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/refs/heads/main/images/cross_demo_assets/AI_Agent_GIFs/AI_agent_export_from_playground.gif" alt="Exporting Agent" width="70%">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Deploy Your AI Agent
# MAGIC
# MAGIC 1. Navigate to where your exported "driver" notebook is located.
# MAGIC 2. Follow that notebook's documentation and guide.
# MAGIC 3. Close this notebook.
# MAGIC
# MAGIC ### What's next?
# MAGIC * Agent Evaluation: Test and validate performance
# MAGIC * Agent Deployment: Place your agent in a Databricks App or the Playground
# MAGIC
# MAGIC [Go back to the introduction]($../00-FSI-fraud-detection-introduction-lakehouse)
