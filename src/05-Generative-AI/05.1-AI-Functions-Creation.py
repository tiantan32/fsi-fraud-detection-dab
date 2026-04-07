# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Gen AI-Powered Fraud Detection in Modern Banking
# MAGIC
# MAGIC Banks today face unprecedented fraud risks as criminals use advanced digital tactics to exploit vulnerabilities in online banking.
# MAGIC
# MAGIC <div style="background: #f7f7f7; border-left: 5px solid #ff5f46; padding: 20px; margin: 20px 0; font-size: 18px;">
# MAGIC   "Fraud scams and bank fraud schemes resulted in <b>$485.6 billion</b> in losses globally last year." -
# MAGIC   <a href="https://www.nasdaq.com/global-financial-crime-report" target="_blank">Nasdaq 2024 Global Financial Crime Report</a>
# MAGIC </div>
# MAGIC
# MAGIC This demo shows how banks can use Databricks and AI to automate fraud detection, improve accuracy, and streamline compliance.

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Function 1: Get Customer Details

# COMMAND ----------

# DBTITLE 1,Retrieve Customer Details by Transaction ID
# MAGIC %sql
# MAGIC DROP FUNCTION IF EXISTS get_customer_details;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION get_customer_details (
# MAGIC   tran_id STRING COMMENT 'Transaction ID of the customer to be searched'
# MAGIC )
# MAGIC RETURNS TABLE(
# MAGIC   id STRING,
# MAGIC   is_fraud BOOLEAN,
# MAGIC   amount DOUBLE,
# MAGIC   customer_id STRING,
# MAGIC   nameDest STRING,
# MAGIC   nameOrig STRING,
# MAGIC   type STRING,
# MAGIC   firstname STRING,
# MAGIC   lastname STRING,
# MAGIC   email STRING,
# MAGIC   address STRING,
# MAGIC   country STRING,
# MAGIC   creation_date STRING,
# MAGIC   age_group DOUBLE,
# MAGIC   countryOrig_name STRING,
# MAGIC   countryDest_name STRING
# MAGIC )
# MAGIC COMMENT "This function returns the customer details for a given Transaction ID, along with their transaction details"
# MAGIC RETURN (
# MAGIC   SELECT
# MAGIC     id,
# MAGIC     is_fraud,
# MAGIC     amount,
# MAGIC     customer_id,
# MAGIC     nameDest,
# MAGIC     nameOrig,
# MAGIC     type,
# MAGIC     firstname,
# MAGIC     lastname,
# MAGIC     email,
# MAGIC     address,
# MAGIC     country,
# MAGIC     creation_date,
# MAGIC     age_group,
# MAGIC     countryOrig_name,
# MAGIC     countryDest_name
# MAGIC   FROM
# MAGIC     gold_transactions
# MAGIC   WHERE
# MAGIC     id = tran_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Fraudulent transaction lookup

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_customer_details(
# MAGIC     '001df0ff-a9a6-4b94-a548-bfb6d5393698'
# MAGIC ) AS prediction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Non-fraudulent transaction lookup

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_customer_details(
# MAGIC     '3d1dd327-0120-495e-bb37-34008d7587a9'
# MAGIC ) AS prediction

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Function 2: Generate Fraud Report

# COMMAND ----------

# DBTITLE 1,Fraud Detection and Customer Notification Guidelines
prompt = """
You are a fraud detection assistant for banks, helping to identify and report potentially fraudulent transactions.

### Tools Available:
1. **get_customer_details** – Retrieves customer KYC profile, transaction history, and risk indicators for a given **Transaction ID**.

### Your Role:
- **If a transaction is fraudulent:**
    - **Internal Fraud Report:** Generate a professional fraud report for internal banking use, including:
        - Transaction ID, Customer ID, date/time, transaction amount, channel (e.g., online, ATM), and relevant risk indicators (e.g., velocity, geo-location anomaly, device mismatch).
        - Summary of investigation findings (e.g., pattern detected, rule triggered, behavioral anomaly).
        - Recommended actions (e.g., file SAR, freeze account, escalate to compliance).
        - Use clear, structured language and banking terminology.
    - **Customer Notification:** Draft a formal email to the customer:
        - Greet customer by first name.
        - Clearly state that suspicious activity was detected on their account.
        - Advise next steps (e.g., contact fraud team, monitor account, card reissue).
        - If the customer's country is not English-speaking, also provide the email in their native language.
        - Use a professional, reassuring tone and proper formatting.
        - Do not include a closing.
    - Ensure all details match the provided **Transaction ID** and **Customer ID** before proceeding.

- **If the transaction is not fraudulent:**
    - Briefly confirm that the transaction is legitimate and can be processed.

- Do not provide your reasoning step; just give the response.
"""

# COMMAND ----------

# DBTITLE 1,Create Fraud Report Function with Transaction Details
spark.sql("DROP FUNCTION IF EXISTS fraud_report_generator")

spark.sql(f"""
  CREATE OR REPLACE FUNCTION fraud_report_generator(
    id STRING,
    is_fraud BOOLEAN,
    amount DOUBLE,
    customer_id STRING,
    nameDest STRING,
    nameOrig STRING,
    type STRING,
    firstname STRING,
    lastname STRING,
    email STRING,
    address STRING,
    country STRING,
    creation_date STRING,
    age_group DOUBLE,
    countryOrig_name STRING,
    countryDest_name STRING
  )
  RETURNS STRING
  LANGUAGE SQL
  COMMENT "This function generates a fraud report based on the transaction details and customer information."
  RETURN (
    SELECT ai_query(
      'databricks-meta-llama-3-1-405b-instruct',
      CONCAT(
        "{prompt}",
        'Transaction and customer details: ',
        TO_JSON(NAMED_STRUCT(
          'id', id,
          'is_fraud', is_fraud,
          'amount', amount,
          'customer_id', customer_id,
          'nameDest', nameDest,
          'nameOrig', nameOrig,
          'type', type,
          'firstname', firstname,
          'lastname', lastname,
          'email', email,
          'address', address,
          'country', country,
          'creation_date', creation_date,
          'age_group', age_group,
          'countryOrig_name', countryOrig_name,
          'countryDest_name', countryDest_name
        ))
      )
    )
  )
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Fraud report for a fraudulent transaction

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT fraud_report_generator(
# MAGIC     '001df0ff-a9a6-4b94-a548-bfb6d5393698', true, 400000, '80a60ee7-a482-4e69-aa8f-2914f051f5b5', 'CC0290993355', 'C0047984490', 'TRANSFER', 'Angela', 'Tran', 'eburch@smith.com', '282 Amanda Road Apt. 209 Matthewview, GU 81248' , 'GRC', '2021-11-05', 6, 'France', 'Nigeria'
# MAGIC   ) as transaction_report

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Functions in Batch to Suspicious Transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE fraudulent_transactions_report_table AS
# MAGIC SELECT
# MAGIC   id AS id,
# MAGIC   is_fraud,
# MAGIC   fraud_report_generator(
# MAGIC     id, is_fraud, amount, customer_id, nameDest, nameOrig, type, firstname, lastname,
# MAGIC     email, address, country, creation_date, age_group, countryOrig_name, countryDest_name
# MAGIC   ) AS claim_report
# MAGIC FROM (
# MAGIC   SELECT *
# MAGIC   FROM gold_transactions
# MAGIC   WHERE is_fraud = true
# MAGIC   LIMIT 5
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM fraudulent_transactions_report_table;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Proceed to notebook **[05.2-Agent-Creation-Guide]($./05.2-Agent-Creation-Guide)** to package the above functions into an AI Agent with Databricks.
