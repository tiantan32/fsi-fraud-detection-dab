# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Step 1: Feature Engineering with Feature Store
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/fsi/fraud-detection/fsi-fraud-ds.png" width="900px" style="float: right; margin-left: 10px"/>
# MAGIC
# MAGIC Our gold transaction data from the SDP pipeline is ready. Now we need to:
# MAGIC
# MAGIC 1. **Create features** suitable for fraud detection (balance ratios, transaction velocity, geographic risk)
# MAGIC 2. **Register them in Feature Store** for discoverability, lineage, and real-time serving
# MAGIC 3. **Separate labels** to avoid label leakage and enable point-in-time lookups
# MAGIC 4. **Create on-demand feature functions** for dynamic computation at serving time

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and explore gold transactions

# COMMAND ----------

gold_df = spark.table("gold_transactions")
display(gold_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Engineer features for fraud detection

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Build features from gold transactions
features_df = gold_df.select(
    # Primary key
    F.col("id").alias("transaction_id"),
    # Timestamp for point-in-time lookups
    F.current_timestamp().alias("event_ts"),
    # Transaction features
    "step", "amount", "type",
    # Balance features
    "oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest",
    "diffOrig", "diffDest",
    # Customer features
    "age_group", "country",
    F.col("isUnauthorizedOverdraft").cast("double").alias("is_unauthorized_overdraft"),
    # Geographic features
    "countryOrig", "countryDest",
    "countryOrig_name", "countryDest_name",
    "countryLongOrig_long", "countryLatOrig_lat",
    "countryLongDest_long", "countryLatDest_lat",
).withColumns({
    # Engineered: balance ratio — how much of the origin balance is being sent
    "balance_orig_ratio": F.when(F.col("oldBalanceOrig") > 0,
        F.col("amount") / F.col("oldBalanceOrig")).otherwise(F.lit(0.0)),
    # Engineered: is the entire balance being transferred?
    "is_full_transfer": F.when(
        (F.col("newBalanceOrig") == 0) & (F.col("oldBalanceOrig") > 0),
        F.lit(1.0)).otherwise(F.lit(0.0)),
    # Engineered: cross-border indicator
    "is_cross_border": F.when(
        F.col("countryOrig") != F.col("countryDest"),
        F.lit(1.0)).otherwise(F.lit(0.0)),
    # Engineered: destination balance anomaly — large deposits to empty accounts
    "dest_balance_anomaly": F.when(
        (F.col("oldBalanceDest") == 0) & (F.col("amount") > 100000),
        F.lit(1.0)).otherwise(F.lit(0.0)),
}).dropDuplicates(["transaction_id"])

print(f"Feature table shape: {features_df.count()} rows, {len(features_df.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register features in Feature Store

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

feature_table_name = f"{catalog}.{db}.fraud_feature_table"

# DO NOT drop + recreate the feature table on each run. Doing so gives it a
# new UC table_id, which orphans the downstream synced (Lakebase-backed)
# online table — that synced table is bound to a specific source table_id
# and silently syncs 0 rows after a drop+create, breaking realtime serving
# with: "No suitable online store found for feature table ...".
#
# Idempotent pattern: create-if-missing, then MERGE the new feature rows.
# Delta history (and the synced-table binding) is preserved across runs.
try:
    fe.get_table(name=feature_table_name)
    print(f"Feature table {feature_table_name} exists; merging new rows.")
except Exception:
    print(f"Feature table {feature_table_name} does not exist; creating.")
    fe.create_table(
        name=feature_table_name,
        primary_keys=["transaction_id", "event_ts"],
        timestamp_keys=["event_ts"],
        schema=features_df.schema,
        description="Transaction features for fraud detection. Includes balance ratios, geographic risk indicators, and engineered signals.",
    )

fe.write_table(
    name=feature_table_name,
    df=features_df,
    mode="merge",
)

display(fe.read_table(name=feature_table_name).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create label table (separate from features to avoid leakage)

# COMMAND ----------

label_df = gold_df.select(
    F.col("id").alias("transaction_id"),
    F.current_timestamp().alias("event_ts"),
    F.col("is_fraud").cast("int").alias("is_fraud"),
    # Keep split indicator for reproducible train/test splits
    F.when(F.rand(seed=42) < 0.8, F.lit("train")).otherwise(F.lit("test")).alias("split"),
).dropDuplicates(["transaction_id"])

label_df.write.mode("overwrite").option("overwriteSchema", True) \
    .saveAsTable(f"{catalog}.{db}.fraud_label_table")

display(spark.table(f"{catalog}.{db}.fraud_label_table").groupBy("is_fraud", "split").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create on-demand feature function
# MAGIC
# MAGIC This UC function computes a risk score at serving time based on transaction characteristics.
# MAGIC It's evaluated dynamically — no need to pre-compute and store.

# COMMAND ----------

spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"USE SCHEMA `{db}`")

spark.sql("""
CREATE OR REPLACE FUNCTION transaction_risk_score(
    amount_in DOUBLE,
    balance_orig_ratio_in DOUBLE,
    is_full_transfer_in DOUBLE,
    is_cross_border_in DOUBLE
)
RETURNS FLOAT
LANGUAGE PYTHON
COMMENT 'Computes a heuristic risk score for a transaction based on amount, balance usage, and geography'
AS $$
    score = 0.0
    # High-value transactions
    if amount_in > 200000:
        score += 0.3
    elif amount_in > 100000:
        score += 0.15
    # Full balance drain
    if is_full_transfer_in > 0:
        score += 0.25
    # High balance ratio
    if balance_orig_ratio_in > 0.9:
        score += 0.15
    # Cross-border
    if is_cross_border_in > 0:
        score += 0.1
    return min(score, 1.0)
$$
""")

print("On-demand feature function 'transaction_risk_score' created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Enable Change Data Feed for online table sync

# COMMAND ----------

spark.sql(f"ALTER TABLE `{catalog}`.`{db}`.fraud_feature_table SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("Change Data Feed enabled on fraud_feature_table.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We created:
# MAGIC - **`fraud_feature_table`**: Feature Store table with 20+ features, time-series keyed
# MAGIC - **`fraud_label_table`**: Separate label table with train/test split
# MAGIC - **`transaction_risk_score`**: On-demand UC function for dynamic risk scoring
# MAGIC
# MAGIC Next: [02_model_training_hpo]($./02_model_training_hpo) — Train with Optuna hyperparameter optimization
