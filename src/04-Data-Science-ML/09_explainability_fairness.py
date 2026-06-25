# Databricks notebook source
# MAGIC %md
# MAGIC # Step 9: Explainability (SHAP) + Subgroup Fairness
# MAGIC
# MAGIC Runs **immediately after `02_model_training_hpo`** and **BEFORE registration**
# MAGIC so a biased model never reaches the UC registry. The model is loaded from
# MAGIC the MLflow run (via the `best_run_id` task value), not from a `@Challenger`
# MAGIC alias.
# MAGIC
# MAGIC Outputs:
# MAGIC
# MAGIC 1. **Global SHAP importance** — mean |SHAP value| per feature, written to
# MAGIC    `model_global_importance` UC table and logged into the MLflow run.
# MAGIC 2. **Per-row SHAP** on a sample of the test set, written to
# MAGIC    `model_row_explanations` UC table for the AIBI dashboard to render
# MAGIC    case-level explanations.
# MAGIC 3. **Subgroup fairness metrics** — precision / recall / FPR / selection rate
# MAGIC    sliced by `country`, `age_group`, and `is_cross_border`, plus
# MAGIC    disparate-impact ratio. Written to `model_fairness_metrics` UC table and
# MAGIC    logged into the MLflow run.
# MAGIC
# MAGIC Hands off `fairness_passed` and `worst_disparate_impact_ratio` to downstream
# MAGIC tasks via `dbutils.jobs.taskValues`. If `fairness_passed=False` this task
# MAGIC RAISES — the DAG halts and `register_model` never runs.
# MAGIC
# MAGIC SHAP > LIME for tree models because attributions are exact and additive;
# MAGIC LIME perturbations are non-deterministic and slow. The same per-row table
# MAGIC also feeds any LIME-style consumer.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow shap==0.46.* lightgbm xgboost
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking.client import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

# Resolve the training run via the task value set by 02_model_training_hpo.
# This runs BEFORE register_model, so there is NO @Challenger alias yet —
# we explain the run directly. The run_id is later attached as a tag to the
# UC model version by 03_model_registration.
try:
    run_id = dbutils.jobs.taskValues.get(taskKey="model_training", key="best_run_id")
except Exception:
    # Fallback for interactive runs: pick the most recent fraud-hpo run.
    current_user = spark.sql("SELECT current_user()").first()[0]
    experiment_name = f"/Users/{current_user}/fsi-fraud-detection/fraud_model_hpo"
    exp = mlflow.get_experiment_by_name(experiment_name)
    best_run = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="run_name = 'fraud-hpo-best-run'",
        order_by=["start_time DESC"], max_results=1,
    )
    run_id = best_run.iloc[0]["run_id"]

print(f"Explaining MLflow run_id={run_id} (registration has not happened yet).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the Challenger model and a stratified sample of test data

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureFunction, FeatureLookup

fe = FeatureEngineeringClient()

feature_table = f"{catalog}.{db}.fraud_feature_table"
label_table = f"{catalog}.{db}.fraud_label_table"

feature_lookups = [
    FeatureLookup(
        table_name=feature_table,
        lookup_key=["transaction_id"],
        timestamp_lookup_key="event_ts",
    ),
    FeatureFunction(
        udf_name=f"{catalog}.{db}.transaction_risk_score",
        input_bindings={
            "amount_in": "amount",
            "balance_orig_ratio_in": "balance_orig_ratio",
            "is_full_transfer_in": "is_full_transfer",
            "is_cross_border_in": "is_cross_border",
        },
        output_name="risk_score",
    ),
]

labels_df = spark.table(label_table).filter("split = 'test'")
training_set = fe.create_training_set(
    df=labels_df,
    label="is_fraud",
    feature_lookups=feature_lookups,
    exclude_columns=["transaction_id", "event_ts"],
    exclude_null_labels=True,
)
test_pdf = training_set.load_df().sample(fraction=0.05, seed=42).toPandas()

# Down-sample to keep SHAP tractable on serverless drivers
explain_pdf = test_pdf.sample(n=min(2000, len(test_pdf)), random_state=42).reset_index(drop=True)
y_true = explain_pdf["is_fraud"].astype(int).values
X = explain_pdf.drop(columns=["is_fraud"])

print(f"Explainability sample: {len(explain_pdf)} rows")

# COMMAND ----------

# Load the sklearn pipeline from the dedicated "raw_model" artifact that
# 02_model_training_hpo logs alongside the feature-store model. The "model"
# artifact is feature-store-wrapped (pyfunc) and `mlflow.sklearn.load_model`
# refuses it with `Model does not have the "sklearn" flavor`. The "raw_model"
# artifact is unwrapped sklearn flavor, only used here for SHAP introspection
# — never served, never promoted.
sklearn_pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/raw_model")
preprocessor = sklearn_pipeline.named_steps["preprocessor"]
classifier = sklearn_pipeline.named_steps["classifier"]
clf_name = type(classifier).__name__
print(f"Classifier under inspection: {clf_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. SHAP global + per-row attributions

# COMMAND ----------

import shap

# Transform once so SHAP sees the post-preprocessing feature space
X_transformed = preprocessor.transform(X)
# Recover post-OHE column names from the fitted ColumnTransformer
try:
    feature_names_out = preprocessor.get_feature_names_out().tolist()
except Exception:
    feature_names_out = [f"f_{i}" for i in range(X_transformed.shape[1])]

# Pick the right explainer per classifier type
if clf_name in {"LGBMClassifier", "XGBClassifier"}:
    explainer = shap.TreeExplainer(classifier)
    shap_vals = explainer.shap_values(X_transformed)
    # XGBoost binary returns ndarray; LightGBM binary returns list[neg, pos]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
else:
    # LogisticRegression and any future linear/kernel models
    explainer = shap.LinearExplainer(classifier, X_transformed)
    shap_vals = explainer.shap_values(X_transformed)

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
global_importance = (
    pd.DataFrame({"feature": feature_names_out, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
print("Top 15 features by mean |SHAP|:")
print(global_importance.head(15).to_string(index=False))

# Per-row SHAP table — write transaction_id-keyed so the AIBI dashboard
# can render case-level explanations.
per_row_long = pd.DataFrame(shap_vals, columns=feature_names_out)
per_row_long["transaction_id"] = explain_pdf.get("transaction_id", pd.Series(range(len(explain_pdf))))
per_row_long["model_version"] = model_version
per_row_long["explained_at"] = pd.Timestamp.utcnow()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Subgroup fairness metrics
# MAGIC
# MAGIC We compute fraud-classifier behavior on each protected/sensitive slice.
# MAGIC No external dependency on fairlearn — the four metrics below are the
# MAGIC standard set Reg uses for adverse-action review and are easy to audit.

# COMMAND ----------

y_pred = sklearn_pipeline.predict(X)

def group_metrics(df, group_col, y_true_in, y_pred_in):
    """Return precision / recall / FPR / selection_rate per group level."""
    rows = []
    for level, idx in df.groupby(group_col).groups.items():
        yt = y_true_in[idx]
        yp = y_pred_in[idx]
        n = len(yt)
        positives = int(yt.sum())
        negatives = n - positives
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / positives if positives > 0 else float("nan")
        fpr = fp / negatives if negatives > 0 else float("nan")
        sel = float(yp.mean())
        rows.append({
            "slice_dimension": group_col,
            "slice_value": str(level),
            "n": n,
            "positive_rate": positives / n if n else float("nan"),
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "selection_rate": sel,
        })
    return pd.DataFrame(rows)

import os
import sys

# Add the bundle's src/ to sys.path so we can import the unit-tested lib.
# The bundle syncs src/ to /Workspace/.../files/src/; this notebook lives at
# /Workspace/.../files/src/04-Data-Science-ML/09_explainability_fairness,
# so two os.path.dirname() calls give us the src/ root.
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()
_src_root = "/Workspace" + os.path.dirname(os.path.dirname(_nb_path))
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)
print(f"Added to sys.path: {_src_root}")

from lib.fairness import group_metrics, is_fair as _is_fair, select_protected_columns

explain_pdf_reset = explain_pdf.reset_index(drop=True)
slices = [
    group_metrics(explain_pdf_reset, col, y_true, np.asarray(y_pred))
    for col in select_protected_columns(explain_pdf_reset.columns)
]
fairness_df = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame()
print(fairness_df.to_string(index=False))

# Disparate-impact gate; threshold is a policy choice (see lib/fairness.py).
is_fair, worst = _is_fair(fairness_df, threshold=1.25)
if not fairness_df.empty:
    print("\nDisparate-impact ratio by dimension:")
    by_dim = fairness_df.groupby("slice_dimension")["selection_rate"]
    print((by_dim.max() / by_dim.min().replace(0, np.nan)).to_string())

print(f"\nworst_disparate_impact_ratio={worst:.3f}  is_fair={is_fair}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Persist explanations + fairness as UC tables; tag the model version

# COMMAND ----------

importance_table = f"{catalog}.{db}.model_global_importance"
perrow_table = f"{catalog}.{db}.model_row_explanations"
fairness_table = f"{catalog}.{db}.model_fairness_metrics"

# Key everything by run_id (model version does not exist yet). 03_model_registration
# will read this back and stamp the model version with fairness tags + a pointer.
global_importance_to_write = global_importance.assign(
    run_id=run_id,
    computed_at=pd.Timestamp.utcnow(),
)
per_row_long["run_id"] = run_id

(
    spark.createDataFrame(global_importance_to_write)
    .write.mode("append").option("mergeSchema", "true")
    .saveAsTable(importance_table)
)
(
    spark.createDataFrame(per_row_long)
    .write.mode("append").option("mergeSchema", "true")
    .saveAsTable(perrow_table)
)
if not fairness_df.empty:
    (
        spark.createDataFrame(
            fairness_df.assign(
                run_id=run_id,
                computed_at=pd.Timestamp.utcnow(),
            )
        )
        .write.mode("append").option("mergeSchema", "true")
        .saveAsTable(fairness_table)
    )

# Log to the MLflow run so the model card surfaces both SHAP + fairness
with mlflow.start_run(run_id=run_id):
    mlflow.log_dict(
        global_importance.head(50).to_dict(orient="records"),
        "explanations/global_shap_importance.json",
    )
    if not fairness_df.empty:
        mlflow.log_dict(
            fairness_df.to_dict(orient="records"),
            "fairness/subgroup_metrics.json",
        )
    mlflow.log_metric("worst_disparate_impact_ratio", worst if not np.isnan(worst) else -1)
    mlflow.set_tag("fairness_check", "Passed" if is_fair else "Failed")
    mlflow.set_tag("explained_with_shap", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Hand off downstream task values
# MAGIC
# MAGIC Downstream tasks (validate_model, approval_gate) can branch on these.

# COMMAND ----------

dbutils.jobs.taskValues.set(key="fairness_passed", value=str(is_fair).lower())
dbutils.jobs.taskValues.set(key="worst_disparate_impact_ratio", value=f"{worst:.4f}")
dbutils.jobs.taskValues.set(key="explained_run_id", value=run_id)

# Hard gate: halt the DAG before registration if fairness fails. The
# disparate-impact threshold above is a policy choice — tune per the bank's
# adverse-action review framework. Halting here means no biased model
# version is ever created in UC.
if not is_fair:
    raise Exception(
        f"FAIRNESS GATE FAILED for run {run_id}: "
        f"worst disparate-impact ratio = {worst:.3f} > 1.25. "
        f"See {fairness_table} for the per-slice metrics. "
        f"Registration will NOT proceed."
    )

print(
    f"Fairness PASSED (worst DI ratio = {worst:.3f}). "
    f"Wrote {importance_table}, {perrow_table}, {fairness_table}. "
    f"Run-level tag fairness_check=Passed on MLflow run {run_id}."
)

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [03_model_registration]($./03_model_registration) — registers the
# MAGIC model in UC and copies SHAP/fairness results onto the model version tags.
