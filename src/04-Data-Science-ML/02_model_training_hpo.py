# Databricks notebook source
# MAGIC %md
# MAGIC # HPO Model Training using Optuna & MLflow
# MAGIC
# MAGIC Train fraud detection models with:
# MAGIC - Feature Store lookups with point-in-time semantics
# MAGIC - On-demand feature functions (transaction_risk_score)
# MAGIC - Optuna HPO across LightGBM, XGBoost, and LogisticRegression
# MAGIC - MLflow parent/child run structure for experiment tracking

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow --upgrade lightgbm xgboost optuna
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define Feature Lookups and On-Demand Functions

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureFunction, FeatureLookup

fe = FeatureEngineeringClient()

feature_table_name = f"{catalog}.{db}.fraud_feature_table"
label_table_name = f"{catalog}.{db}.fraud_label_table"

feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Training Set from Feature Store

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, last, max

labels_df = spark.table(label_table_name)
label_col = "is_fraud"

# Create training set specifications
training_set_specs = fe.create_training_set(
    df=labels_df,
    label=label_col,
    feature_lookups=feature_lookups,
    exclude_columns=["transaction_id", "event_ts"],
    exclude_null_labels=True,
)

# Load as pandas, split by the split column
training_pdf = training_set_specs.load_df().filter("split == 'train'").drop("split").sample(fraction=0.05, seed=42).toPandas()
test_pdf = training_set_specs.load_df().filter("split == 'test'").drop("split").sample(fraction=0.05, seed=42).toPandas()

X_train, Y_train = (training_pdf.drop(label_col, axis=1), training_pdf[label_col])
X_test, Y_test = (test_pdf.drop(label_col, axis=1), test_pdf[label_col])

print(f"Training: {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} rows, {X_test.shape[1]} features")
print(f"Fraud rate (train): {Y_train.mean():.3%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define Preprocessors

# COMMAND ----------

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

# Identify column types
categorical_cols = [c for c in X_train.columns if X_train[c].dtype == 'object' or c in ['type', 'countryOrig', 'countryDest', 'country']]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols and pd.api.types.is_numeric_dtype(X_train[c])]

print(f"Numeric: {numeric_cols}")
print(f"Categorical: {categorical_cols}")

# Numerical pipeline
num_imputers = [("impute_mean", SimpleImputer(), numeric_cols)]
numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])
numerical_transformers = [("numerical", numerical_pipeline, numeric_cols)]

# Categorical pipeline
one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer([], remainder="passthrough")),
    ("one_hot_encoder", SklearnOneHotEncoder(handle_unknown="ignore")),
])
categorical_transformers = [("onehot", one_hot_pipeline, categorical_cols)]

transformers = numerical_transformers + categorical_transformers
preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Optuna Objective Function

# COMMAND ----------

import optuna
import mlflow
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class ObjectiveOptuna(object):
    def __init__(self, X_train_in, Y_train_in, preprocessor_in, rng_seed=2025):
        self.preprocessor = preprocessor_in
        self.rng_seed = rng_seed
        X_tr, X_val, Y_tr, Y_val = train_test_split(X_train_in, Y_train_in, test_size=0.1, random_state=rng_seed)
        self.X_train = X_tr
        self.Y_train = Y_tr
        self.X_val = X_val
        self.Y_val = Y_val

    def __call__(self, trial):
        classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "LightGBM", "XGBoost"])

        if classifier_name == "LogisticRegression":
            lr_C = trial.suggest_float("C", 1e-2, 1, log=True)
            lr_tol = trial.suggest_float("tol", 1e-6, 1e-3, step=1e-6)
            classifier_obj = LogisticRegression(C=lr_C, tol=lr_tol, random_state=self.rng_seed)

        elif classifier_name == "LightGBM":
            n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.9)
            max_bin = trial.suggest_int("max_bin", 2, 256)
            num_leaves = trial.suggest_int("num_leaves", 2, 256)
            classifier_obj = LGBMClassifier(
                force_row_wise=True, verbose=-1,
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, max_bin=max_bin,
                num_leaves=num_leaves, random_state=self.rng_seed,
            )

        elif classifier_name == "XGBoost":
            n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.9)
            classifier_obj = XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, verbosity=0,
                use_label_encoder=False, eval_metric="logloss",
                random_state=self.rng_seed,
            )

        # Assemble pipeline: preprocessor + classifier
        this_model = Pipeline(steps=[("preprocessor", self.preprocessor), ("classifier", classifier_obj)])

        mlflow.sklearn.autolog(disable=True)
        this_model.fit(self.X_train, self.Y_train)

        y_val_pred = this_model.predict(self.X_val)
        return f1_score(self.Y_val, y_val_pred, average="binary", pos_label=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Setup experiment and run HPO

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

current_user = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{current_user}/fsi-fraud-detection/fraud_model_hpo"

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
try:
    w.workspace.mkdirs(f"/Users/{current_user}/fsi-fraud-detection")
except:
    pass

try:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
except:
    experiment_id = mlflow.create_experiment(name=experiment_name)

mlflow.set_experiment(experiment_id=experiment_id)
print(f"Experiment: {experiment_name} (id={experiment_id})")

client = MlflowClient()

# COMMAND ----------

optuna_sampler = optuna.samplers.TPESampler(seed=2025)

from optuna.pruners import BasePruner

class NoneValuePruner(BasePruner):
    def prune(self, study, trial):
        return trial.value is None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Main training function

# COMMAND ----------

from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc


def optuna_hpo_fn(n_trials, X_train, Y_train, X_test, Y_test, training_set_specs_in, preprocessor_in, experiment_id, rng_seed_in=2025, run_name="mlops-hpo-best-run", n_jobs=1):

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as parent_run:

        # Callback to log each trial as a nested run
        def mlflow_callback(study, trial):
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("f1_score", trial.value)
                mlflow.log_metric("trial_number", trial.number)
                mlflow.set_tag("trial_state", trial.state.name)

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna_sampler,
            pruner=NoneValuePruner(),
            study_name=run_name,
        )

        # Distributed HPO strategy
        # ------------------------
        # On SERVERLESS compute (V5+), there are no Spark executors that joblib
        # can fan trials out to — serverless is a single-driver runtime by
        # design. We therefore use driver-side multi-process parallelism
        # (n_jobs=-1 fans across the driver's cores).
        #
        # On CLASSIC compute with a multi-node Spark cluster, registering
        # joblib-spark would fan each Optuna trial out as a Spark task — but
        # serverless's immutable-package-constraints.txt rejects joblib-spark
        # at install time, so we can't even import it conditionally. The
        # canonical "distributed HPO" pattern for Databricks on serverless is
        # Ray on Databricks (see follow-ups in README).
        objective_fn = ObjectiveOptuna(X_train, Y_train, preprocessor_in, rng_seed_in)
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[mlflow_callback],
        )
        mlflow.set_tag("hpo_backend", "driver-multiprocess")
        print(f"HPO ran with driver-side parallelism (n_jobs={n_jobs}, {n_trials} trials).")

        # Log best trial info to parent run
        mlflow.log_params({
            "best_trial_number": study.best_trial.number,
            "best_classifier": study.best_params.get("classifier", "unknown"),
        })
        mlflow.log_metric("best_f1_score", study.best_value)

    # Extract best params and reproduce best classifier
    best_model_params = study.best_params.copy()
    best_model_params["random_state"] = rng_seed_in
    classifier_type = best_model_params.pop("classifier")

    if classifier_type == "LogisticRegression":
        best_model = LogisticRegression(**best_model_params)
    elif classifier_type == "LightGBM":
        best_model = LGBMClassifier(force_row_wise=True, verbose=-1, **best_model_params)
    elif classifier_type == "XGBoost":
        best_model = XGBClassifier(verbosity=0, use_label_encoder=False, eval_metric="logloss", **best_model_params)

    # Enable autolog for final model
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False, silent=True)

    run_id = parent_run.info.run_id

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        # Build final pipeline and fit
        model_pipeline = Pipeline(steps=[("preprocessor", objective_fn.preprocessor), ("classifier", best_model)])
        model_pipeline.fit(X_train, Y_train)

        mlflow.log_input(mlflow.data.from_pandas(X_train), context="training")

        # Evaluate model
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model_pipeline)

        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(label_col): Y_train}),
            targets=label_col,
            model_type="classifier",
            evaluator_config={"log_model_explainability": False, "metric_prefix": "training_", "pos_label": 1},
        )

        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(label_col): Y_test}),
            targets=label_col,
            model_type="classifier",
            evaluator_config={"log_model_explainability": True, "metric_prefix": "test_", "pos_label": 1},
        )

        # Log model with Feature Store for serving-time feature lookups.
        # Use cloudpickle: skops (the new mlflow.sklearn default) refuses to
        # round-trip pipelines containing lambdas / xgboost.Booster as
        # "untrusted types", which breaks log_model on this preprocessor.
        # Pin pip_requirements explicitly so the serving image build doesn't
        # try to resolve against DBR's bleeding-edge pandas/numpy.
        # NOTE: do NOT include databricks-feature-engineering — fe.log_model
        # auto-injects databricks-feature-lookup and the two conflict.
        import sklearn, lightgbm, xgboost, cloudpickle  # noqa: F401
        pip_requirements = [
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"lightgbm=={lightgbm.__version__}",
            f"xgboost=={xgboost.__version__}",
            "numpy>=1.26,<2",
            "pandas>=2.1,<3",
            "cloudpickle",
        ]
        fe.log_model(
            model=model_pipeline,
            artifact_path="model",
            flavor=mlflow.sklearn,
            training_set=training_set_specs_in,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=pip_requirements,
        )
        # ALSO log the raw sklearn pipeline at a separate artifact path so the
        # explainability notebook can load it as sklearn flavor and introspect
        # the fitted classifier for SHAP. fe.log_model wraps the model in a
        # feature-store pyfunc, which hides the sklearn flavor and breaks
        # `mlflow.sklearn.load_model("runs:/<id>/model")`. The "raw_model"
        # artifact is ONLY for explainability — never served, never promoted.
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="raw_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=pip_requirements,
        )
        mlflow.end_run()

    return study

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Execute training

# COMMAND ----------

distributed_study = optuna_hpo_fn(
    n_trials=8,
    X_train=X_train,
    X_test=X_test,
    Y_train=Y_train,
    Y_test=Y_test,
    training_set_specs_in=training_set_specs,
    preprocessor_in=preprocessor,
    experiment_id=experiment_id,
    rng_seed_in=2025,
    run_name="fraud-hpo-best-run",
    n_jobs=-1,
)

print(f"\nBest trial F1: {distributed_study.best_value:.4f}")
print(f"Best params: {distributed_study.best_params}")

# COMMAND ----------

# Save run ID for downstream notebooks
best_run = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="run_name = 'fraud-hpo-best-run'",
    order_by=["start_time DESC"],
    max_results=1,
)
best_run_id = best_run.iloc[0]["run_id"]
dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)
dbutils.jobs.taskValues.set(key="best_f1_score", value=round(distributed_study.best_value, 4))

print(f"Run ID: {best_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC Next: [Register model]($./03_model_registration)
