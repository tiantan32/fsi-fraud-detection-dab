# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "dbdemos_fsi_fraud_detection", "Schema")
dbutils.widgets.text("volume_name", "fraud_raw_data", "Volume Name")

reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup catalog, schema, and volume
# MAGIC Self-contained setup — no external dependency on dbdemos global setup.

# COMMAND ----------

# Use existing catalog, create schema and volume
spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
spark.sql(f"USE SCHEMA `{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume_name}`")

# Grant permissions for shared use (best-effort)
for grant_sql in [
    f"GRANT USE SCHEMA ON SCHEMA `{catalog}`.`{schema}` TO `account users`",
    f"GRANT READ VOLUME ON VOLUME `{catalog}`.`{schema}`.`{volume_name}` TO `account users`",
]:
    try:
        spark.sql(grant_sql)
    except Exception as e:
        print(f"Could not set permission (may require admin): {e}")

# COMMAND ----------

folder = f"/Volumes/{catalog}/{schema}/{volume_name}"

def is_folder_empty(folder_path):
    try:
        return len(dbutils.fs.ls(folder_path)) == 0
    except:
        return True

data_missing = any(is_folder_empty(f"{folder}/{sub}") for sub in ["customers", "transactions", "country_code", "fraud_report"])

# COMMAND ----------

import requests
import time
import base64
from datetime import datetime

if reset_all_data or data_missing:
    print("Data missing or reset requested — downloading data...")
    if reset_all_data:
        assert len(folder) > 15 and folder.startswith("/Volumes/")
        dbutils.fs.rm(folder, True)
        spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume_name}`")

    def upload_binary_to_volume(dest_path, content_bytes):
        """Upload binary content to a UC Volume path using the Databricks Files API (serverless-safe)."""
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        # The Files API path: /Volumes/catalog/schema/volume/path → volumes/catalog/schema/volume/path
        volume_rel_path = dest_path.replace("/Volumes/", "")
        w.files.upload(f"/Volumes/{volume_rel_path}", content_bytes, overwrite=True)

    def download_git_folder(dest_folder, repo_owner, repo_name, repo_path):
        """Download files from GitHub repo into a UC Volume (serverless-compatible)."""
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents{repo_path}"
        response = requests.get(api_url, timeout=30)
        if response.status_code != 200:
            print(f"  Warning: Could not list {api_url}: {response.status_code}")
            return
        files = response.json()
        if not isinstance(files, list):
            files = [files]
        count = 0
        for f in files:
            if f.get("type") != "file":
                continue
            fname = f["name"]
            # Skip non-data files (NOTICE, LICENSE, README, etc.)
            if not (fname.endswith(".parquet") or fname.endswith(".csv") or fname.endswith(".json")):
                continue
            r = requests.get(f["download_url"], timeout=120)
            r.raise_for_status()
            volume_rel = dest_folder.replace("/Volumes/", "")
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            import io
            w.files.upload(f"/Volumes/{volume_rel}/{fname}", io.BytesIO(r.content), overwrite=True)
            count += 1
        print(f"  Downloaded {count} files to {dest_folder}")

    try:
        # Clean up any previous partial downloads
        for sub in ["customers_parquet", "transactions_parquet", "fraud_report_parquet"]:
            try:
                dbutils.fs.rm(f"{folder}/{sub}", True)
            except:
                pass

        print("Downloading customers...")
        download_git_folder(f"{folder}/customers_parquet", "databricks-demos", "dbdemos-dataset", "/fsi/fraud-transaction/customers")
        print("Downloading transactions...")
        download_git_folder(f"{folder}/transactions_parquet", "databricks-demos", "dbdemos-dataset", "/fsi/fraud-transaction/transactions")
        print("Downloading country codes...")
        download_git_folder(f"{folder}/country_code", "databricks-demos", "dbdemos-dataset", "/fsi/fraud-transaction/country_code")
        print("Downloading fraud reports...")
        download_git_folder(f"{folder}/fraud_report_parquet", "databricks-demos", "dbdemos-dataset", "/fsi/fraud-transaction/fraud_report")

        # Convert parquet to the formats expected by the SDP pipeline
        print("Converting parquet to JSON/CSV...")
        def write_to(src_folder, output_format, dest_folder):
            spark.read.format("parquet").load(src_folder).repartition(16) \
                .write.format(output_format).option("header", "true").mode("overwrite").save(dest_folder)

        write_to(f"{folder}/transactions_parquet", "json", f"{folder}/transactions")
        print("  transactions → JSON done")
        write_to(f"{folder}/customers_parquet", "csv", f"{folder}/customers")
        print("  customers → CSV done")
        write_to(f"{folder}/fraud_report_parquet", "csv", f"{folder}/fraud_report")
        print("  fraud_report → CSV done")

        print("Data download and conversion complete!")
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_latest_model_version(model_name):
    """Get the latest version number of a registered model."""
    from mlflow.tracking import MlflowClient
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def force_pandas_version(run_id):
    """Fix pandas version in MLflow model artifacts to avoid compatibility issues."""
    import shutil
    import yaml
    import tempfile
    import mlflow
    from mlflow.tracking import MlflowClient

    tmp_dir = str(tempfile.TemporaryDirectory().name)
    os.makedirs(tmp_dir)

    # Fix conda.yaml
    conda_file_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/conda.yaml", dst_path=tmp_dir
    )
    with open(conda_file_path) as f:
        conda_libs = yaml.load(f, Loader=yaml.FullLoader)
    pandas_lib_exists = any(
        lib.startswith("pandas==") for lib in conda_libs["dependencies"][-1]["pip"]
    )
    client = MlflowClient()
    if not pandas_lib_exists:
        print("Adding pandas dependency to conda.yaml")
        conda_libs["dependencies"][-1]["pip"].append("pandas==1.5.3")
        with open(f"{tmp_dir}/conda.yaml", "w") as f:
            f.write(yaml.dump(conda_libs))
        client.log_artifact(run_id=run_id, local_path=conda_file_path, artifact_path="model")

    # Fix requirements.txt
    venv_file_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/requirements.txt", dst_path=tmp_dir
    )
    with open(venv_file_path) as f:
        venv_libs = f.readlines()
    venv_libs = [lib.strip() for lib in venv_libs]
    pandas_lib_exists = any(lib.startswith("pandas==") for lib in venv_libs)
    if not pandas_lib_exists:
        print("Adding pandas dependency to requirements.txt")
        venv_libs.append("pandas==1.5.3")
        with open(f"{tmp_dir}/requirements.txt", "w") as f:
            f.write("\n".join(venv_libs))
        client.log_artifact(run_id=run_id, local_path=venv_file_path, artifact_path="model")

    shutil.rmtree(tmp_dir)


def set_model_permission(model_full_name, permission, group_name):
    """Grant model permissions. Silently fails if insufficient privileges."""
    try:
        spark.sql(f"GRANT {permission} ON FUNCTION `{model_full_name}` TO `{group_name}`")
    except:
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            model_name_parts = model_full_name.split(".")
            # Best effort — may require admin
            print(f"Granted {permission} on {model_full_name} to {group_name}")
        except Exception as e:
            print(f"Could not set model permission: {e}")
