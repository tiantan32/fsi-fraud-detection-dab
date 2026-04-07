# FSI Fraud Detection — Databricks Asset Bundle

End-to-end fraud detection demo for banking transactions, deployed via Databricks Asset Bundles (DABs).

## Quick Start

```bash
# Install Databricks CLI if needed
pip install databricks-cli

# Configure authentication
databricks configure --token

# Validate the bundle
databricks bundle validate -t dev

# Deploy to dev
databricks bundle deploy -t dev

# Run the initialization job
databricks bundle run fsi_fraud_init -t dev
```

## Architecture

```
Raw Data (UC Volumes) → SDP Pipeline (Bronze/Silver/Gold)
                          ↓
                        Feature Store (offline + on-demand functions)
                          ↓
                        Model Training (Optuna HPO: LightGBM/XGBoost/LR)
                          ↓
                        Register to UC (@Challenger)
                          ↓
                        Automated Validation (5 quality gates)
                          ↓
                        Promote to @Champion
                          ↓
                     ┌────┴────┐
                     ↓         ↓
              Batch Inference  Real-time Serving (endpoint + A/B)
                     ↓
              Lakehouse Monitor (custom fraud metrics)
                     ↓
              Drift Detection → [if violations] → Retrain (closed loop)
                     ↓
              GenAI (Fraud Reports + Agent)
```

## Project Structure

```
fsi-fraud-detection-dab/
├── databricks.yml                          # Bundle config (variables, targets)
├── resources/
│   ├── fsi_fraud_pipeline.yml              # SDP pipeline definition
│   └── fsi_fraud_job.yml                   # Workflow job definition
├── src/
│   ├── config.py                           # Parameterized config
│   ├── _resources/00-setup.py              # Self-contained data setup
│   ├── 00-FSI-fraud-detection-introduction-lakehouse.sql
│   ├── 01-Data-ingestion/
│   │   ├── 01.1-sdp-sql/                  # SQL SDP (reference)
│   │   └── 01.2-sdp-python/               # Python SDP (primary, parameterized)
│   ├── 02-Data-governance/                 # Unity Catalog ACLs
│   ├── 03-BI-data-warehousing/             # DBSQL analytics
│   ├── 04-Data-Science-ML/                 # Full MLOps lifecycle (8 notebooks)
│   ├── 05-Generative-AI/                   # AI Functions + Agent
│   └── 06-Workflow-orchestration/          # Orchestration guide
├── dashboards/
│   └── fraud-detection.lvdash.json         # Lakeview dashboard
└── README.md
```

## Targets

| Target    | Catalog | Schema                                    | Mode        |
|-----------|---------|-------------------------------------------|-------------|
| **dev**   | main    | `dbdemos_fsi_fraud_detection_dev_<user>`   | development |
| **staging** | main  | `dbdemos_fsi_fraud_detection_staging`      | default     |
| **prod**  | main    | `dbdemos_fsi_fraud_detection`              | production  |

## Key Changes from dbdemos Version

1. **Self-contained**: No dependency on `00-global-setup-v2.py` — setup is standalone
2. **Parameterized**: All notebooks read catalog/schema from job parameters or widgets
3. **DABs resources**: Pipeline and job defined as YAML, not Python dicts
4. **Multi-environment**: dev/staging/prod targets with variable substitution
5. **Python SDP primary**: Volume paths dynamically constructed from `spark.conf` pipeline variables
6. **SQL SDP reference**: Uses pipeline configuration variable substitution (`${catalog_name}`, etc.)

## Jobs

### `fsi_fraud_init` — Full MLOps Setup Pipeline

| Task | Notebook | Description |
|------|----------|-------------|
| `init_data` | `_resources/00-setup.py` | Download PaySim data, create UC volumes |
| `start_sdp_pipeline` | (pipeline) | SDP: Bronze → Silver → Gold |
| `feature_engineering` | `01_feature_engineering.py` | Feature Store + on-demand functions |
| `model_training` | `02_model_training_hpo.py` | Optuna HPO (LightGBM/XGBoost/LR) |
| `register_model` | `03_model_registration.py` | Register to UC as @Challenger |
| `validate_model` | `04_challenger_validation.py` | 5 quality gates → promote to @Champion |
| `batch_inference` | `05_batch_inference.py` | Score transactions, save predictions |
| `deploy_serving` | `06_realtime_serving.py` | Model Serving endpoint + auto-capture |
| `setup_monitoring` | `07_model_monitoring.py` | Lakehouse Monitor with custom fraud metrics |
| `create_ai_functions` | `05.1-AI-Functions-Creation.py` | GenAI fraud report functions |

### `fsi_fraud_monitor_retrain` — Scheduled Monitoring & Retraining

| Task | Description |
|------|-------------|
| `refresh_pipeline` | Ingest new transaction data via SDP |
| `batch_inference` | Score new data with Champion model |
| `drift_detection` | Check drift metrics and performance |
| `check_violations` | **Condition task**: violations > 0? |
| `retrain_model` | *(conditional)* Retrain with Optuna HPO |
| `reregister_model` | *(conditional)* Register new Challenger |
| `revalidate_model` | *(conditional)* Validate and promote |

### MLOps Quality Gates (04_challenger_validation)

1. **Description check** — Model version has documentation
2. **Artifact check** — Model artifacts downloadable
3. **Prediction check** — Model scores via Feature Store
4. **F1 metric check** — Challenger >= Champion
5. **Business metric check** — Challenger catches more fraud (dollar value)

## Dataset

Synthetic banking transactions from [PaySim](https://github.com/EdgarLopezPhD/PaySim):
- Transactions (JSON), Customers (CSV), Country codes (CSV), Fraud reports (CSV)
- ~3% fraud rate, ~9% by transaction amount
- 23 features including geographic, temporal, and balance data
