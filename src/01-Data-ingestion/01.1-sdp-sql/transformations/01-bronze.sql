-- ----------------------------------
-- Bronze layer: Ingest raw data from Unity Catalog volumes
-- NOTE: Volume paths below must match the catalog/schema set on the SDP pipeline.
--       When changing targets (dev/staging/prod), update these paths to match
--       the target's catalog and schema, or use the Python SDP alternative.
-- ----------------------------------

CREATE STREAMING TABLE bronze_transactions
COMMENT "Historical banking transaction to be trained on fraud detection"
AS
SELECT *
FROM STREAM read_files(
  '/Volumes/${catalog_name}/${schema_name}/${volume_name}/transactions',
  format => 'json',
  maxFilesPerTrigger => 1,
  inferColumnTypes => true
);

CREATE STREAMING TABLE banking_customers (
  CONSTRAINT correct_schema EXPECT (_rescued_data IS NULL)
)
COMMENT "Customer data coming from csv files ingested in incremental with Auto Loader to support schema inference and evolution"
AS
SELECT *
FROM STREAM read_files(
  '/Volumes/${catalog_name}/${schema_name}/${volume_name}/customers',
  format => 'csv',
  multiLine => true,
  inferColumnTypes => true
);

CREATE STREAMING TABLE country_coordinates
AS
SELECT *
FROM STREAM read_files(
  '/Volumes/${catalog_name}/${schema_name}/${volume_name}/country_code',
  format => 'csv'
);

CREATE STREAMING TABLE fraud_reports
AS
SELECT *
FROM STREAM read_files(
  '/Volumes/${catalog_name}/${schema_name}/${volume_name}/fraud_report',
  format => 'csv'
);
