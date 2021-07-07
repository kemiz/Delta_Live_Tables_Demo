-- Databricks notebook source
CREATE LIVE TABLE raw
AS SELECT * FROM csv.`dbfs:/home/christos.erotocritou@databricks.com/multi-touch-attribution/raw`

-- COMMAND ----------

CREATE LIVE TABLE bronze
AS SELECT *, 
  to_timestamp(time) AS current_page_title,
  CAST(conversion AS INT) AS conversion
FROM live.raw
WITH SCHEMA schema_of_csv(live.raw)

-- COMMAND ----------

CREATE TABLE bronze
AS SELECT *, 
  to_timestamp(time) AS current_page_title,
  CAST(conversion AS INT) AS conversion
FROM (SELECT * FROM csv.`dbfs:/home/christos.erotocritou@databricks.com/multi-touch-attribution/raw`)


-- COMMAND ----------


