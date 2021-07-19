-- Databricks notebook source
-- DBTITLE 1,Create Bronze Table
CREATE LIVE TABLE bronze
AS 
SELECT uid, interaction, channel, to_timestamp("time", "yyyy-MM-dd HH:mm:ss") as time, cast(conversion as int) FROM raw

-- COMMAND ----------

-- DBTITLE 1,Create Gold User Journey Table
CREATE LIVE TABLE gold_user_journey
AS
SELECT sub2.uid AS uid
  ,CASE 
      WHEN sub2.conversion==1 then concat('Start > ', sub2.path, ' > Conversion') 
      ELSE concat('Start > ', sub2.path, ' > Null') 
   END AS path
  , sub2.first_interaction AS first_interaction
  , sub2.last_interaction AS last_interaction
  , sub2.conversion AS conversion
  , sub2.visiting_order AS visiting_order
  FROM
    (
      SELECT sub.uid AS uid
      , concat_ws(' > ', collect_list(sub.channel)) AS path
      , element_at(collect_list(sub.channel), 1) AS first_interaction
      , element_at(collect_list(sub.channel), -1) AS last_interaction
      , element_at(collect_list(sub.conversion), -1) AS conversion
      , collect_list(sub.visit_order) AS visiting_order
      FROM
      (
        SELECT uid
        , channel
        , time
        , conversion
        , dense_rank() OVER (PARTITION BY uid ORDER BY time asc) as visit_order 
        FROM live.bronze
      ) AS sub GROUP BY sub.uid
    ) AS sub2;

-- COMMAND ----------

-- DBTITLE 1,Create Gold Attribution Table
CREATE LIVE TABLE gold_attribution
AS
SELECT
  'first_touch' AS attribution_model,
  first_interaction AS channel,
  round(count(*) / (SELECT COUNT(*) FROM multi_touch_attribution.gold_user_journey WHERE conversion = 1),2) AS attribution_percent
FROM live.gold_user_journey
WHERE conversion = 1
GROUP BY first_interaction
UNION
SELECT
  'last_touch' AS attribution_model,
  last_interaction AS channel,
  round(count(*) /(SELECT COUNT(*) FROM multi_touch_attribution.gold_user_journey WHERE conversion = 1),2) AS attribution_percent
FROM live.gold_user_journey
WHERE conversion = 1
GROUP BY last_interaction
