# Databricks notebook source
import dlt
import pyspark.sql.functions as F
from pyspark.sql.types import *
import json
import pandas as pd
import uuid
import random
from random import randrange
from datetime import datetime, timedelta 
from pyspark.sql.window import Window
from pyspark.sql.types import *

# COMMAND ----------

raw_data_path = "dbfs:/home/christos.erotocritou@databricks.com/multi-touch-attribution/raw"

# COMMAND ----------

@dlt.table(
  name="raw"
)
def load_raw_data():
  return (spark.read.option("header", "true").csv(raw_data_path))

# COMMAND ----------

@dlt.table(
  name="bronze"
)
def bronze_data():
  return (dlt.read("raw")
             .withColumn("time", F.to_timestamp(F.col("time"),"yyyy-MM-dd HH:mm:ss"))
             .withColumn("conversion", F.col("conversion").cast("int")))

# COMMAND ----------

@dlt.table(
  name="gold_user_journey",
  table_properties={"pipelines.autoOptimize.zOrderCols" : "uid"},
)
def generate_user_journey():
  windowSpec  = Window.partitionBy("uid").orderBy(F.col("time").asc())
  sub = dlt.read("bronze").select("uid", "channel", "time", "conversion", F.dense_rank().over(windowSpec).alias("visit_order"))
  sub2 = sub.select("uid", 
                    F.concat_ws(' > ', F.collect_list("channel").over(windowSpec)).alias("path"),
                    F.element_at(F.collect_list("channel").over(windowSpec), 1).alias("first_interaction"),
                    F.element_at(F.collect_list("channel").over(windowSpec), -1).alias("last_interaction"),
                    F.element_at(F.collect_list("conversion").over(windowSpec), -1).alias("conversion"),
                    F.collect_list(sub.visit_order).over(windowSpec).alias("visiting_order"))
  user_journey_view = sub2.select("uid", "first_interaction", "last_interaction", "conversion", "visiting_order", "path")
  user_journey_view = user_journey_view.withColumn(
    "path", 
    F.when(user_journey_view.conversion == 1, F.concat(F.lit("Start > "), F.col("path"), F.lit(" > Conversion")))
     .otherwise(F.concat(F.lit("Start"), F.col("path"), F.lit(" > Null"))))
  return user_journey_view

# COMMAND ----------

@dlt.table(
  name="gold_attribution"
)
def generate_gold_attribution():
  return(spark.sql('''
  SELECT
    'first_touch' AS attribution_model,
    first_interaction AS channel,
    round(count(*) / (
       SELECT COUNT(*)
       FROM multi_touch_attribution.gold_user_journey
       WHERE conversion = 1),2) AS attribution_percent
  FROM multi_touch_attribution.gold_user_journey
  WHERE conversion = 1
  GROUP BY first_interaction
  UNION
  SELECT
    'last_touch' AS attribution_model,
    last_interaction AS channel,
    round(count(*) /(
        SELECT COUNT(*)
        FROM multi_touch_attribution.gold_user_journey
        WHERE conversion = 1),2) AS attribution_percent
  FROM multi_touch_attribution.gold_user_journey
  WHERE conversion = 1
  GROUP BY last_interaction
  '''))
