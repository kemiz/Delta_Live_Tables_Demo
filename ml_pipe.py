# Databricks notebook source
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

@dlt.table(
  comment="The raw products dataset, ingested from /databricks-datasets.",
  name="Products"
)
def products_raw():
  return (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .option("delimiter", ";")
       .csv("/databricks-datasets/retail-org/products/products.csv"))
