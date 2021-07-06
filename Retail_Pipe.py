# Databricks notebook source
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Create Raw Products Table
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

# COMMAND ----------

# DBTITLE 1,Create Raw Customers Table
@dlt.table(
  comment="The raw customers dataset, ingested from /databricks-datasets.",
  name="Customers"
)
def customers_raw():
  return (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .option("delimiter", ",")
       .csv("/databricks-datasets/retail-org/customers/customers.csv"))

# COMMAND ----------

# DBTITLE 1,Create Raw Sales Table
@dlt.table(
  comment="The raw sales stream dataset, ingested from /databricks-datasets.",
  name="Sales"
)
def sales_raw():
  return (spark.read.json("/databricks-datasets/retail-org/sales_stream/sales_stream.json"))

# COMMAND ----------

# DBTITLE 1,Create Silver Sales Table
import pyspark.sql.functions as F

@dlt.table(
  comment="The raw customers dataset, ingested from /databricks-datasets.",
  name="Sales_Filtered"
)
@dlt.expect("order quantity", "quantity > 2")
def sales_raw():
  sales_stream_df = dlt.read("Sales")
  line_items_df = (sales_stream_df
                     .select(sales_stream_df.customer_id,
                             sales_stream_df.order_number, 
                             sales_stream_df.order_datetime, 
                             F.explode(sales_stream_df.ordered_products).alias("line_item"))
                  )
  return (line_items_df.withColumn('product_id', F.col('line_item').getItem(1)).withColumn('quantity', F.col('line_item').getItem(4)).drop("line_item"))

# COMMAND ----------

@dlt.table(
  comment="The raw customers dataset, ingested from /databricks-datasets.",
  name="Sales_Joined"
)
def sales_joined():
  return dlt.read("Sales_Filtered").join(dlt.read("Customers"), ["customer_id"], "left")

# COMMAND ----------

@dlt.table(
  comment="The raw customers dataset, ingested from /databricks-datasets.",
  name="Sales_Products_Joined"
)
def sales_joined():
  return dlt.read("Sales_Joined").join(dlt.read("Products"), ["product_id"], "left")
