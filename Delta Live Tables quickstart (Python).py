# Databricks notebook source
# MAGIC %md # Delta Live Tables quickstart (Python)
# MAGIC 
# MAGIC A notebook that provides an example Delta Live Tables pipeline to:
# MAGIC 
# MAGIC - Read raw JSON clickstream data into a table.
# MAGIC - Read records from the raw data table and use a Delta Live Tables query and expectations to create a new table with cleaned and prepared data.
# MAGIC - Perform an analysis on the prepared data with a Delta Live Tables query.

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/amazon/users

# COMMAND ----------

df = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .option("delimiter", ",")
       .csv("/databricks-datasets/amazon/users/part-r-00000-f8d9888b-ba9e-47bb-9501-a877f2574b3c.csv"))
        
display(df)

# COMMAND ----------

customers_df = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .option("delimiter", ",")
       .csv("/databricks-datasets/retail-org/customers/customers.csv"))
        
display(customers_df)

# COMMAND ----------

products_df = (spark.read
       .option("header", True)
       .option("inferSchema", True)
       .option("delimiter", ";")
       .csv("/databricks-datasets/retail-org/products/products.csv"))
        
display(products_df)

# COMMAND ----------

sales_stream_df = (spark.read
       .json("/databricks-datasets/retail-org/sales_stream/sales_stream.json"))
        
display(sales_stream_df)

# COMMAND ----------

import pyspark.sql.functions as F

line_items_df = (sales_stream_df
                   .select(sales_stream_df.customer_id,
                           sales_stream_df.customer_name,
                           sales_stream_df.order_number, 
                           sales_stream_df.order_datetime, 
                           F.explode(sales_stream_df.ordered_products).alias("line_item"))
                )

line_items_df = line_items_df.withColumn('product_id', F.col('line_item').getItem(1)).withColumn('quantity', F.col('line_item').getItem(4)).drop("line_item").orderBy('customer_id')
display(line_items_df)

# COMMAND ----------

df = sales_stream_df.select("customer_id", "ordered_products")
display(df)

# COMMAND ----------

display(df.orderBy("customer_id"))

# COMMAND ----------

display(line_items_df.join(products_df, products_df.product_id == line_items_df.product_id).join(customers_df, customers_df.customer_id == line_items_df.customer_id))

# COMMAND ----------


