# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left">
# MAGIC   <img src="https://brysmiwasb.blob.core.windows.net/demos/images/ME_solution-accelerator.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC ### In this lesson you:
# MAGIC * Build a sharable dashboard with end users in the organization to visualize curent state of the campaign and make decisions to optimize spending.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure the Environment
# MAGIC 
# MAGIC In this step, we will:
# MAGIC   1. Import libraries
# MAGIC   2. Run the `utils` notebook to gain access to the functions `get_params`
# MAGIC   3. `get_params` and store values in variables
# MAGIC   4. Set the current database so that it doesn't need to be manually specified each time it's used

# COMMAND ----------

dbutils.widgets.text("adspend", "10000", "Campaign Budget in $")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.1: Import libraries

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.2: Run the `utils` notebook to gain access to the functions `get_params`
# MAGIC * `%run` is a magic command provided within Databricks that enables you to run notebooks from within other notebooks.
# MAGIC * `get_params` is a helper function that returns a few parameters used throughout this solution accelerator. Usage of these parameters will be explicit.

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.3: `get_params` and store values in variables
# MAGIC 
# MAGIC * Three of the parameters returned by `get_params` are used in this notebook. For convenience, we will store the values for these parameters in new variables. 
# MAGIC 
# MAGIC   * **database_name:** the name of the database created in notebook `02_load_data`. The default value can be overridden in the notebook `99_config`
# MAGIC   * **gold_user_journey_tbl_path:** the path used in `03_load_data` to write out gold-level user journey data in delta format.
# MAGIC   * **gold_attribution_tbl_path:** the path used in `03_load_data` to write out gold-level attribution data in delta format.

# COMMAND ----------

params = get_params()
database_name = params['database_name']
gold_user_journey_tbl_path = params['gold_user_journey_tbl_path']
gold_attribution_tbl_path = params['gold_attribution_tbl_path']
gold_ad_spend_tbl_path = params['gold_ad_spend_tbl_path']

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.4: Set the current database so that it doesn't need to be manually specified each time it's used
# MAGIC * Please note that this is a completely optional step. An alternative approach would be to use the syntax `database_name`.`table_name` when querying the respective tables. 

# COMMAND ----------

_ = spark.sql("use {}".format(database_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create and Populate Ad Spend Table

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2.1: Create ad spend table

# COMMAND ----------

_ = spark.sql('''
  CREATE OR REPLACE TABLE gold_ad_spend (
    campaign_id STRING, 
    total_spend_in_dollars FLOAT, 
    channel_spend MAP<STRING, FLOAT>, 
    campaign_start_date TIMESTAMP)
  USING DELTA
  LOCATION '{}'
  '''.format(gold_ad_spend_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2.2: Populate ad spend table

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO TABLE gold_ad_spend
# MAGIC VALUES ("3d65f7e92e81480cac52a20dfdf64d5b", int($adspend),
# MAGIC           MAP('Social Network', .2,
# MAGIC               'Search Engine Marketing', .2,  
# MAGIC               'Google Display Network', .2, 
# MAGIC               'Affiliates', .2, 
# MAGIC               'Email', .2), 
# MAGIC          make_timestamp(2020, 5, 17, 0, 0, 0));

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2.3: View campaign ad spend details

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_ad_spend

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2.4: Explode struct into multiple rows

# COMMAND ----------

ad_spend_df = spark.sql('select explode(channel_spend) as (channel, pct_spend), \
                         round(total_spend_in_dollars * pct_spend, 2) as dollar_spend \
                         from gold_ad_spend')

ad_spend_df.createOrReplaceTempView("exploded_gold_ad_spend")
display(ad_spend_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: View Campaign Performance

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Base Conversion Rate

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE base_conversion_rate
# MAGIC USING DELTA AS
# MAGIC SELECT count(*) as count,
# MAGIC   CASE 
# MAGIC     WHEN conversion == 0 
# MAGIC     THEN 'Impression'
# MAGIC     ELSE 'Conversion'
# MAGIC   END AS interaction_type
# MAGIC FROM
# MAGIC   gold_user_journey
# MAGIC GROUP BY
# MAGIC   conversion;
# MAGIC   
# MAGIC --SELECT * FROM  base_conversion_rate;

# COMMAND ----------

base_converion_rate_pd = spark.table("base_conversion_rate").toPandas()

pie, ax = plt.subplots(figsize=[20,9])
labels = base_converion_rate_pd['interaction_type']
plt.pie(x=base_converion_rate_pd['count'], autopct="%.1f%%", explode=[0.05]*2, labels=labels, pctdistance=0.5)
plt.title("Base Conversion Rate", fontsize=14);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Conversions By Date

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE conversions_by_date 
# MAGIC USING DELTA AS
# MAGIC SELECT count(*) AS count,
# MAGIC   'Conversion' AS interaction_type,
# MAGIC   date(time) AS date
# MAGIC FROM bronze
# MAGIC WHERE conversion = 1
# MAGIC GROUP BY date
# MAGIC ORDER BY date;
# MAGIC 
# MAGIC --SELECT * FROM conversions_by_date;

# COMMAND ----------

conversions_by_date_pd = spark.table("conversions_by_date").toPandas()

plt.figure(figsize=(20,9))
pt = sns.lineplot(x='date',y='count',data=conversions_by_date_pd)
plt.title("Conversions by Date", fontsize=14);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Attribution by model type

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE attribution_by_model_type 
# MAGIC USING DELTA AS
# MAGIC SELECT attribution_model, channel, round(attribution_percent * (
# MAGIC     SELECT count(*) FROM gold_user_journey WHERE conversion = 1)) AS conversions_attributed
# MAGIC FROM gold_attribution;
# MAGIC 
# MAGIC --SELECT * FROM attribution_by_model_type;

# COMMAND ----------

attribution_by_model_type_pd = spark.table("attribution_by_model_type").toPandas()

sns.set(font_scale=1.1) 
pt = sns.catplot(x='channel',y='conversions_attributed',hue='attribution_model',data=attribution_by_model_type_pd, kind='bar', aspect=2)
pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)
plt.title("Channel Performance")
plt.ylabel("Conversions")
plt.xlabel("Channels")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cost per acquisition

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TABLE cpa_summary 
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT
# MAGIC   spending.channel,
# MAGIC   spending.dollar_spend,
# MAGIC   attribution_count.attribution_model,
# MAGIC   attribution_count.conversions_attributed,
# MAGIC   round(spending.dollar_spend / attribution_count.conversions_attributed,2) AS CPA_in_Dollars
# MAGIC FROM
# MAGIC   (SELECT explode(channel_spend) AS (channel, spend),
# MAGIC    round(total_spend_in_dollars * spend, 2) AS dollar_spend
# MAGIC    FROM gold_ad_spend) AS spending
# MAGIC JOIN
# MAGIC   (SELECT attribution_model, channel, round(attribution_percent * (
# MAGIC       SELECT count(*) FROM gold_user_journey WHERE conversion = 1)) AS conversions_attributed
# MAGIC    FROM gold_attribution) AS attribution_count
# MAGIC ON spending.channel = attribution_count.channel;
# MAGIC 
# MAGIC --SELECT * FROM cpa_summary;

# COMMAND ----------

cpa_summary_pd = spark.table("cpa_summary").toPandas()

sns.set(font_scale=1.1)
pt = sns.catplot(x='channel', y='CPA_in_Dollars',hue='attribution_model',data=cpa_summary_pd, kind='bar', aspect=2, ci=None)
plt.title("Cost of Aquisition by Channel")
pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)
plt.title("Channel Cost per Aquisition")
plt.ylabel("CPA in $")
plt.xlabel("Channels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Budget Allocation Optimization.
# MAGIC Based on Markov attribution, we can have a more data-driven approach for Conversion attribution and spend on various channels.
# MAGIC * **One of the KPI we can take a look at is Return on Ad Spend (ROAS)** <br>
# MAGIC ``ROAS = CHANNEL CONVERSION WEIGHT / CHANNEL BUDGET WEIGHT``
# MAGIC   * ROAS value > 1 signifies that the channel has been allocated less budget than warranted by its conversion rate.
# MAGIC   * ROAS value < 1 signifies that the channel has been allocated more budget than warranted by its conversion rate.
# MAGIC   * ROAS value = 1 signifies and optimized budget allocation. 
# MAGIC 
# MAGIC * **From ROAS we can calculate Proposed Budget for each channel** <br>
# MAGIC ``Proposed budget =  Current budget X ROAS``
# MAGIC 
# MAGIC <br>
# MAGIC To calculate ROAS we will join the following Delta Tables:
# MAGIC * **gold_attribution:** This table contains the calculated attribution % per channel based on different attribution models.
# MAGIC * **exploded_gold_ad_spend:** This Table contains the current budget allocated per channel. The column pct_spend documents how much % of the total budget has been allocated to this channel. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM exploded_gold_ad_spend;

# COMMAND ----------

ad_spend_pd = spark.table("exploded_gold_ad_spend").toPandas()

sns.set(font_scale=1.1)
pt = sns.catplot(x='channel', y='dollar_spend', data=ad_spend_pd, kind='bar', aspect=2, ci=None)
pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)
plt.title("Current Budget Allocated to Channels")
plt.ylabel("Budget allocation in $")
plt.xlabel("Channels")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_attribution;

# COMMAND ----------

gold_attrbution_pd = spark.sql("SELECT * FROM gold_attribution where attribution_model='markov_chain'").toPandas()
gold_attrbution_pd['percentage'] = gold_attrbution_pd['attribution_percent']*100

sns.set(font_scale=1.1)
pt = sns.catplot(x='channel', y='percentage', hue='attribution_model', data=gold_attrbution_pd, kind='bar', aspect=2, ci=None)
pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)
plt.title("Calculated Attribution per Channel")
plt.ylabel("Attribution in %")
plt.xlabel("Channels")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from exploded_gold_ad_spend

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE spend_optimization_view 
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT
# MAGIC   a.channel,
# MAGIC   a.pct_spend,
# MAGIC   b.attribution_percent,
# MAGIC   b.attribution_percent / a.pct_spend as ROAS,
# MAGIC   a.dollar_spend,
# MAGIC   round(
# MAGIC     (b.attribution_percent / a.pct_spend) * a.dollar_spend,
# MAGIC     2
# MAGIC   ) as proposed_dollar_spend
# MAGIC FROM
# MAGIC   exploded_gold_ad_spend a
# MAGIC   JOIN gold_attribution b on a.channel = b.channel
# MAGIC   and attribution_model = 'markov_chain';
# MAGIC   
# MAGIC CREATE
# MAGIC OR REPLACE TABLE spend_optimization_final 
# MAGIC USING DELTA AS
# MAGIC SELECT
# MAGIC   channel,
# MAGIC   'current_spending' AS spending,
# MAGIC   dollar_spend as budget
# MAGIC  FROM exploded_gold_ad_spend
# MAGIC UNION
# MAGIC SELECT
# MAGIC   channel,
# MAGIC   'proposed_spending' AS spending,
# MAGIC   proposed_dollar_spend as budget
# MAGIC FROM
# MAGIC   spend_optimization_view;  

# COMMAND ----------

spend_optimization_final_pd = spark.table("spend_optimization_final").toPandas()

sns.set(font_scale=1.1)
pt = sns.catplot(x='channel', y='budget', hue='spending', data=spend_optimization_final_pd, kind='bar', aspect=2, ci=None)

pt.fig.set_figwidth(20)
pt.fig.set_figheight(9)
plt.title("Spend Optimization per Channel")
plt.ylabel("Budget in $")
plt.xlabel("Channels")

# COMMAND ----------


