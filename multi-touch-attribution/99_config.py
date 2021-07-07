# Databricks notebook source
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user') # user name associated with your account
project_directory = '/home/{}/multi-touch-attribution'.format(user) # files will be written to this directory
database_name = 'multi_touch_attribution' # tables will be stored in this database

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"project_directory":project_directory,"database_name":database_name}))
