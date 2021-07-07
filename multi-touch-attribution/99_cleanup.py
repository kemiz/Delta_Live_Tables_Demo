# Databricks notebook source
# MAGIC %md
# MAGIC ##### Step 1.4: Reset Workspace (Optional)
# MAGIC * This is only necessary if you would like to drop artifacts created in this accelerator during a previous run.
# MAGIC * To use this function, change reset to `"True"`

# COMMAND ----------

# MAGIC %run ./99_utils 

# COMMAND ----------

reset = "False"

# COMMAND ----------

if reset == "True":
  reset_workspace("True")
else:
  pass

# COMMAND ----------


