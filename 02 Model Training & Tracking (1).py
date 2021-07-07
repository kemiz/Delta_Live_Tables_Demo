# Databricks notebook source
# MAGIC %fs cp "/x" "/f"

# COMMAND ----------

# MAGIC %md # Creating a Customer Segmentation Model
# MAGIC 
# MAGIC In this notebook we will train and fine-tune a customer segmentation model.
# MAGIC 
# MAGIC We will learn:
# MAGIC 
# MAGIC - How to create a classification model using a Random Forest Classifier
# MAGIC - How to use MLflow to track experiment run parameters, metrics, models and artifacts
# MAGIC - How to automatically fine-tune our model hyperparameters
# MAGIC - How to register a model with the Model Registry

# COMMAND ----------

# MAGIC %md ## 2.1 Training a Multi-label Classifier

# COMMAND ----------

spark.sql("USE christos")

# COMMAND ----------

# MAGIC %sql select * from test_data_customers_post_etl_aggr_extended limit 100

# COMMAND ----------

# DBTITLE 1,Setup environment
spark.conf.set('spark.databricks.mlflow.trackHyperopt.enabled', 'false')
spark.conf.set("spark.databricks.mlflow.trackMLlib.enabled", "false")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# Library imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn
from mlflow.tracking.client import MlflowClient
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt import SparkTrials
from hyperopt import space_eval
from sklearn.model_selection import GroupKFold
from pyspark.sql.functions import pandas_udf, PandasUDFType

import os
#import databricks.koalas as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import *

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC <div style="float:right"><img src="https://i.imgur.com/rALRDsk.png" style="height: 230px"/></div>
# MAGIC The objective of our model is simple: Classify customers into different segments.
# MAGIC <br><br>
# MAGIC Infact, customers can be in multiple segments therefore we need to use a multi-label classification model. For this reason we will be chosing RandomForestClassifier as our algorithm and because our dataset is relatively small and fits in memory, we'll be using Scikit-learn for training and inference.
# MAGIC 
# MAGIC MLFlow will help us to track our experiment: **the model itself** (serialized), **the dependencies**, **hyper-parameters**, **model performance**, **code revision** (the notebook version), data used, images etc

# COMMAND ----------

# DBTITLE 1,First we load the customer data from Delta table and convert to Pandas DataFrame
customers = spark.sql("select * from test_data_customers_post_etl_aggr_extended limit 100").toPandas()
delta_version = sql(f"SELECT MAX(version) AS VERSION FROM (DESCRIBE HISTORY christos.test_data_customers_post_etl_aggr_extended)").head()[0]
display(customers)

# COMMAND ----------

# DBTITLE 1,We split the original dataset to get our training and test datasets
def prepare_data(customers):
  brand_and_category_cols = [column for column in customers.columns if column.startswith("category_") or column.startswith("brand_")]
  X = customers[['num_order', 
                 'num_sessions', 
                 'num_clicks', 
                 'num_add2bag', 
                 'num_add2wishlist', 
                 'total_gmv_dollars', 
                 'perct_sales_dollars', 
                 'perct_return_item_level',
                 'perct_return_dollars', 
                 'gender', 
                 'is_vip', 
                 'has_pc'] + brand_and_category_cols]
  y = customers['image_style']
  # Here we split our data into a test set and a training set with a 30/70 split
  return train_test_split(X, y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = prepare_data(customers)

# COMMAND ----------

# DBTITLE 1,Create a Scikit Learn pipeline builder
# Our pipeline consists of a StandardScaler & the Random Forest Estimator
def get_scikit_learn_rf_pipeline(n_estimators = 50, max_features='sqrt', criterion='gini', max_depth=8):
  return Pipeline([
      ('StandardScaler', preprocessing.StandardScaler()),
      ('randomForest', RandomForestClassifier(n_estimators=n_estimators, 
                                              criterion=criterion, 
                                              max_depth=max_depth, 
                                              max_features=max_features))])

# COMMAND ----------

# DBTITLE 1,And now let's go ahead and train our RandomForestClassifier
# Note how we are starting a new MLFlow run and giving it a name. This configures the current 
# instance of the mlfow to associate any tracked parameters, metrics and artifacts to this run 
with mlflow.start_run(run_name='Customer Classification - Single Run') as run:
  import pandas as pd
  from sklearn.pipeline import Pipeline
  print("MLflow:")
  print(f"  run_id: {run.info.run_id}")
  print(f"  experiment_id: {run.info.experiment_id}")
  print(f"  experiment_id: {run.info.artifact_uri}")
  
  n_estimators = 500
  criterion = 'gini'
  max_depth = 16
  max_features = 'sqrt'
  
  # Logging parameters                      
  ####################
  # Here we are recording the parameters relevant to our model training run
  # This can typically include the model hyperparmeters, version of the dataset, size of the
  # and other relevant information for this run
  mlflow.log_param("Algo", 'Random Forest')
  mlflow.log_param("Delta version", delta_version)
  mlflow.log_param("Data size", len(customers.index))
  mlflow.log_param("Number of Trees", n_estimators)
  mlflow.log_param("Max Features", max_features)
  mlflow.log_param("Max Depth", max_depth)
  mlflow.log_param("Criterion", criterion)
  mlflow.log_param("Pickled Model", "True")
    
  # Here we are creating a new scikit pipeline with the following hyperparameters
  pipeline = get_scikit_learn_rf_pipeline(
    n_estimators=n_estimators, 
    criterion=criterion, 
    max_depth=max_depth, 
    max_features=max_features)
  
  # Here we fit our training data to the pipeline and make some predictions using our test set.
  # The test set is used as a benchmark to give us metrics by which we can measure the performance
  pipeline.fit(X_train, y_train)
  predictions = pipeline.predict(X_test)
  probas = pipeline.predict_proba(X_test)

  # Logging metrics                         
  #################
  # At this point we are recording these metrics for this experiment run using mlflow
  mlflow.log_metric("F1", metrics.f1_score(y_test, predictions, average='micro'))
  
  # Logging model
  ###############
  # Here we are infact logging a pickled version of our model. This will allow us to later
  # deploy and use our model for inference
  mlflow.sklearn.log_model(pipeline, "model")
  
  # Logging artifacts
  ###################
  cm = confusion_matrix(y_test, predictions)
  df_cm = pd.DataFrame(cm, index = range(cm.shape[0]), columns = range(cm.shape[0]))
  plt.figure(figsize = (20,5))
  sns.set(font_scale=1.4)
  svm = sns.heatmap(df_cm, annot=True, fmt='g')
  fig = svm.get_figure()
  os.makedirs(f"/dbfs/home/{username}/farfetch/demo/mlflow/classification", exist_ok=True)
  fig.savefig(f"/dbfs/home/{username}/farfetch/demo/mlflow/classification/confusion_matrix.png", dpi=400)
  mlflow.log_artifact(f"/dbfs/home/{username}/farfetch/demo/mlflow/classification/confusion_matrix.png", "model")
  
  mlflow.end_run()

# COMMAND ----------

# MAGIC %md ## 2.2 Automated hyper-parameter optimization

# COMMAND ----------

# DBTITLE 0,2.2 Automated hyper-parameter optimization
# MAGIC %md-sandbox 
# MAGIC <div style="float:right;width:500px"><img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2F3.bp.blogspot.com%2F-kaq-LFU3tQg%2FViFM19wYkuI%2FAAAAAAAAQ0o%2F6QEHauXfw-U%2Fs1600%2Fweightiterations.jpg&f=1&nofb=1
# MAGIC "/></div>
# MAGIC At this point our **model, its dependencies, hyper-parameters, metrics and artifacts are saved in MLFLow and available in the MLFLow side-bar and the UI.**
# MAGIC <br>Because we logged the model as an artifact, we'll be able to use it to deploy it in production later, and track our model performance over time!
# MAGIC <br>In addition to the model, we can save custom artifact in our repository, for example a confusion matrix, or some custom data.
# MAGIC <br><br>
# MAGIC Running a single model training is good. But what if we want to fine tune our model and find the best hyper-parameters? This is called hyperparameter optimization.
# MAGIC <br><br>
# MAGIC In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned.

# COMMAND ----------

# MAGIC %md ### 2.2.1 Sklearn Grid Search Cross-Validation

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC <div style="float:right"><img src="https://i.imgur.com/UhSX78b.png" style="height: 280px; margin-left:20px"/></div>
# MAGIC Grid search is a method to perform hyper-parameter optimisation. In this scenario, you have several models, each with a different combination of hyper-parameters. Each of these combinations of parameters, which correspond to a single model, can be said to lie on a point of a "grid". The goal is then to train each of these models and evaluate them e.g. using cross-validation. You then select the one that performed best.

# COMMAND ----------

# DBTITLE 1,First let's define our parameter search space
parameters = {'randomForest__n_estimators': [x*50 for x in range(1,4)],
              'randomForest__max_features': ['auto', 'sqrt', 'log2'],
              'randomForest__max_depth' : [x for x in range(4, 6)],
              'randomForest__criterion' :['gini', 'entropy'] }

# COMMAND ----------

# DBTITLE 1,Now we can train our model
with mlflow.start_run(run_name="Customer Classification - GridSearchCV") as run:
  from sklearn.pipeline import Pipeline
  #############################################
  # let's train our Scikit Learn Pipeline rf: #
  ############################################# 
  print("MLflow:")
  print(f"  run_id: {run.info.run_id}")
  print(f"  experiment_id: {run.info.experiment_id}")
  print(f"  experiment_id: {run.info.artifact_uri}")
  
  # Build the pipeline
  pipeline = get_scikit_learn_rf_pipeline()  
  
  # This is where we execute our training runs with various hyperparameter combinations
  grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, scoring='f1_micro', cv=5)
  
  search_result = grid_search.fit(X_train, y_train)
  predictions = search_result.best_estimator_.predict(X_test)
  
  # Log the best parameters
  mlflow.log_param("Delta version", delta_version)
  mlflow.log_param("Data size", len(customers.index))
  mlflow.log_param("Algo", 'Random Forest')
  mlflow.log_param("Pickled Model", "True")
  mlflow.log_param('Number of Trees', search_result.best_params_['randomForest__n_estimators'])
  mlflow.log_param('Max Features', search_result.best_params_['randomForest__max_features'])
  mlflow.log_param('Max Depth', search_result.best_params_['randomForest__max_depth'])
  mlflow.log_param('Criterion', search_result.best_params_['randomForest__criterion'])
  
  #log the metrics
  mlflow.log_metric("F1", search_result.best_score_)
  
  #log the best model
  mlflow.sklearn.log_model(search_result.best_estimator_, "model")

# COMMAND ----------

# MAGIC %md ### 2.2.2 HyperOpt (Bayesian optimization)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/bayesian-model.png" style="height: 330px"/></div>
# MAGIC 
# MAGIC Scikit-learn GridSearch is great, but not very efficient when the parameter dimension increase and the model is getting slow to train due to a massive amount of data.
# MAGIC 
# MAGIC HyperOpt search accross your parameter space for the minimum loss of your model, using Bayesian optimization instead of a random walk

# COMMAND ----------

# DBTITLE 1,Similar to GridSearch we need to define our search space
param_hyperopt = {
  'randomForest__n_estimators': hp.choice('randomForest__n_estimators', [x for x in range(50, 1000, 10)]),
  'randomForest__max_features': hp.choice('randomForest__max_features', ['auto', 'sqrt', 'log2']),
  'randomForest__max_depth': hp.choice('randomForest__max_depth', [x for x in range(32, 256)]),
  'randomForest__criterion': hp.choice('randomForest__criterion', ['gini', 'entropy'])
}

# COMMAND ----------

# DBTITLE 1,We then need to define a function which we will use with our minimiser function
# This function gets called with a different set of parameters in each run
def train_model_hyperopt(param_hyperopt): 
  from sklearn.pipeline import Pipeline
  import mlflow
  # Build the pipeline
  pipeline = get_scikit_learn_rf_pipeline(
    n_estimators=param_hyperopt['randomForest__n_estimators'], 
    criterion=param_hyperopt['randomForest__criterion'], 
    max_depth=param_hyperopt['randomForest__max_depth'], 
    max_features=param_hyperopt['randomForest__max_features'])
    
  # Fit our pipeline and make some predictions using our test set
  pipeline.fit(X_train, y_train)
  predictions = pipeline.predict(X_test)
  
  f1 = metrics.f1_score(y_test, predictions, average='micro')
  
  # This is where we return the parameters needed by the optimiser but also any
  # model hyperparameters and metrics that we can use to log into mlflow
  return {
    'loss': -f1,
    'status': STATUS_OK, 
    'model': pipeline, 
    'param_hyperopt': param_hyperopt}

# COMMAND ----------

# DBTITLE 1,Helper methods for logging runs
def log_best_run(trials):
  with mlflow.start_run(run_name="Customer Classification - HyperOpt") as run:
    mlflow.log_param("Pickled Model", "True")
    log_run(trials.best_trial)
    mlflow.sklearn.log_model(trials.best_trial['result']['model'], "model")   
    mlflow.end_run()

def log_all_runs(trials):
  for trial in trials:
    with mlflow.start_run(run_name="Customer Classification - HyperOpt") as run:
      log_run(trial) 
      mlflow.end_run()

def log_run(trial):
    mlflow.log_param("Delta version", delta_version)
    mlflow.log_param("Data size", len(customers.index))
    mlflow.log_param("Algo", 'Random Forest')
    mlflow.log_param('Number of Trees', trial['result']['param_hyperopt']['randomForest__n_estimators'])
    mlflow.log_param('Max Features', trial['result']['param_hyperopt']['randomForest__max_features'])
    mlflow.log_param('Max Depth', trial['result']['param_hyperopt']['randomForest__max_depth'])
    mlflow.log_param('Criterion', trial['result']['param_hyperopt']['randomForest__criterion'])
    mlflow.log_metric("F1", trial['result']['loss']*-1)

# COMMAND ----------

# DBTITLE 1,Local HyperOpt Training
from sklearn.pipeline import Pipeline

trials = Trials()
# fmin will optimize our function against the loss it returns
fmin(train_model_hyperopt, 
     param_hyperopt, 
     algo=tpe.suggest, 
     max_evals=16, 
     show_progressbar=True, 
     trials=trials)

log_all_runs(trials)
log_best_run(trials)

# COMMAND ----------

# DBTITLE 1,Distributed HyperOpt Training
from sklearn.pipeline import Pipeline

spark_trials = SparkTrials(parallelism=16)
# Start the training
fmin(train_model_hyperopt, 
     param_hyperopt, 
     algo=tpe.suggest, 
     max_evals=16, 
     show_progressbar=True, 
     trials=spark_trials)

log_all_runs(spark_trials)
log_best_run(spark_trials)

# COMMAND ----------

# MAGIC %md ## 2.3 Registering a model

# COMMAND ----------

# MAGIC %md 
# MAGIC Now that we have trained several models let's pick the best one based on the 'f1' metric and register it as a new version for staging. When we register a model we are able to track the incremental model versions as we push changes.
# MAGIC 
# MAGIC An MLflow Model is a standard format for packaging machine learning models that can be used in a variety of downstream tools—for example, real-time serving through a REST API or batch inference on Apache Spark. The format defines a convention that lets you save a model in different “flavors” that can be understood by different downstream

# COMMAND ----------

# DBTITLE 1,Query the MLflow API for the best run for this experiment
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import mlflow.sklearn

exp_id = mlflow.get_experiment_by_name('/Users/christos.erotocritou@databricks.com/Demos/MLflow/Customer Segmentation/02 Model Training & Tracking').experiment_id

best_run = MlflowClient().search_runs(
  experiment_ids=exp_id,
  filter_string="params.`Pickled Model` = 'True'",
  run_view_type=ViewType.ACTIVE_ONLY,
  max_results=1,
  order_by=["metrics.F1 DESC"]
)[0]

best_run

# COMMAND ----------

# DBTITLE 1,Finally register the model
result = mlflow.register_model(
    best_run.info.artifact_uri + '/model',
    "Customer-Segmentation-Model"
)

# COMMAND ----------

client = MlflowClient()
models = client.search_model_versions("name='Customer-Segmentation-Model'")
print(models)
