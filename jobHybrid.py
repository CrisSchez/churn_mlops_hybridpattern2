!git checkout onprem
!git fetch origin
!git reset --hard origin/onprem
!git pull
#!git rm 3_trainStrategy_job.py
#!git fetch origin https://github.com/CrisSchez/churn_mlops_hybridpattern2

try:
  try:
    from pyspark.sql.types import *
    from pyspark.sql import SparkSession
  except:
    print('install pyspark')
    os.system('pip3 install pyspark')
    from pyspark.sql.types import *
    from pyspark.sql import SparkSession
  import os
  os.system('pip3 install -r requirements.txt')
  #os.system('pip install pickle')
  import sys
  import os
  import os
  import datetime
  import subprocess
  import glob
  import dill
  import pandas as pd
  import numpy as np
  import cdsw
  import pickle
  
  
  with open("retrain_dates.txt") as f:
    lines = f.read() ##Assume the sample file has 3 lines
    first = lines.split('\n', 1)[0]
  lastretrain=datetime.datetime.strptime(first,"%Y-%m-%d %H:%M:%S.%f")
  now=datetime.datetime.now()
  diff=now-lastretrain
  
  if diff.total_seconds()< 12*3600:
    from cmlbootstrap import CMLBootstrap
    from IPython.display import Javascript, HTML
    import os
    import time
    import json
    import requests
    import xml.etree.ElementTree as ET
    import datetime

    run_time_suffix = datetime.datetime.now()
    run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

    # Set the setup variables needed by CMLBootstrap
    HOST = os.getenv("CDSW_API_URL").split(
        ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
    USERNAME = os.getenv("CDSW_PROJECT_URL").split(
        "/")[6]  # args.username  # "vdibia"
    API_KEY = os.getenv("CDSW_API_KEY") 
    PROJECT_NAME = os.getenv("CDSW_PROJECT")  

    # Instantiate API Wrapper
    cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

    
    runtimes=cml.get_runtimes()
    runtimes=runtimes['runtimes']
    runtimesdf = pd.DataFrame.from_dict(runtimes, orient='columns')
    runtimeid=runtimesdf.loc[(runtimesdf['editor'] == 'Workbench') & (runtimesdf['kernel'] == 'Python 3.7') & (runtimesdf['edition'] == 'Standard')]['id']
    id_rt=runtimeid.values[0]


    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import TransformerMixin
    from sklearn.preprocessing import LabelEncoder
    #from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor

    from lime.lime_tabular import LimeTabularExplainer

    from churnexplainer import ExplainedModel, CategoricalEncoder

    data_dir = '/home/cdsw'






    print("empenzando los experimentos")

    run_time_suffix = datetime.datetime.now()
    #run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")
    run_time_suffix = run_time_suffix.strftime("%M%S")
    #X, y = load_data()



    project_id = cml.get_project()['id']
    params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}


    default_engine_details = cml.get_default_engine({})
    default_engine_image_id = default_engine_details["id"]


    print("creando el modelo")

    example_model_input = {"StreamingTV": "No", "MonthlyCharges": 70.35, "PhoneService": "No", "PaperlessBilling": "No", "Partner": "No", "OnlineBackup": "No", "gender": "Female", "Contract": "Month-to-month", "TotalCharges": 1397.475,
                   "StreamingMovies": "No", "DeviceProtection": "No", "PaymentMethod": "Bank transfer (automatic)", "tenure": 29, "Dependents": "No", "OnlineSecurity": "No", "MultipleLines": "No", "InternetService": "DSL", "SeniorCitizen": "No", "TechSupport": "No"}
    try:


                  # Create the YAML file for the model lineage
        yaml_text = \
            """"ModelOpsChurn":
          hive_table_qualified_names:                # this is a predefined key to link to training data
            - "default.telco_churn@cm"               # the qualifiedName of the hive_table object representing                
          metadata:                                  # this is a predefined key for additional metadata
            query: "select * from historical_data"   # suggested use case: query used to extract training data
            training_file: "3_trainStrategy.py"       # suggested use case: training file used
        """

        with open('lineage.yml', 'w') as lineage:
            lineage.write(yaml_text)
        model_id = cml.get_models(params)[0]['id']
        latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

        build_model_params = {
          "modelId": latest_model['latestModelBuild']['modelId'],
          "projectId": latest_model['latestModelBuild']['projectId'],
          "targetFilePath": "11_best_model_serve.py",
          "targetFunctionName": "explain",
          "engineImageId": default_engine_image_id,
          "kernel": "python3",
          "examples": latest_model['latestModelBuild']['examples'],
          "cpuMillicores": 1000,
          "memoryMb": 2048,
          "nvidiaGPUs": 0,
          "replicationPolicy": {"type": "fixed", "numReplicas": 1},
          "environment": {},"runtimeId":int(id_rt)}

        cml.rebuild_model(build_model_params)
        sys.argv=[]
        print('rebuilding...')
        # Wait for the model to deploy.
        hola='prueba'




    except:


                # Create the YAML file for the model lineage
      yaml_text = \
          """"ModelChurn":
        hive_table_qualified_names:                # this is a predefined key to link to training data
          - "default.telco_churn@cm"               # the qualifiedName of the hive_table object representing                
        metadata:                                  # this is a predefined key for additional metadata
          query: "select * from historical_data"   # suggested use case: query used to extract training data
          training_file: "3_trainStrategy.py"       # suggested use case: training file used
      """

      with open('lineage.yml', 'w') as lineage:
          lineage.write(yaml_text)

      create_model_params = {
          "projectId": project_id,
          "name": "ModelOpsChurn",
          "description": "Explain a given model prediction",
          "visibility": "private",
          "enableAuth": False,
          "targetFilePath": "11_best_model_serve.py",
          "targetFunctionName": "explain",
          "engineImageId": default_engine_image_id,
          "kernel": "python3",
          "examples": [
              {
                  "request": example_model_input,
                  "response": {}
              }],
          "cpuMillicores": 1000,
          "memoryMb": 2048,
          "nvidiaGPUs": 0,
          "replicationPolicy": {"type": "fixed", "numReplicas": 1},
          "environment": {},"runtimeId":int(id_rt)}
      print("creando nuevo modelo")
      new_model_details = cml.create_model(create_model_params)
      access_key = new_model_details["accessKey"]  # todo check for bad response
      model_id = new_model_details["id"]

      print("New model created with access key", access_key)

      # Disable model_authentication
      cml.set_model_auth({"id": model_id, "enableAuth": False})
      sys.argv=[]

      # Wait for the model to deploy.
      is_deployed = False
      while is_deployed == False:
          model = cml.get_model({"id": str(
              new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
          if model["latestModelDeployment"]["status"] == 'deployed':
              print("Model is deployed")
              break
          else:
              print("Deploying Model.....")
              time.sleep(10)
  else:
    print("no hay que reentrenar")
except:
  print("no hay fichero")
