try:
  from pyspark.sql.types import *
  from pyspark.sql import SparkSession
except:
  print('install pyspark')
  !pip3 install pyspark
  from pyspark.sql.types import *
  from pyspark.sql import SparkSession
  
!pip3 install -r requirements.txt  
!pip3 install pickle
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
import xml.etree.ElementTree as ET
from cmlbootstrap import CMLBootstrap
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


data_dir = '/home/cdsw'

idcol = 'customerID'
labelcol = 'Churn'
cols = (('gender', True),
        ('SeniorCitizen', True),
        ('Partner', True),
        ('Dependents', True),
        ('tenure', False),
        ('PhoneService', True),
        ('MultipleLines', True),
        ('InternetService', True),
        ('OnlineSecurity', True),
        ('OnlineBackup', True),
        ('DeviceProtection', True),
        ('TechSupport', True),
        ('StreamingTV', True),
        ('StreamingMovies', True),
        ('Contract', True),
        ('PaperlessBilling', True),
        ('PaymentMethod', True),
        ('MonthlyCharges', False),
        ('TotalCharges', False))


# This is a fail safe incase the hive table did not get created in the last step.
try:
    spark = SparkSession\
        .builder\
        .appName("PythonSQL")\
        .master("local[*]")\
        .getOrCreate()

    if (spark.sql("SELECT count(*) FROM default.telco_churn").collect()[0][0] > 0):
        df = spark.sql("SELECT * FROM default.telco_churn").toPandas()
except:
    print("Hive table has not been created")
    df = pd.read_csv(os.path.join(
        'raw', 'WA_Fn-UseC_-Telco-Customer-Churn-.csv'))





df = df.replace(r'^\s$', np.nan, regex=True).dropna().reset_index()
df.index.name = 'id'
data, labels = df.drop(labelcol, axis=1), df[labelcol]
data = data.replace({'SeniorCitizen': {1: 'Yes', 0: 'No'}})
# This is Mike's lovely short hand syntax for looping through data and doing useful things. I think if we started to pay him by the ASCII char, we'd get more readable code.
data = data[[c for c, _ in cols]]
catcols = (c for c, iscat in cols if iscat)
for col in catcols:
    data[col] = pd.Categorical(data[col])
labels = (labels == 'Yes')

ce = CategoricalEncoder()
X = ce.fit_transform(data)


y=labels.values
print("empenzando los experimentos")

run_time_suffix = datetime.datetime.now()
#run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")
run_time_suffix = run_time_suffix.strftime("%M%S")
#X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 3
rf = RandomForestRegressor(n_estimators=n_estimators)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

filename = './models/champion/ce.pkl'
pickle.dump(ce, open(filename, 'wb'))

filename = './models/champion/champion.pkl'
pickle.dump(rf, open(filename, 'wb'))


project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}


default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]


print("creando el modelo")

example_model_input = {"StreamingTV": "No", "MonthlyCharges": 70.35, "PhoneService": "No", "PaperlessBilling": "No", "Partner": "No", "OnlineBackup": "No", "gender": "Female", "Contract": "Month-to-month", "TotalCharges": 1397.475,
               "StreamingMovies": "No", "DeviceProtection": "No", "PaymentMethod": "Bank transfer (automatic)", "tenure": 29, "Dependents": "No", "OnlineSecurity": "No", "MultipleLines": "No", "InternetService": "DSL", "SeniorCitizen": "No", "TechSupport": "No"}



          # Create the YAML file for the model lineage
yaml_text = \
    """"ModelChurn":
  hive_table_qualified_names:                # this is a predefined key to link to training data
    - "default.telco_churn@cm"               # the qualifiedName of the hive_table object representing                
  metadata:                                  # this is a predefined key for additional metadata
    query: "select * from historical_data"   # suggested use case: query used to extract training data
    training_file: "3_trainStrategy_job.py"       # suggested use case: training file used
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



    
