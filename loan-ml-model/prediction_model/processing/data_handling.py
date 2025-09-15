import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

#create an env variable to track the absolute path of the location of the package to allow packing functions to be imported
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))  #add the package root to the sys.path to add this location to areas for package imports

from prediction_model.config import config # read in the config file from the prediction model package

#Define a function to load the dataset
def load_dataset(filename, train=True):
    filepath = os.path.join(config.DATAPATH,filename) # get full path of the file being read
    _data = pd.read_csv(filepath) # read the input dsc  file into the dataframe _data
    _data.columns = [c.strip() for c in _data.columns] # remove white spaces from around the column names
    return _data[config.FEATURES]   

# Separate X and y
def separate_data(data):
    X = data.drop(config.TARGET, axis=1)
    y= data[config.TARGET]
    return X,y

#Split the dataset

def split_data(X,y,test_size=0.2, random_state = 42):
    #Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    return X_train, X_test, y_train, y_test

#Serialisation - Save pipeline (model)
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME) 
    joblib.dump(pipeline_to_save,save_path)
    print(f"Model has been saved as {config.MODEL_NAME}")

#De-serialistation Load pipeline (model)
def load_pipeline():
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print(f"Model {config.MODEL_NAME} is loaded") 
    return model_loaded