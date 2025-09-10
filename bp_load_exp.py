import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import ModelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

        

df = pd.read_csv("../loan_approval_dataset.csv")
print(df.head())
#print(df.shape)#check num rows and columas
#print(df.nunique()) # check that each field has at least 2 or more unique values
#print(df.duplicated().sum())  # check that there are no duplicated rows
#print(df.columns)  # check column names, leading spaces observed
df.columns = [s.strip() for s in df.columns]  # strip spaces from column names
#print(df.columns)  #recheck column names
Y = df["loan_status"]
df.drop(columns=["loan_status","loan_id"],inplace = True)
X = df
print(X.head())
print(Y.head())
print(X.info())
#Pre-processing step, replace all of the individual asset values with a total asset value column
X['total_assets'] = X['residential_assets_value'] + X['commercial_assets_value'] + X['luxury_assets_value'] + X['bank_asset_value']

X = X.drop(columns=['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'])
print(X.head())
print(X['education'].unique())
print(X['self_employed'].unique())
print(Y.unique())
X['self_employed'] = X['self_employed'].str.strip()
X['education'] = X['education'].str.strip()
Y = Y.str.strip()
print(X['education'].unique())
print(X['self_employed'].unique())
print(Y.unique())
print(X.info())

X = X.replace({'education':{'Graduate':1,'Not Graduate':0}, 'self_employed': {'Yes':1,'No':0} } )
Y = Y.map({'Approved':1,'Rejected':0})
print(X.head())
#Now apply log transformation to the numerical columns with large values
log_cols = ['income_annum','loan_amount', 'total_assets']
X[log_cols] = np.log(X[log_cols])
print(X.head())
print(Y.head())

#Split the dataset to training and testing groups

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3,random_state=1)
#Now build the model
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, Y_train)
Y_pred = log.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_pred,Y_test)
print(f"Accuracy is {acc}")
