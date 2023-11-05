
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def booler(data):
  for i in range(1,len(data)):

    if data[i]>0.65:
      data[i]=1.0
    else:
      data[i]=0.0
  return data.astype(int)

def f(x):
  try:
    x = int(x)
  except:
    x = np.nan
  return x

def mean_age(dataframe):
  sum=0.0
  n=0
  np.arr_nan=dataframe.Age.isnull()
  np.arr_ageindex=dataframe.Age
  for i in range(1,len(np.arr_nan)):
    if np.arr_nan[i]==False:
      n+=1
      sum+=int(float(np.arr_ageindex[i]))
  sred=sum/n
  return sred

def sigmoid(predict_):
  return (1.0 / (1 + np.exp(-predict_)))

def accuracy(target_,prediction_):
  point=0
  for i in range(1,len(prediction_)):
    if target_[i]==prediction_[i]:
      point+=1
  accur = (point/len(prediction_))*100
  return accur

df = pd.read_csv('train_new.csv', sep=',', names=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
df_obj = df.select_dtypes(include=['object']).copy()

#df_obj.info()
#df_obj["Cabin"].value_counts()

df_obj=df_obj.drop(labels = [0], axis = 0)
#df_obj=df_obj.dropna()

del df_obj['Name']
del df_obj['Cabin']
del df_obj['PassengerId']
del df_obj['Ticket']

df_obj["Age"] = df_obj["Age"].dropna().apply(f).dropna().astype(int)

cleanup_nums = {"Sex":     {"male": 1, "female": 0}}
df_obj=df_obj.replace(cleanup_nums)


#//////////////////////////////////


df_obj.Age=df_obj.Age.fillna(mean_age(df_obj))
df_obj=df_obj.dropna ()


df_obj=pd.get_dummies(df_obj, columns=["Pclass","Embarked"], prefix=["Class","Emb"])

df_obj["Survived"] = df_obj["Survived"].astype(float)
df_obj["Sex"] = df_obj["Sex"].astype(float)
df_obj["SibSp"] = df_obj["SibSp"].astype(float)
df_obj["Parch"] = df_obj["Parch"].astype(float)
df_obj["Fare"] = df_obj["Fare"].astype(float)
df_obj["Class_1"] = df_obj["Class_1"].astype(float)
df_obj["Class_2"] = df_obj["Class_2"].astype(float)
df_obj["Class_3"] = df_obj["Class_3"].astype(float)
df_obj["Emb_C"] = df_obj["Emb_C"].astype(float)
df_obj["Emb_Q"] = df_obj["Emb_Q"].astype(float)
df_obj["Emb_S"] = df_obj["Emb_S"].astype(float)

from sklearn.linear_model import LinearRegression
model =LinearRegression()

y=df_obj['Survived']
x=df_obj.drop('Survived',axis=1)
model.fit(x,y)

pred=x@model.coef_+model.intercept_


pred_sigmoid=sigmoid(pred)


#booler(pred)
#print(booler(pred))
#print(y)


print('Accuracy:  ' ,accuracy(y.astype(int),booler(pred_sigmoid)))
