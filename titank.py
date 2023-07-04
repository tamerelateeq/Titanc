#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[387]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob 
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score , classification_report
import pandas_profiling
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import cross_validate


# In[388]:


t=pd.read_csv("test.csv")
t.info()


# # Import Data 

# In[389]:


def wrangel(path):
    # read data 
    df=pd.read_csv(path)
    #extract the social name
    df["title"]=df["Name"].str.extract("([A-Za-z]+)\.",expand=False)
    #convert title categorcal data 
    df.loc[df["title"]=="Mr"   , "title"]   = 0   
    df.loc[df["title"]=="Miss"   , "title"]   = 1
    df.loc[df["title"]=="Mrs"   , "title"]   = 2 
    df.loc[df["title"]=="Master"   , "title"]   = 3    
    conditions = (df["title"] == 'Ms') | (df["title"] == 'Col') | (df["title"] == 'Rev') | (df["title"] == 'Dr') | (df["title"] == 'Dona')
    df.loc[conditions, "title"] = 4
    #fill NAN Value of Fare Accorging to Social Name
    df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"),inplace=True)
    #fill NAN Value of Age Accorging to Social Name
    df["Age"].fillna(df.groupby("title")["Age"].transform("median"),inplace=True)
    #fill NAN Value of Embarked Accorging to Median
    df["Embarked"]=df["Embarked"].fillna("S")
    #remove nan columns
    drop=[]
    drop.append("Cabin")
    drop.append("Name")
    drop.append("Ticket")
    drop.append("title")
    df.drop(columns=drop,inplace=True)
    #convert Sex categorcal data 
    df.loc[df["Sex"]=="male"     , "Sex"]   = 0         # Male   ---> 0
    df.loc[df["Sex"]=="female"   , "Sex"]   = 1         # Female ---> 1
    #convert Embarked categorcal data 
    df.loc[df["Embarked"]=="S"   , "Embarked"]   = 0    # S ---> 1
    df.loc[df["Embarked"]=="C"   , "Embarked"]   = 1    # C ---> 2
    df.loc[df["Embarked"]=="Q"   , "Embarked"]   = 2    # Q ---> 3
    
    return df


# In[390]:


test  =  wrangel("test.csv")
df    =  wrangel("train.csv")


# In[340]:


df.head()


# In[341]:


df.info()


# In[391]:


pandas_profiling.ProfileReport(df)


# In[343]:


df["Embarked"].value_counts()


# In[344]:


test.info()


# In[352]:


test.isnull().sum()


# # Exploer Data

# In[353]:


print("Survive :",(df["Survived"]==1).sum())
print("Deceased :",(df["Survived"]==0).sum())


# In[354]:


df.describe()


# In[355]:


# Create the pie chart
values=df["Survived"].value_counts()
label=["Deceased ","Survive "]
plt.pie(values, labels=label,autopct='%1.1f%%')
# Add a title
plt.title('Distribution of Survived')
# Display the chart
plt.show()


# In[356]:


plt.hist(df["Parch"],bins=5, edgecolor='black');
plt.xlabel('Values')
plt.ylabel('Frequancy')
plt.title("Values of Parch")
plt.show();


# In[357]:


survive=df[df["Survived"]==1]["SibSp"].value_counts()
death=df[df["Survived"]==0]["SibSp"].value_counts()
dx=pd.DataFrame([survive,death],index=["survive","death"])
dx.plot(kind="bar");
plt.title("Survive of SibSp ");


# In[358]:


survive=df[df["Survived"]==1]["Pclass"].value_counts()
death=df[df["Survived"]==0]["Pclass"].value_counts()
dx=pd.DataFrame([survive,death],index=["survive","death"])
dx.plot(kind="bar");
plt.title("Survive of Pclass ");


# In[359]:


class1=df[df["Pclass"]==1]["Embarked"].value_counts()
class2=df[df["Pclass"]==2]["Embarked"].value_counts()
class3=df[df["Pclass"]==3]["Embarked"].value_counts()
dx=pd.DataFrame([class1,class2,class3],index=["class 1","class 2","class 3"])
dx.plot(kind="bar",stacked=True);
plt.title("Survive of Pclass ");


# We Found that Embarked from S in 1st & 2nd & 3rd Class

# In[360]:


# Create the pie chart
values=df["Sex"].value_counts()
label=["male","female"]
plt.pie(values, labels=label,autopct='%1.1f%%')
# Add a title
plt.title('Distribution of Survived')
# Display the chart
plt.show()


# In[361]:


survive = df[df["Survived"]==1]["Sex"].value_counts()
death   = df[df["Survived"]==0]["Sex"].value_counts()
dx      = pd.DataFrame([survive,death],index=["survive","death"])
dx=dx.rename(columns={0:"male",1:"female"})
dx.plot(kind="bar")
plt.legend()
plt.title("Survive of Sex");


# In[ ]:





# In[362]:


corrleation = df.drop(columns="Survived").corr()
sns.heatmap(corrleation)


# # Split Data

# In[ ]:





# In[363]:


df


# In[364]:


target="Survived"
y = df[target]
X = df.drop(columns=target)
x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# # Baseline

# In[365]:


y_train_mean = y_train.mean()
print ("Baseline :",round(y_train_mean,2))


# # Logestic Regression

# # Itrate

# In[366]:


log_model = LogisticRegression(max_iter=10000)


# In[367]:


log_model.fit(x_train,y_train)


# # 

# # Evaluate

# In[368]:


accuracy=classification_report(y_test,log_model.predict(x_test))
print(accuracy)


# In[369]:


acc_test = accuracy_score(y_test,log_model.predict(x_test))
acc_test = accuracy_score(y_test,log_model.predict(x_test))
acc_train= accuracy_score(y_train,log_model.predict(x_train))
print("Accuracy test:",round(acc_test,2))
print("Accuracy train:",round(acc_train,2))


# # KNN Classfier 

# In[370]:


knn= KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train,y_train)


# In[371]:


accuracy=classification_report(y_test,knn.predict(x_test))
print(accuracy)


# In[372]:


scoring="accuracy"
score = cross_validate(knn , x_train.drop(columns=["PassengerId"],axis=1),y_train,cv=k_fold, n_jobs=1,scoring=scoring)
print(score['test_score'])


# In[373]:


print("Accuracy :",round(np.mean(score['test_score']),2))


# # Descion Tree

# In[374]:


# Create a decision tree classifier
dec_tree=  DecisionTreeClassifier()
# Train the classifier
dec_tree.fit(x_train, y_train)


# In[375]:


accuracy=classification_report(y_test,dec_tree.predict(x_test))
print(accuracy)


# In[376]:


acc_test = accuracy_score(y_test,dec_tree.predict(x_test))
print("Accuracy test:",round(acc_test,2))


# In[377]:


scoring="accuracy"
score = cross_validate(dec_tree , x_train.drop(columns=["PassengerId"],axis=1),y_train,cv=k_fold, n_jobs=1,scoring=scoring)
print("Accuracy :",round(np.mean(score['test_score']),2))


# # Random Forest 

# In[378]:


# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()
# Train the classifier
rf_classifier.fit(x_train, y_train)


# In[379]:


# Calculate the accuracy
accuracy = accuracy_score(y_test, rf_classifier.predict(x_test))
print("Accuracy:", round(accuracy,2))


# In[380]:


scoring="accuracy"
score = cross_validate(rf_classifier , x_train.drop(columns=["PassengerId"],axis=1),y_train, n_jobs=1,scoring=scoring)
print("Accuracy :",round(np.mean(score['test_score']),1))


# # Naive Bayes

# In[381]:


nav= GaussianNB()
# Train the classifier
nav.fit(x_train, y_train)


# In[382]:


# Calculate the accuracy
accuracy = accuracy_score(y_test, nav.predict(x_test))
print("Accuracy:", round(accuracy,2))


# In[383]:


scoring="accuracy"
score = cross_validate(nav , x_train.drop(columns=["PassengerId"],axis=1),y_train, n_jobs=1,scoring=scoring)
print("Accuracy :",round(np.mean(score['test_score']),2))


# # Communicat

# The best model is Random Forest with Accuracy : 82 

# In[384]:


pred_test=rf_classifier.predict(test)

data = pd.DataFrame({'PassengerId': test["PassengerId"], 'Survived': pred_test})


# In[385]:


data.head()


# In[386]:


data.to_csv(r'D:\projects\gender_submission.csv', index=False)


# In[ ]:




