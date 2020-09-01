# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:07:58 2020

@author: kushm
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv('C:/Users/\kushm/Vision DataScience/Practice/diabetes.csv')
print(df.shape)
df.describe()
df.head(3)
df.columns
feature_cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X=df[feature_cols]
y=df.Outcome
X_train1,X_test1,y_train1,y_test1=train_test_split(X,y,test_size=0.3,random_state=40)
logreg=LogisticRegression(penalty='none',solver='saga')
logreg.fit(X_train1,y_train1)
y_pred=logreg.predict(X_test1)
logreg.coef_
print('Accuracy of logistic regression classifer on test set: {:.2f}'.format(logreg.score(X_test1,y_test1)))
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test1,y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test1,y_pred))
def logit_pvalue(model,x):
    p=model.predict_proba(x)
    n=len(p)
    m=len(model.coef_[0])+1
    coefs=np.concatenate([model.intercept_,model.coef_[0]])
    x_full=np.matrix(np.insert(np.array(x),0,1,axis=1))
    ans=np.zeros((m,m))
    for i in range(n):
        ans=ans+np.dot(np.transpose(x_full[i,:]),x_full[i,:])*p[i,1]*p[i,0]
    vcov=np.linalg.inv(np.matrix(ans))
    se=np.sqrt(np.diag(vcov))
    t=coefs/se
    p=(1-(norm.cdf(abs(t))))*2
    return p
print(logit_pvalue(logreg,X_train1))