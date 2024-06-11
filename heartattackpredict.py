import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')

print(len(df['output']))
x = df.drop(['output'], axis=1)
y = df['output']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
sc = StandardScaler()
sc.fit_transform(x_train)
print(x_train)
sc.transform(x_test)
clf = LogisticRegression(random_state = 10)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
acc=accuracy_score(y_pred,y_test)
print(acc)
# print(y_pred.reshape(-1,1))
x_test['Score']=y_pred
x_test['Label_value']=y_test

# x_test=x_test.drop(x_test.columns[0],axis=1)

x_test.to_csv('new_dataset.csv',index=False)
print(len(y_pred))