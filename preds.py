import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

pima=pd.read_csv("plsplspls2.csv")

pima['age'] = pima['age'].astype(str)
pima['age'].values[pima['age'].values == str(0)] = str("< 25")
pima['age'].values[pima['age'].values == str(1)] = str(">= 25")

pima.to_csv("plsplspls3.csv", index=False)


# pima=pd.read_csv("bank-additional2.csv")

# pima2=pd.read_csv("bank-additional.csv",sep=';')
# # pima2['age'].values[pima['age'].values < 25] = str("< 25")
# # pima2['age'].values[pima['age'].values >= 25] = str(">= 25")
# pima2.to_csv("bank-additional2.csv", index=False)


# # pima=pima.drop(columns=["id"])
# df=pima.copy()
# # pima=pima.replace(['Excellent','Vg'],1)
# # pima=pima.replace(['Average','Good'],0)
# # pima.to_csv('new_students.csv',index=False)
# # feature_cols = ['Gender','Caste','coaching','time','Class_ten_education','twelve_education','medium','Class_','Class_XII_Percentage','Father_occupation','Mother_occupation']

# X = df.drop(columns=["y"]) # Features
# feature_cols = X.select_dtypes(np.object).columns
# y = pima.y # Target variable
# print(X)
# print(feature_cols)
# y=y.replace('no',0)
# y=y.replace('yes',1)
# X_train1,X_test1,y_train1,y_test1=train_test_split(X,y,test_size=0.25,random_state=0)

# for col in feature_cols:
#         df_onehot = pd.concat([X[col], pd.get_dummies(X[col])], axis=1)
#         df_onehot=df_onehot.drop([col], axis=1)
#         X=X.drop([col], axis=1)
#         X=pd.concat([X,df_onehot],axis=1)


# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# logreg = LogisticRegression().fit(X_train,y_train)
# y_pred=logreg.predict(X_test)

# y_pred2=logreg.predict(X_train)
# preds=np.concatenate((y_pred, y_pred2), axis=None)
# truth=np.concatenate((y_test, y_train), axis=None)

# whole=np.concatenate((X_train1, X_test1), axis=0)
# pima['y']=truth
# pima['Score']=preds
# pima=pima.rename(columns = {'y':'Label_value'})
# print(pima.select_dtypes(np.object).columns)
# # pima.to_csv('plsplspls2.csv',index=False)
# print(len(truth))
# print(len(preds))
# print(len(pima))
# # whole = whole.assign(Score=preds.reshape(-1,1))
# # whole = whole.assign(Label_value=truth.reshape(-1,1))
# # print(whole)

# # # whole2 = train.append(test, ignore_index=True)
# # # # # prediction = pd.DataFrame(y_pred, columns=['predictions'])
# # # # # prediction=pd.concat([prediction,X_test],axis=1)
# # # # # prediction=pd.concat([prediction,y_test],axis=1)
# # # # # prediction=prediction.dropna()
# # # # # df = pd.dataframe()
# # # # print(y_test)
# # # # prediction = pd.DataFrame(y_test, columns=['Label_value'])


# # # clf=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
# # # y_pred1=clf.predict(X_test)

# # # print(y_pred.reshape(-1,1))
# # # print(y_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
# print("Recall:",metrics.recall_score(y_test, y_pred))

# print("Accuracy:",metrics.accuracy_score(y_train, y_pred2))
# print("Precision:",metrics.precision_score(y_train, y_pred2))
# print("Recall:",metrics.recall_score(y_train, y_pred2))

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
# print("Precision:",metrics.precision_score(y_test, y_pred1))
# print("Recall:",metrics.recall_score(y_test, y_pred1))

# cnf_matrix = metrics.confusion_matrix(y_test, y_pred1)
# cnf_matrix

# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="PuBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# plt.subplots_adjust(bottom=.1,left=.10)
# plt.show()

# y_pred_proba = logreg.predict_proba(X_test)[::,1]
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()