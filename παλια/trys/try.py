import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from aif360.metrics import BinaryLabelDatasetMetric,DatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset,StructuredDataset,StandardDataset
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer
import proccess as prcs
df = pd.read_csv("titanic.csv")

numCols = df.select_dtypes(np.number).columns
catCols = df.select_dtypes(np.object).columns

# numcols=list(set(numCols))
# catcols=list(set(catCols))
print(numCols)
print(catCols)

df1=df[numCols]
df2=df[catCols]

# print(df1)
# df1.to_csv('gianadoume.csv', index=False)
# df2.to_csv('giana.csv', index=False)



# for col in df2.columns:
#     print(col)
#     df3=pd.get_dummies(df[col],drop_first=True)
#     df=pd.concat([df3,df],axis=1)

# df.to_csv('giana2.csv', index=False)    
print(df.mean())
df=df.fillna(df.mean())
# df.to_csv('filled.csv',index=False)

for col in df2.columns:
    df_onehot = pd.concat([df2[col], pd.get_dummies(df2[col])], axis=1)
    df_onehot=df_onehot.drop([col], axis=1)
    df=df.drop([col], axis=1)
    df=pd.concat([df,df_onehot],axis=1)


# df.to_csv('tisto.csv',index=False)

df_aif = BinaryLabelDataset(df=df, label_names=['Survived'], protected_attribute_names=['Sex'],favorable_label=1,unfavorable_label=0)
df_aif2 = StructuredDataset(df=df, label_names=['Survived'], protected_attribute_names=['male'])




privileged_group = [{'male': 0}]
unprivileged_group = [{'male': 1}]

df_orig_trn, df_orig_val, df_orig_tst = df_aif.split([0.5, 0.8], shuffle=True)

print([x.features.shape for x in [df_orig_trn, df_orig_val, df_orig_tst]])

# df_orig_tst.to_csv('check.csv', index=False)
print(df_orig_val)
metric_orig_trn = BinaryLabelDatasetMetric( df_orig_val, unprivileged_group, privileged_group)

df_what=DatasetMetric(df_aif2, unprivileged_group, privileged_group)
print(df_what)
yo=1- np.minimum(metric_orig_trn.disparate_impact().round(3),1/metric_orig_trn.disparate_impact().round(3))
text_expl = MetricTextExplainer(metric_orig_trn)
print(yo)
print(text_expl.disparate_impact())
print(text_expl.mean_difference())
print(text_expl.num_positives())

RW = Reweighing(unprivileged_group, privileged_group)
df_transf_trn = RW.fit_transform(df_orig_trn)
print(df_transf_trn)
metric_transf_trn = BinaryLabelDatasetMetric(df_transf_trn, unprivileged_group, privileged_group)
yo=1- np.minimum(metric_transf_trn.disparate_impact().round(3),1/metric_transf_trn.disparate_impact().round(3))
print(yo)
