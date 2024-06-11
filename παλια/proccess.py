from typing import List, Union, Dict
import os
import json
import itertools
# Modelling. Warnings will be used to silence various model warnings for tidier output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
import warnings

# Data handling/display
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns


# IBM's fairness tooolbox:
from aif360.datasets import BinaryLabelDataset,StandardDataset  # To handle the data
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric  # For calculating metrics
from aif360.explainers import MetricTextExplainer  # For explaining metrics

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# For the logistic regression model
from sklearn.preprocessing import StandardScaler

class MetricAdditions:
    def explain(self,
                disp: bool=True) -> Union[None, str]:
        """Explain everything available for the given metric."""

        # Find intersecting methods/attributes between MetricTextExplainer and provided metric.
        inter = set(dir(self)).intersection(set(dir(self.metric)))

        # Ignore private and dunder methods
        metric_methods = [getattr(self, c) for c in inter if c.startswith('_') < 1]

        # Call methods, join to new lines
        s = "\n".join([f() for f in metric_methods if callable(f)])

        if disp:
            print(s)
        else:
            return s
        
        
class MetricTextExplainer_(MetricTextExplainer, MetricAdditions):
    """Combine explainer and .explain."""
    pass

def preprocess_dataset(df,catCols,targets):

    df=df.fillna(df.mean())

    for col in catCols:
        df[col] = df[col].fillna("Missing value")

    for col in catCols:
        df_onehot = pd.concat([df[col], pd.get_dummies(df[col])], axis=1)
        df_onehot=df_onehot.drop([col], axis=1)
        df=df.drop([col], axis=1)
        df=pd.concat([df,df_onehot],axis=1)

    df2=pd.concat([targets['Label_value'],df],axis=1)
    df2=df2.rename(columns = {'Label_value':'Score'})
    df=pd.concat([targets['Score'],df],axis=1)

    return df,df2

def get_fairness_metrics(path_to_csv,protected_variables,protected_variables_values):

    data = pd.read_csv(path_to_csv)
    df=data.copy()

    target_cols=df[['Score','Label_value']]
    df=df.drop(['Score', 'Label_value'],axis=1)
    df.to_csv('erepousti.csv',index=False)
    
    catCols = df.select_dtypes(np.object).columns

    score_data,label_data=preprocess_dataset(df,catCols,target_cols)

    list_of_metrics=[]
    list_of_explainers=[]

    for i in range(len(protected_variables)):

        truth = BinaryLabelDataset(df=label_data,
                                        label_names=['Score'],
                                        protected_attribute_names=[protected_variables_values[i]],
                                        favorable_label=1,
                                        unfavorable_label=0,
                                        )

        preds = BinaryLabelDataset(df=score_data,
                                        label_names=['Score'],
                                        protected_attribute_names=[protected_variables_values[i]],
                                        favorable_label=1,
                                        unfavorable_label=0,
                                        )

        metric=ClassificationMetric(truth, 
                                        preds, 
                                        unprivileged_groups=[{protected_variables_values[i]: 0 }], 
                                        privileged_groups=[{protected_variables_values[i]: 1}],
                                        )

        explainer = MetricTextExplainer_(metric)                                                                    
        list_of_metrics.append(metric)
        list_of_explainers.append(explainer)

    return list_of_metrics,list_of_explainers

def fair_check(metric,objective,threshold):
    if metric >= 0:
        return  abs(metric - objective) <= threshold
    else:
        return abs(metric + objective) <= threshold

def get_data(the_metrics,fairness_metrics,protected_variables_values):
    x=[]
    
    for metric in fairness_metrics:
        values=[]
        variables=[]
        for i in range(len(the_metrics)):
            a=getattr(the_metrics[i], metric)()
            values.append(a)
            variables.append(protected_variables_values[i])
        x.append({'Metric': metric,'Protected_Attributes':variables,'Values':values})
    
    return x

def prepare_df(threshold,the_metrics,fairness_metrics,protected_variables):

    columns=["Value","Metric"]
    graph_df=pd.DataFrame(columns=columns)

    for i in range(len(the_metrics)):
        for el in fairness_metrics:
            data=[]
            a=getattr(the_metrics[i], el)()
            data.append(a)
            data.append(el)
            # data = list(filter(None, data))
            new=pd.DataFrame([data],columns=columns)
            graph_df=pd.concat([graph_df,new],axis=0)

    data=[]
    for el in protected_variables:
        for j in range(len(fairness_metrics)):
            data.append(el)

    new= {'Attribute':data} 
    new=pd.DataFrame(new)
    new.reset_index(drop=True, inplace=True)
    graph_df.reset_index(drop=True, inplace=True)
    graph_df=pd.concat([graph_df,new],axis=1)   
    return graph_df   

def show_graph(graph_df,fairness_metrics):
    print(graph_df)
    for el in fairness_metrics:
        path=r'static/'
        selected=graph_df[graph_df['Metric'].str.match(el)]
        sns.set()
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(y="Value",x="Attribute",data=selected).set_title(el) 
        file_name= el+".png"
        path=path+file_name
        print("what")
        ax.figure.savefig(path)
        # plt.show()

    return None
    
# fairness_metrics=['mean_difference','generalized_entropy_index','disparate_impact','average_odds_difference']
# the_metric,the_explainers = get_fairness_metrics( path_to_csv='check.csv',    protected_variables=['Gender', 'Attrition_Flag', 'Income_Category'],protected_variables_values=['M', 'Attrited Customer', 'Less than $40K'])

# y=get_data(the_metric,fairness_metrics,['M', 'Attrited Customer', 'Less than $40K'])

# print(type(y))
# for el in y:
#     for mertic in el.items():
#         print(el['Protected_Attributes'],el['Metric'],el['Values'])
# df2=prepare_df(0.2,the_metric,fairness_metrics,['Gender', 'Attrition_Flag', 'Income_Category'])
# show_graph(df2,fairness_metrics)



def preprocess_dataset(df,catCols,numCols,targets):

    atts=[]

    df=df.fillna(df.mean())

    for col in catCols:
        df[col] = df[col].fillna("Missing value")

    for col in catCols:
        for c in df[col].values:
            if (col+":"+c) not in atts:
                colname=col+":"+ c
                atts.append(colname)

    for col in numCols:
        for c in df[col].values:
            if (col+":"+str(c)) not in atts:
                colname=col+":"+str(c)
                atts.append(colname)            

    for col in catCols:
        # df_onehot = pd.concat([df[col], pd.get_dummies(df[col])], axis=1)
        df_onehot = pd.concat([df[col], pd.get_dummies(df[col]).rename(columns=lambda x:col +":"+x)], axis=1)
        df_onehot=df_onehot.drop([col], axis=1)
        df=df.drop([col], axis=1)
        df=pd.concat([df,df_onehot],axis=1)

    for col in numCols:
        df_onehot = pd.concat([df[col], pd.get_dummies(df[col]).rename(columns=lambda x:col +":"+str(x))], axis=1)
        df_onehot=df_onehot.drop([col], axis=1)
        df=df.drop([col], axis=1)
        df=pd.concat([df,df_onehot],axis=1)    

    df2=pd.concat([targets['Label_value'],df],axis=1)
    df2=df2.rename(columns = {'Label_value':'Score'})
    df=pd.concat([targets['Score'],df],axis=1)

    return df,df2,atts