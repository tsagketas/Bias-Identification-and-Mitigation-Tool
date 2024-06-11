
# data manipulation libraries
import pandas as pd
import numpy as np
import metrics 
from time import time

# Graphs libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('seaborn-white')
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from plotly import tools

# Libraries to study
from aif360.datasets import StandardDataset,StructuredDataset,BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import LFR, Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification

# ML libraries
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.utils import shuffle
import tensorflow as tf

# Design libraries
from IPython.display import Markdown, display
import warnings
warnings.filterwarnings("ignore")

def decode_dataset(data, encoders, numerical_features, categorical_features):
    df = data.copy()
    for feat in df.columns.values:
        if feat in numerical_features:
            df[feat] = encoders[feat].inverse_transform(np.array(df[feat]).reshape(-1, 1))
    for feat in categorical_features:
        df[feat] = encoders[feat].inverse_transform(df[feat].astype(int))
    return df

def meta_data(dataset):
    # print out some labels, names, etc.
    display(Markdown("#### Dataset shape"))
    print(dataset.features.shape)
    display(Markdown("#### Favorable and unfavorable labels"))
    print(dataset.favorable_label, dataset.unfavorable_label)
    display(Markdown("#### Protected attribute names"))
    print(dataset.protected_attribute_names)
    display(Markdown("#### Privileged and unprivileged protected attribute values"))
    print(dataset.privileged_protected_attributes, dataset.unprivileged_protected_attributes)
    display(Markdown("#### Dataset feature names"))
    print(dataset.feature_names)

def add_to_df_algo_metrics(algo_metrics, model, fair_metrics, preds, probs, name):
    return algo_metrics.append(pd.DataFrame(data=[[model, fair_metrics, preds, probs]], columns=['model', 'fair_metrics', 'prediction', 'probs'], index=[name]))

def get_fair_metrics_and_plot(data, model, plot=True, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    # # fair_metrics function available in the metrics.py file
    fair = metrics.fair_metrics(data, pred)

    if plot:
        # plot_fair_metrics function available in the visualisations.py file
        # The visualisation of this function is inspired by the dashboard on the demo of IBM aif360 
        # metrics.plot_fair_metrics(fair)
        display(fair)
    
    return fair

df = pd.read_csv("titanic.csv")
df = df.drop('PassengerId',axis=1)
cat_columns = []
num_columns = []
Y_columns = ['Survived'] #here we have the label_value column

#edw kanoume teleia vriskoume cat kai num cols and we handle the missing num and cat values
numCols = df.select_dtypes(np.number).columns
catCols = df.select_dtypes(np.object).columns

print(df.mean())
df=df.fillna(df.mean())

for col in df:
    if col in catCols and col not in Y_columns:
        df[col] = df[col].fillna("Missing value")

#--------------------------------------------------------------------------------------------

data_encoded = df.copy()
categorical_names = {}
encoders = {}

#encode to number both cat and num cols
for feature in catCols:
    le = LabelEncoder()
    le.fit(data_encoded[feature])
    
    data_encoded[feature] = le.transform(data_encoded[feature])
    
    categorical_names[feature] = le.classes_
    encoders[feature] = le

numerical_features = [c for c in df.columns.values if c not in catCols and c not in Y_columns]

for feature in numCols:
    val = data_encoded[feature].values[:, np.newaxis]
    mms = MinMaxScaler().fit(val)
    data_encoded[feature] = mms.transform(val)
    encoders[feature] = mms
    
data_encoded = data_encoded.astype(float)
# data_encoded.to_csv('coded.csv',index=False)
data_decoded = decode_dataset(data_encoded,encoders, numerical_features,catCols)
# data_decoded.to_csv('decoded.csv',index=False)

#----------------------------------------MODEL---------------------------------
# x=data_encoded.drop(columns=['Survived'])
# x2=data_encoded
# y=data_encoded['Survived']

# model = RandomForestClassifier()
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.3)
# model.fit(x_train,y_train)
# preds=model.predict(x_test)
# acc=model.score(x_test,y_test)
# print(acc)
#---------------------------------------DATASET----------------------------------
# df1 = pd.DataFrame(preds, columns = ['Score']) # edw ftiaxnoume to protipo dataset
# df1= df1.join(x2)
# df1.to_csv('preds2.csv',index=False)
#---------------------------------------------------------------------------------

df2 = pd.read_csv("preds2.csv")

print(encoders['Sex'])
preds=df2['Score']
data_encoded= df2.drop('Score',axis=1)
preds2=data_encoded.copy()
preds2['Survived']=preds.values
preds2.to_csv('arxida',index=False)
train_pp_bld = BinaryLabelDataset(df=data_encoded,
                                  label_names=['Survived'],
                                  protected_attribute_names=['Sex'],
                                  favorable_label=1,
                                  unfavorable_label=0)

train_pp_bld2 = BinaryLabelDataset(df=preds2,
                                  label_names=['Survived'],
                                  protected_attribute_names=['Sex'],
                                  favorable_label=1,
                                  unfavorable_label=0)

privileged_groups = [{'Sex': 0}]
unprivileged_groups = [{'Sex': 1}]

doulepse=ClassificationMetric(train_pp_bld, train_pp_bld2, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)                             

print(doulepse.disparate_impact())