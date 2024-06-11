from typing import List, Union, Dict
import os
import json
import itertools
from tqdm import tqdm

# Modelling. Warnings will be used to silence various model warnings for tidier output
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings
import math
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

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
from aif360.algorithms.preprocessing import Reweighing,DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover,MetaFairClassifier
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# For the logistic regression model
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler

df1 = pd.read_csv("plsplspls3.csv")
df2 = pd.read_csv("oriste.csv")

df2['Score']=df1['Score']

dataset = BinaryLabelDataset(df=df2,
                                label_names=['Score'],
                                protected_attribute_names=['education'],
                                favorable_label=2,
                                unfavorable_label=[0,1,3,4,5,6,7],
                                )
                