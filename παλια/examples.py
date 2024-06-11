from matplotlib import pyplot as plt
import json 
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from aif360.datasets import AdultDataset,GermanDataset,CompasDataset,BankDataset,BinaryLabelDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.explainers import MetricJSONExplainer

def example(dataset):
    
    cm=[]

    if dataset == "Adult" :
        data=load_preproc_data_adult()
        privileged_groups = [{'sex': 1},{'race': 1}]
        unprivileged_groups = [{'sex': 0},{'race': 0}]
    elif dataset == "German":
        data=load_preproc_data_german()
        privileged_groups = [{'sex': 1},{'age': 1}]
        unprivileged_groups = [{'sex': 0},{'age': 0}]
    else :
        data=load_preproc_data_compas()
        privileged_groups = [{'sex': 1},{'race': 1}]
        unprivileged_groups = [{'sex': 0},{'race': 0}]
    
    test, train = data.split([0.4])

    lmod = LogisticRegression()
    lmod.fit(train.features, train.labels.ravel())

    test_repd_pred = test.copy()
    test_repd_pred.labels = lmod.predict(test.features)

    for i in range (0,2):
        cm.append(ClassificationMetric(test_repd_pred,test,privileged_groups=[privileged_groups[i]], unprivileged_groups=[unprivileged_groups[i]]))

        yo=MetricJSONExplainer(cm[i])

        expl=json.loads(yo.mean_difference())
        print(cm[i].mean_difference(),expl["ideal"])
        expl=json.loads(yo.disparate_impact())
        print(cm[i].disparate_impact(),expl["ideal"])
        expl=json.loads(yo.average_abs_odds_difference())
        print(cm[i].average_abs_odds_difference(),expl["ideal"])

    expl=json.loads(yo.false_negative_rate_ratio())
    print(cm[0].false_negative_rate_ratio(),expl["ideal"])
    print(cm[1].false_negative_rate_ratio(),expl["ideal"])
    return cm
    

yo=example("German")