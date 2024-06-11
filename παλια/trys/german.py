
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_adult, load_preproc_data_compas

def german():

    cm=[]
    protected=['sex','age']

    data = GermanDataset()
    scaler = StandardScaler(copy=False)

    train, test = data.split([0.7], shuffle=True)
    X_train = scaler.fit_transform(train.features)
    y_train = train.labels.ravel()
    print(y_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = test.copy()
    preds.labels = model.predict(test)

    for el in protected:

        p = [{ el: 1 }]
        u = [{ el: 0 }]
        print("yo")
        cm.append(BinaryLabelDatasetMetric(preds, privileged_groups=p, unprivileged_groups=u))

    return cm
    

yo=german()

