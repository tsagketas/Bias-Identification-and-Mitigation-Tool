
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from aif360.datasets import AdultDataset,BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric


def adult():
    ad=[]
    cm=[]
    ct=[]
    protected=['sex','race','relationship','native-country']
    protected_values=['Male','White','Husband','United-States']

    for i in range(0,4):
        ad.append(AdultDataset(protected_attribute_names=[protected[i]],
            privileged_classes=[[protected_values[i]]], categorical_features=[],
            features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']))

        scaler = MinMaxScaler(copy=False)

        test, train = ad[i].split([16281])
        train.features = scaler.fit_transform(train.features)
        test.features = scaler.fit_transform(test.features)

        index = train.feature_names.index(protected[i])

        X_tr = np.delete(train.features, index, axis=1)
        X_te = np.delete(test.features, index, axis=1)
        y_tr = train.labels.ravel()

        lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
        lmod.fit(X_tr, y_tr)
            
        test_repd_pred = test.copy()
        test_repd_pred.labels = lmod.predict(X_te)

        p = [{protected[i]: 1}]
        u = [{protected[i]: 0}]
        
        metrics=ClassificationMetric(test_repd_pred,test,privileged_groups=p, unprivileged_groups=u)

        print(metrics.false_discovery_rate())

    return cm
    

yo=adult()

