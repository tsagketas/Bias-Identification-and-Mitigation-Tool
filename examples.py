import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

import tensorflow.compat.v1 as tf

from att_values import AttsnValues

import proccess
tf.disable_eager_execution()

def create_attsn_value(attribute, descriptions):
    obj = AttsnValues(attribute)
    for desc in descriptions:
        obj.add_value(desc)
    return obj

def get_groups(dataset_name):
    groups = {
        "German": (
            [{'sex': 1}],  # Privileged: Male
            [{'sex': 0}]  # Unprivileged: Female
        ),
        "Compas": (
            [{'race': 0}],  # Privileged: Caucasian
            [{'race': 1}]  # Unprivileged: African-American
        ),
        "Bank": (
            [{'age': 0}],  # Privileged: 25 or older
            [{'age': 1}]  # Unprivileged: Younger than 25
        )
    }

    return groups.get(dataset_name, ([], []))

def get_groups_human_readable(dataset_name):
    groups = {
        "German": (
            "Male",  # Privileged
            "Female",  # Unprivileged
            "Gender"  # Attribute
        ),
        "Compas": (
            "Caucasian",  # Privileged
            "African-American",  # Unprivileged
            "Race"  # Attribute
        ),
        "Bank": (
            "25 or older",  # Privileged
            "Younger than 25",  # Unprivileged
            "Age"  # Attribute
        )
    }

    return groups.get(dataset_name.lower(), ("Unknown", "Unknown", "Unknown"))

def get_example_attributes(dataset):
    data = []
    if dataset == "German":
        data.append(create_attsn_value("sex", [
            "Male is considered privileged (value = 1) and Female is considered unprivileged (value = 0)"
        ]))
    elif dataset == "Compas":
        data.append(create_attsn_value("race", [
            "Caucasian is considered privileged (value = 1) and African-American is considered unprivileged (value = 0)"
        ]))
    else:
        data.append(create_attsn_value("age", [
            "Age is considered as the protected attribute Individuals aged 25 or older are considered privileged (value = 0), while individuals younger than 25 are considered unprivileged (value = 1)"
        ]))

    return data

def get_data(dataset):
    train, test = dataset.split([0.7], shuffle=True)

    # Preprocess data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.features)
    y_train = train.labels.ravel()
    X_test = scaler.transform(test.features)
    y_test = test.labels.ravel()

    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    return lr, train, test, X_train, y_train, X_test, y_test

def get_example_dataset(dataset_name: str):
    datasets = {
        "German": GermanDataset,
        "Compas": CompasDataset,
        "Bank": BankDataset  # Default to BankDataset for any other input
    }

    return datasets.get(dataset_name, BankDataset)()

def get_structured_info(dataset_name, metric, metrics_to_calculate):
    results = []
    privileged, unprivileged, attribute = get_groups_human_readable(dataset_name)
    for metric_name in metrics_to_calculate:
        metric_value = getattr(metric, metric_name)()
        metric_info = proccess.construct_metric_info(metric_name, metric_value, attribute, privileged, unprivileged)
        results.append(metric_info)

    return results

def train_and_evaluate(dataset_name, metrics_to_calculate):
    dataset = get_example_dataset(dataset_name)
    privileged_groups, unprivileged_groups = get_groups(dataset_name.lower())

    lr, train, test, X_train, y_train, X_test, y_test = get_data(dataset)

    y_pred = lr.predict(X_test)

    pred_dataset = test.copy()
    pred_dataset.labels = y_pred

    metric = ClassificationMetric(test, pred_dataset, unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)

    model_metrics, y_pred = proccess.calculate_model_metrics(lr, X_test, y_test)
    return model_metrics, proccess.group_results_by_metric(get_structured_info(dataset_name, metric,
                                              metrics_to_calculate))

