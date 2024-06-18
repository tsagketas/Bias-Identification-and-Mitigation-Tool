from typing import List, Union, Dict
import os
import json
import itertools
from collections import defaultdict
from tqdm import tqdm
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.exceptions import DataConversionWarning
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.explainers import MetricTextExplainer
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover, MetaFairClassifier
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, \
    RejectOptionClassification

# Disable warnings for cleaner output
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Function to prepare the data
def prepare_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    X = process_numerical_variables(X)
    X, label_encoders = encode_categorical_variables(X)
    return X, y, label_encoders


# Function to encode numerical variables
def process_numerical_variables(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        median = df[column].median()
        df[column].fillna(median, inplace=True)

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

# Function to encode categorical variables
def encode_categorical_variables(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders


# Function to train the selected model
def train_model(model_name, X_train, y_train):
    if model_name == "logistic_regression":
        model = LogisticRegression()
    elif model_name == "naive_bayes":
        model = GaussianNB()
    elif model_name == "random_forest":
        model = RandomForestClassifier()
    elif model_name == "svm":
        model = SVC(probability=True)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    model.fit(X_train, y_train)
    return model


# Function to calculate model metrics
def calculate_model_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    return metrics, y_pred


# Function to convert to BinaryLabelDataset
def convert_to_binary_label_dataset(X, y, protected_attributes, label_name='label'):
    df = X.copy()
    df[label_name] = y

    # Remove duplicates
    return BinaryLabelDataset(df=df, label_names=[label_name],
                              protected_attribute_names=list(set(protected_attributes)))


def get_binary_datasets(X, y_true, y_pred, protected_attributes):
    ground_truth_dataset = convert_to_binary_label_dataset(X, y_true, protected_attributes,
                                                           label_name='true_label')
    predicted_dataset = ground_truth_dataset.copy()
    predicted_dataset.labels = y_pred.reshape(-1, 1)

    return ground_truth_dataset, predicted_dataset


# Function to calculate fairness metrics
def calculate_fairness_metrics(X, y_true, y_pred, label_encoders, atts_n_vals_picked, metrics_to_calculate, threshold):
    results = []

    for att in atts_n_vals_picked:
        protected_attributes = []
        protected_attributes.append(att['attribute'])
        if 'intersection' in att and att['intersection']:
            for intersection_att in att['intersection']:
                protected_attributes.append(intersection_att['attribute'])
                ground_truth_dataset, predicted_dataset = get_binary_datasets(X, y_true, y_pred, protected_attributes)
                results.extend(
                    calculate_intersectional_metrics(att, intersection_att, ground_truth_dataset, predicted_dataset,
                                                     label_encoders, metrics_to_calculate, threshold))
        else:
            ground_truth_dataset, predicted_dataset = get_binary_datasets(X, y_true, y_pred, protected_attributes)
            results.extend(calculate_standard_metrics(att, ground_truth_dataset, predicted_dataset, label_encoders,
                                                      metrics_to_calculate, threshold))

    return group_results_by_metric(results)


# Function to calculate standard metrics
def calculate_standard_metrics(att, ground_truth_dataset, predicted_dataset, label_encoders, metrics_to_calculate,
                               threshold):
    results = []

    privileged_group = [{att['attribute']: label_encoders[att['attribute']].transform([att['privileged']])[0]}]
    unprivileged_group = [{att['attribute']: label_encoders[att['attribute']].transform([att['unprivileged']])[0]}]

    metric = ClassificationMetric(ground_truth_dataset, predicted_dataset, privileged_groups=privileged_group,
                                  unprivileged_groups=unprivileged_group)

    for metric_name in metrics_to_calculate:
        metric_value = getattr(metric, metric_name)()
        if not fair_check(metric_value, metric_name, threshold):
            metric_info = construct_metric_info(metric_name, metric_value, [att['attribute']], att['privileged'],
                                                att['unprivileged'])
            results.append(metric_info)

    return results


# Function to calculate intersectional metrics
def calculate_intersectional_metrics(att, intersection_att, ground_truth_dataset, predicted_dataset, label_encoders,
                                     metrics_to_calculate, threshold):
    results = []

    privileged_group = [
        {
            att['attribute']: label_encoders[att['attribute']].transform([att['privileged']])[0],
            intersection_att['attribute']:
                label_encoders[intersection_att['attribute']].transform([intersection_att['privileged']])[0]
        }
    ]
    unprivileged_group = [
        {
            att['attribute']: label_encoders[att['attribute']].transform([att['unprivileged']])[0],
            intersection_att['attribute']:
                label_encoders[intersection_att['attribute']].transform([intersection_att['unprivileged']])[0]
        }
    ]

    metric = ClassificationMetric(ground_truth_dataset, predicted_dataset, privileged_groups=privileged_group,
                                  unprivileged_groups=unprivileged_group)

    for metric_name in metrics_to_calculate:
        metric_value = getattr(metric, metric_name)()
        if not fair_check(metric_value, metric_name, threshold):
            metric_info = construct_metric_info(metric_name, metric_value,
                                                [att['attribute'], intersection_att['attribute']], att['privileged'],
                                                att['unprivileged'], intersectional_attributes=[intersection_att])
            results.append(metric_info)

    return results


# Function to construct metric information
def construct_metric_info(metric_name, metric_value, protected_attributes, privileged_group, unprivileged_group,
                          intersectional_attributes=None):
    info = {
        'Metric': metric_name,
        'Protected_Attributes': protected_attributes,
        'Values': metric_value,
        'Privileged_Group': privileged_group,
        'Unprivileged_Group': unprivileged_group,
        'Intersectional_Attributes': intersectional_attributes if intersectional_attributes else []
    }
    return info

    # Function to group results by metric


def get_ideal_fairness_value(metric_name):
    if metric_name in ['disparate_impact']:
        return 1.0
    else:
        return 0.0


# Function to group results by metric
def group_results_by_metric(results):
    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result['Metric']].append({
            'Protected_Attributes': result['Protected_Attributes'],
            'Values': result['Values'],
            'Ideal_Fairness_Value': get_ideal_fairness_value(result['Metric']),
            'Privileged_Group': result['Privileged_Group'],
            'Unprivileged_Group': result['Unprivileged_Group'],
            'Intersectional_Attributes': result.get('Intersectional_Attributes', [])
        })
    return grouped_results


# Function to check fairness
def fair_check(metric_value, metric_name, threshold):
    ideal_value = 1 if metric_name in ['disparate_impact'] else 0

    try:
        threshold = float(threshold)
    except ValueError as e:
        return False

    if ideal_value == 1:
        return metric_value >= (threshold / 100) and metric_value <= (2 - (threshold / 100))
    elif ideal_value == 0:
        return metric_value <= (1 - (threshold / 100)) and metric_value >= (-1 + (threshold / 100))
    else:
        return False


# Function to train and evaluate the model
def train_and_evaluate(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate, threshold):
    X, y, label_encoders = prepare_data(path_to_csv)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(model_name, X_train, y_train)

    # Calculate model metrics
    model_metrics, y_pred = calculate_model_metrics(model, X_test, y_test)

    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(X_test, y_test, y_pred, label_encoders, atts_n_vals_picked,
                                                  metrics_to_calculate, threshold)

    return model_metrics, fairness_metrics


def get_mitigated_results(path_to_csv, model_name, atts_n_vals_picked, algorithms, biased_data, biased_model_data):
    foreach(algorithms)
    # the algorithms is an array create a foreach loop and exec exec the mitigate function ( you create it )
    # Depending of the algorithm picked execute the mitigation correctly
    # Create a reponse like the train_and_evaluate function but in the model_metrics will contain the biased_data and the mitigated value of each metric and the same for the model_data take a look at the construct_metric_info
    # use as many already created functions
    # create as many function as you like
    return
