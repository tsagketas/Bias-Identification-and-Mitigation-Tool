from typing import List, Union, Dict
import os
import json
import itertools
from collections import defaultdict
from tqdm import tqdm
import warnings
import sys
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report
from sklearn.exceptions import DataConversionWarning
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
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
    df = df.dropna(how='any', axis=0)
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
def train_model(model_name, X_train, y_train, weights=None):
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

    if weights is not None:
        model.fit(X_train, y_train, sample_weight=weights)
    else:
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


def get_privildged_group(att, label_encoders, intersectional=False, intersection_att=[]):
    if (intersectional):
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
    else:
        privileged_group = [{att['attribute']: label_encoders[att['attribute']].transform([att['privileged']])[0]}]
        unprivileged_group = [{att['attribute']: label_encoders[att['attribute']].transform([att['unprivileged']])[0]}]

    return privileged_group, unprivileged_group


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


def mitigate(algorithm, path_to_csv, model_name, atts_n_vals_picked, threshold, metrics_to_calculate):
    if algorithm == "Disparate Impact Remover":
        return disparate_impact_remover_result(path_to_csv, model_name, atts_n_vals_picked, threshold,
                                               metrics_to_calculate)
    elif algorithm == "Reweighing":
        return reweighing_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate)
    # elif algorithm == "Adversarial Debiasing":
    #     return adversarial_debiasing_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate,
    #                                         threshold)
    elif algorithm == "Prejudice Remover":
        return prejudice_remover_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate, threshold)
    # elif algorithm == "Calibrated Equality of Odds":
    #     return calibrated_eq_odds_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate, threshold)
    else:
        return [], []


def apply_dir_and_train_model(X, y, model_name, protected_attributes, intersectional=False):
    # Convert to binary label dataset
    bld = convert_to_binary_label_dataset(X, y, protected_attributes)

    # Apply Disparate Impact Remover (DIR)
    dir = DisparateImpactRemover(**({'sensitive_attribute': protected_attributes[0]} if not intersectional else {}),
                                 repair_level=0.5)

    bld_repaired = dir.fit_transform(bld)

    # Convert back to dataframe
    X_repaired = pd.DataFrame(bld_repaired.features, columns=X.columns)
    y_repaired = pd.Series(bld_repaired.labels.ravel(), name='outcome')

    # Split the repaired data
    X_train, X_test, y_train, y_test = train_test_split(X_repaired, y_repaired, test_size=0.2, random_state=42)

    # Train the model on repaired data
    model = train_model(model_name, X_train, y_train)

    model_metrics, y_pred = calculate_model_metrics(model, X_test, y_test)

    return model, X_test, y_test, y_pred, model_metrics


def disparate_impact_remover_result(path_to_csv, model_name, atts_n_vals_picked, threshold, metrics_to_calculate):
    # Prepare data
    X, y, label_encoders = prepare_data(path_to_csv)
    results = []
    model_metrics = {}

    for att in atts_n_vals_picked:
        protected_attributes = [att['attribute']]
        if 'intersection' in att and att['intersection']:
            for intersection_att in att['intersection']:
                protected_attributes.append(intersection_att['attribute'])
                # Apply DIR and train model
                model, X_test, y_test, y_pred, model_metrics = apply_dir_and_train_model(X, y, model_name,
                                                                                         protected_attributes, True)

                # Calculate intersectional metrics (assuming calculate_intersectional_metrics is defined)
                ground_truth_dataset, predicted_dataset = get_binary_datasets(X_test, y_test, y_pred,
                                                                              protected_attributes)

                results.extend(
                    calculate_intersectional_metrics(att, intersection_att, ground_truth_dataset, predicted_dataset,
                                                     label_encoders, metrics_to_calculate, threshold))
        else:
            model, X_test, y_test, y_pred, model_metrics = apply_dir_and_train_model(X, y, model_name,
                                                                                     protected_attributes)

            # Calculate standard metrics (assuming calculate_standard_metrics is defined)
            ground_truth_dataset, predicted_dataset = get_binary_datasets(X_test, y_test, y_pred,
                                                                          protected_attributes)

            results.extend(calculate_standard_metrics(att, ground_truth_dataset, predicted_dataset, label_encoders,
                                                      metrics_to_calculate, threshold))

    return results, model_metrics


def apply_reweighing_and_train_model(X, y, model_name, protected_attributes, att, label_encoders, intersectional=False,
                                     intersection_att=[]):
    bld = convert_to_binary_label_dataset(X, y, protected_attributes)

    privildged_group, unprivildged_group = get_privildged_group(att, label_encoders, intersectional, intersection_att)
    print(protected_attributes,privildged_group, unprivildged_group)
    RW = Reweighing(unprivileged_groups=unprivildged_group, privileged_groups=privildged_group)

    reweighted_data = RW.fit_transform(bld)

    # Split the BinaryLabelDataset
    train_dataset, test_dataset = reweighted_data.split([0.8], shuffle=True, seed=42)

    # Extract features and labels from the datasets
    X_train = pd.DataFrame(train_dataset.features, columns=X.columns)
    y_train = pd.Series(train_dataset.labels.ravel(), name='outcome')
    weights_train = train_dataset.instance_weights

    X_test = pd.DataFrame(test_dataset.features, columns=X.columns)
    y_test = pd.Series(test_dataset.labels.ravel(), name='outcome')
    weights_test = test_dataset.instance_weights

    # Train the model using the training data and weights
    model = train_model(model_name, X_train, y_train, weights_train)

    # Calculate model metrics and predictions
    model_metrics, y_pred = calculate_model_metrics(model, X_test, y_test)

    return model, X_test, y_test, y_pred, model_metrics


def reweighing_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate):
    X, y, label_encoders = prepare_data(path_to_csv)
    results = []
    model_metrics = {}

    for att in atts_n_vals_picked:
        protected_attributes = [att['attribute']]
        if 'intersection' in att and att['intersection']:
            for intersection_att in att['intersection']:
                # Apply DIR and train model
                protected_attributes.append(intersection_att['attribute'])
                model, X_test, y_test, y_pred, model_metrics = apply_reweighing_and_train_model(X, y, model_name,
                                                                                                protected_attributes,
                                                                                                att,
                                                                                                label_encoders,
                                                                                                True,
                                                                                                intersection_att)

                # Calculate intersectional metrics (assuming calculate_intersectional_metrics is defined)
                ground_truth_dataset, predicted_dataset = get_binary_datasets(X_test, y_test, y_pred,
                                                                              protected_attributes)

                results.extend(
                    calculate_intersectional_metrics(att, intersection_att, ground_truth_dataset, predicted_dataset,
                                                     label_encoders, metrics_to_calculate, threshold))
        else:
            model, X_test, y_test, y_pred, model_metrics = apply_reweighing_and_train_model(X, y, model_name,
                                                                                            protected_attributes,
                                                                                            att,
                                                                                            label_encoders)

            ground_truth_dataset, predicted_dataset = get_binary_datasets(X_test, y_test, y_pred,
                                                                          protected_attributes)

            results.extend(calculate_standard_metrics(att, ground_truth_dataset, predicted_dataset, label_encoders,
                                                      metrics_to_calculate, threshold))

    return results, model_metrics


def adversarial_debiasing_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate, threshold):
    X, y, label_encoders = prepare_data(path_to_csv)
    privileged_groups, unprivileged_groups = get_groups(atts_n_vals_picked, label_encoders)
    dataset = convert_to_binary_label_dataset(X, y, label_encoders)
    ad = AdversarialDebiasing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups,
                              scope_name='debiasing', debias=True)
    sess = tf.Session()
    ad.fit(dataset, sess)
    dataset_pred = ad.predict(dataset)
    X_pred = pd.DataFrame(dataset_pred.features, columns=X.columns)
    y_pred = pd.Series(dataset_pred.labels.ravel(), name='outcome')
    model_metrics, y_pred_labels = calculate_model_metrics(ad, X_pred, y_pred)
    fairness_metrics = calculate_fairness_metrics(X, y, y_pred_labels, label_encoders, atts_n_vals_picked,
                                                  metrics_to_calculate, threshold)
    return model_metrics, fairness_metrics


def apply_prejudice_remover_and_train_model(X, y, model_name,protected_attributes, att, threshold):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the training data to a binary label dataset
    train_df = pd.concat([X_train, y_train], axis=1)
    bld_train = BinaryLabelDataset(df=train_df,
                                   label_names=['outcome'],
                                   protected_attribute_names=protected_attributes)

    # Apply the PrejudiceRemover to the training data
    pr = PrejudiceRemover(sensitive_attr=att['attribute'], eta=25)
    pr.fit(bld_train)

    # Transform the test data to a binary label dataset
    test_df = pd.concat([X_test, y_test], axis=1)
    bld_test = BinaryLabelDataset(df=test_df,
                                  label_names=['outcome'],
                                  protected_attribute_names=protected_attributes)

    # Apply the PrejudiceRemover to the test data
    bld_test_repaired = pr.predict(bld_test)

    # Extract the repaired labels and features
    X_test_repaired = bld_test_repaired.features
    y_test_repaired = bld_test_repaired.labels

    # Ensure y_test_repaired is one-dimensional
    if y_test_repaired.ndim > 1:
        y_test_repaired = y_test_repaired.ravel()

    # Calculate model metrics: accuracy, precision, recall, F1-score
    model_metrics = {
        'accuracy': accuracy_score(y_test, y_test_repaired),
        'precision': precision_score(y_test, y_test_repaired),
        'recall': recall_score(y_test, y_test_repaired),
        'f1_score': f1_score(y_test, y_test_repaired)
    }

    return pr, X_test_repaired, y_test, y_test_repaired, model_metrics



def prejudice_remover_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate, threshold):
    X, y, label_encoders = prepare_data(path_to_csv)
    results = []
    model_metrics = {}

    for att in atts_n_vals_picked:
        protected_attributes = [att['attribute']]
        if 'intersection' in att and att['intersection']:
            for intersection_att in att['intersection']:
                protected_attributes.append(intersection_att['attribute'])
                # Apply Prejudice Remover and train model

                model, X_test, y_test, y_pred, model_metrics = apply_prejudice_remover_and_train_model(
                    X, y, model_name, protected_attributes, att, threshold)

                # Calculate intersectional metrics
                ground_truth_dataset, predicted_dataset = get_binary_datasets(X_test, y_test, y_pred,
                                                                              protected_attributes)
                results.extend(
                    calculate_intersectional_metrics(att, intersection_att, ground_truth_dataset, predicted_dataset,
                                                     label_encoders, metrics_to_calculate, threshold))
        else:
            # Apply Prejudice Remover and train model
            model, X_test, y_test, y_pred, model_metrics = apply_prejudice_remover_and_train_model(
                X, y, model_name, protected_attributes, att, threshold)

            # Calculate standard metrics
            ground_truth_dataset, predicted_dataset = get_binary_datasets(X_test, y_test, y_pred, protected_attributes)
            results.extend(calculate_standard_metrics(att, ground_truth_dataset, predicted_dataset, label_encoders,
                                                      metrics_to_calculate, threshold))

    return results, model_metrics


def calibrated_eq_odds_result(path_to_csv, model_name, atts_n_vals_picked, metrics_to_calculate, threshold):
    X, y, label_encoders = prepare_data(path_to_csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_model(model_name, X_train, y_train)
    model_metrics, y_pred = calculate_model_metrics(model, X_test, y_test)
    privileged_groups, unprivileged_groups = get_groups(atts_n_vals_picked, label_encoders)
    dataset_true = convert_to_binary_label_dataset(X_test, y_test, label_encoders)
    dataset_pred = convert_to_binary_label_dataset(X_test, y_pred, label_encoders)
    ceo = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups,
                                         cost_constraint='fnr')
    dataset_transf = ceo.fit_predict(dataset_true, dataset_pred)
    y_transf_pred = pd.Series(dataset_transf.labels.ravel(), name='outcome')
    fairness_metrics = calculate_fairness_metrics(X_test, y_test, y_transf_pred, label_encoders, atts_n_vals_picked,
                                                  metrics_to_calculate, threshold)
    return model_metrics, fairness_metrics


def get_mitigated_results(path_to_csv, model_name, atts_n_vals_picked, algorithms, biased_data, biased_model_data,
                          threshold):
    metrics_to_calculate = list(biased_data.keys())
    results = {}
    for algorithm in algorithms:
        fairness_metrics, model_metrics = mitigate(algorithm, path_to_csv, model_name, atts_n_vals_picked, threshold,
                                                   metrics_to_calculate)
        results[algorithm] = {
            'Model_Metrics': model_metrics,
            'Fairness_Metrics': group_results_by_metric(fairness_metrics)
        }
    return wrap_response(results, biased_data, biased_model_data)


def wrap_response(unbiased_data, biased_data, biased_model_data):
    results = {}

    for algorithm in unbiased_data.keys():
        unbiased_model_metrics = unbiased_data[algorithm]["Model_Metrics"]

        # Model Metrics
        model_metrics = {
            "unbiased_model": unbiased_model_metrics,
            "biased_model": biased_model_data
        }

        # Fairness Metrics
        unbiased_fairness_metrics = unbiased_data[algorithm]["Fairness_Metrics"]
        biased_fairness_metrics = biased_data

        fairness_metrics = {}
        for metric, values in unbiased_fairness_metrics.items():
            fairness_metrics[metric] = []
            for index, item in enumerate(values):
                item_copy = item.copy()  # Make a copy to avoid modifying the original data
                item_copy["unbiased_Values"] = item_copy.pop("Values", None)
                if metric in biased_fairness_metrics and index < len(biased_fairness_metrics[metric]):
                    item_copy["biased_Values"] = biased_fairness_metrics[metric][index].get("Values")
                else:
                    item_copy["biased_Values"] = None
                fairness_metrics[metric].append(item_copy)

        results[algorithm] = {
            "Model_Metrics": model_metrics,
            "Fairness_Metrics": fairness_metrics
        }

    return results


# Example data for unbiased_data, biased_data, and biased_model_data
unbiased_data = {
    "Disparate Impact Remover": {
        "Model_Metrics": {
            "accuracy": 0.8638550516941345,
            "f1_score": 0.6889616463985033,
            "precision": 0.7416918429003021,
            "recall": 0.643231441048035
        },
        "Fairness_Metrics": {
            "disparate_impact": [
                {
                    "Protected_Attributes": ["gender"],
                    "Values": 0.2949893377973472,
                    "Ideal_Fairness_Value": 1.0,
                    "Privileged_Group": "Male",
                    "Unprivileged_Group": "Female",
                    "Intersectional_Attributes": []
                },
                {
                    "Protected_Attributes": ["race", "gender"],
                    "Values": 0.11605360707157113,
                    "Ideal_Fairness_Value": 1.0,
                    "Privileged_Group": "White",
                    "Unprivileged_Group": "Black",
                    "Intersectional_Attributes": [
                        {
                            "attribute": "gender",
                            "privileged": "Male",
                            "unprivileged": "Female"
                        }
                    ]
                }
            ],
            "mean_difference": [
                {
                    "Protected_Attributes": ["race", "gender"],
                    "Values": -0.2432481474397642,
                    "Ideal_Fairness_Value": 0.0,
                    "Privileged_Group": "White",
                    "Unprivileged_Group": "Black",
                    "Intersectional_Attributes": [
                        {
                            "attribute": "gender",
                            "privileged": "Male",
                            "unprivileged": "Female"
                        }
                    ]
                }
            ]
        }
    },
    # it may contain other keys algoriths too but with the same structure
}

# Example usage:
path_to_csv = "adult.csv"
threshold = "80"
model_name = "random_forest"
atts_n_vals_picked = [{'attribute': 'gender', 'privileged': 'Male', 'unprivileged': 'Female', 'intersection': []},
                      {'attribute': 'race', 'privileged': 'White', 'unprivileged': 'Black',
                       'intersection': [{'attribute': 'gender', 'privileged': 'Male', 'unprivileged': 'Female'}]}]
# algorithms = ['Disparate Impact Remover', 'Reweighing', 'Adversarial Debiasing', 'Prejudice Remover',
#               'Calibrated Equality of Odds']

algorithms = ['Prejudice Remover']
biased_data = {
    'disparate_impact': [
        {
            'Protected_Attributes': ['gender'],
            'Values': 0.2970535490163289,
            'Ideal_Fairness_Value': 1.0,
            'Privileged_Group': 'Male',
            'Unprivileged_Group': 'Female',
            'Intersectional_Attributes': []
        },
        {
            'Protected_Attributes': ['race', 'gender'],
            'Values': 0.11524512722496741,
            'Ideal_Fairness_Value': 1.0,
            'Privileged_Group': 'White',
            'Unprivileged_Group': 'Black',
            'Intersectional_Attributes': [{'attribute': 'gender', 'privileged': 'Male', 'unprivileged': 'Female'}]
        }
    ],
    'mean_difference': [
        {
            'Protected_Attributes': ['race', 'gender'],
            'Values': -0.24517864937026612,
            'Ideal_Fairness_Value': 0.0,
            'Privileged_Group': 'White',
            'Unprivileged_Group': 'Black',
            'Intersectional_Attributes': [{'attribute': 'gender', 'privileged': 'Male', 'unprivileged': 'Female'}]
        }
    ]
}
biased_model_data = {
    'accuracy': 0.8635,
    'f1_score': 0.6876,
    'precision': 0.742,
    'recall': 0.6406
}

# results= wrap_response(unbiased_data,biased_data,biased_model_data)

results = get_mitigated_results(path_to_csv, model_name, atts_n_vals_picked, algorithms, biased_data, biased_model_data,
                                threshold)

print(results)
