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


class MetricAdditions:
    def explain(self,
                disp: bool = True) -> Union[None, str]:
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


# Function to prepare the data
def prepare_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    df, label_encoders = encode_categorical_variables(df)
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    return X, y, label_encoders


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
def convert_to_binary_label_dataset(X, y, label_encoders, label_name='label'):
    df = X.copy()
    df[label_name] = y
    return BinaryLabelDataset(df=df, label_names=[label_name], protected_attribute_names=list(label_encoders.keys()))


# Function to calculate fairness metrics
def calculate_fairness_metrics(X, y_true, y_pred, label_encoders, atts_n_vals_picked, metrics_to_calculate, threshold):
    results = []
    ground_truth_dataset = convert_to_binary_label_dataset(X, y_true, label_encoders, label_name='true_label')
    predicted_dataset = ground_truth_dataset.copy()
    predicted_dataset.labels = y_pred.reshape(-1, 1)

    for att in atts_n_vals_picked:
        if 'intersection' in att and att['intersection']:
            for intersection_att in att['intersection']:
                results.extend(
                    calculate_intersectional_metrics(att, intersection_att, ground_truth_dataset, predicted_dataset,
                                                     label_encoders, metrics_to_calculate, threshold))
        else:
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
        return metric_value >= (threshold / 100)
    elif ideal_value == 0:
        return metric_value <= (threshold / 100)
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


def get_data(the_metrics, the_b_metrics, fairness_metrics, attributes, atts_values, threshold, dataset):
    # na prosthesw threshold
    x = []
    objective_1 = ["disparate_impact"]
    flag = True

    for metric in fairness_metrics:

        varbs = []
        values = []
        variables = []

        for i in range(len(the_metrics)):
            if metric == "disparate_impact" or metric == "mean_difference":
                value = getattr(the_b_metrics[i], metric)()

            else:
                value = getattr(the_metrics[i], metric)()

            if ((not math.isinf(value)) and (math.isnan(value) == False)):
                if (metric in objective_1):
                    objective = 1
                else:
                    objective = 0.0

                if dataset == "upload":
                    if (not fair_check(value, objective, (int(threshold) / 100))):
                        flag = False
                        values.append(value)
                        # stuff_in_string = f"Protected Attribute {attributes[i]} with value  {atts_values[i]}"
                        varbs.append(f"{attributes[i]}:{atts_values[i]}")
                        variables.append(atts_values[i])
                else:
                    flag = False
                    values.append(value)
                    variables.append(atts_values[i])
                    varbs.append(f"{attributes[i]}:{atts_values[i]}")

        if (len(variables) != 0 or len(values) != 0):
            x.append({'Metric': metric, 'Protected_Attributes': variables, 'Values': values, "Atts_to_show": varbs})

    return x, flag


def save_conf(y_test, y_pred, method):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="PuBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(bottom=.05, left=.15)
    if method == "Reweighing":
        fig.savefig('static/Reweighing.png')
    elif method == "Disparate Impact Remover":
        fig.savefig('static/Disparate_Impact_Remover.png')
    elif method == "Adversarial Debiasing":
        fig.savefig('static/AdversarialDebiasing.png')
    elif method == "Meta Fair Classifier":
        fig.savefig('static/MetaFairClassifier.png')
    elif method == "Calibrated Equality of Odds":
        fig.savefig('static/CalibratedEqualityofOdds.png')
    else:
        fig.savefig('static/EqualityofOdds.png')


def mitigation_all(to_json, df_truth, df_preds, methods):
    mitigation_json = []
    model_info = []
    model_info2 = []
    temp = []
    counter = 0
    flag1 = flag2 = flag3 = flag4 = flag5 = True
    for method in methods:
        temp = {'Method': method, 'Data': []}
        values = []
        for element in to_json:
            values = []
            mitigation_json_data = {'Metric': element['Metric'], 'ID': counter,
                                    'Protected_Attributes': element['Protected_Attributes'], 'Values': values,
                                    'Biased_values': element['Values'], "Atts_to_show": element['Atts_to_show']}
            for att in element['Protected_Attributes']:

                dataset = BinaryLabelDataset(df=df_truth,
                                             label_names=['Score'],
                                             protected_attribute_names=[att],
                                             favorable_label=1,
                                             unfavorable_label=0,
                                             )

                dataset_pred = BinaryLabelDataset(df=df_preds,
                                                  label_names=['Score'],
                                                  protected_attribute_names=[att],
                                                  favorable_label=1,
                                                  unfavorable_label=0,
                                                  )

                unprivileged_groups = [{att: 0}]
                privileged_groups = [{att: 1}]
                rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                rw_label_data = rw.fit_transform(dataset)
                # dataset_train, dataset_test =dataset.split([0.7], shuffle=True)
                train, test = train_test_split(df_truth, shuffle=True, test_size=0.40)

                dataset_train = BinaryLabelDataset(df=train,
                                                   label_names=['Score'],
                                                   protected_attribute_names=[att],
                                                   favorable_label=1,
                                                   unfavorable_label=0,
                                                   )

                dataset_test = BinaryLabelDataset(df=test,
                                                  label_names=['Score'],
                                                  protected_attribute_names=[att],
                                                  favorable_label=1,
                                                  unfavorable_label=0,
                                                  )
                # dataset_train, dataset_test =dataset.split([0.7], shuffle=True)
                scale_orig = StandardScaler()

                if (method == "Reweighing"):

                    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                    rw_label_data = rw.fit_transform(dataset)

                    X_train = scale_orig.fit_transform(dataset_train.features)
                    y_train = dataset_train.labels.ravel()

                    rw_label_data = rw.fit_transform(dataset_train)

                    model = LogisticRegression()
                    model.fit(X_train, y_train, sample_weight=rw_label_data.instance_weights)

                    x_test = scale_orig.fit_transform(dataset_test.features)
                    y_test = dataset_test.labels.ravel()

                    y_test_pred = model.predict(x_test)
                    arr = np.concatenate((dataset_train.labels, y_test_pred.reshape(-1, 1)))
                    data_all = dataset.copy()
                    data_all.labels = arr

                    if (element['Metric'] == 'mean_difference' or element['Metric'] == 'disparate_impact'):
                        blm = BinaryLabelDatasetMetric(rw_label_data, unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
                        value = getattr(blm, element['Metric'])()
                    else:
                        preds = ClassificationMetric(dataset, data_all,
                                                     unprivileged_groups=[{att: 0}],
                                                     privileged_groups=[{att: 1}],
                                                     )
                        value = getattr(preds, element['Metric'])()

                    values.append(value)
                    if flag1:
                        tn, fp, fn, tp = confusion_matrix(dataset.labels, data_all.labels).ravel()
                        temp2 = {'Accuracy': metrics.accuracy_score(dataset.labels, data_all.labels),
                                 'Precision': metrics.precision_score(dataset.labels, data_all.labels),
                                 'Recall': metrics.recall_score(dataset.labels, data_all.labels), 'True negatives': tn,
                                 'False positives': fp, 'False negatives': fn, 'True positives': tp, 'Method': method}
                        model_info.append(temp2)
                        flag1 = False


                elif (method == "Disparate Impact Remover"):

                    di = DisparateImpactRemover()
                    dataset_train = di.fit_transform(dataset_train)
                    dataset_test = di.fit_transform(dataset_test)

                    X_train = scale_orig.fit_transform(dataset_train.features)
                    y_train = dataset_train.labels.ravel()

                    model = LogisticRegression()
                    model.fit(X_train, y_train)

                    x_test = scale_orig.fit_transform(dataset_test.features)
                    y_test = dataset_test.labels.ravel()
                    y_test_pred = model.predict(x_test)

                    arr = np.concatenate((dataset_train.labels, y_test_pred.reshape(-1, 1)))
                    data_all = dataset.copy()
                    data_all.labels = arr

                    if (element['Metric'] == 'mean_difference' or element['Metric'] == 'disparate_impact'):
                        blm = BinaryLabelDatasetMetric(data_all, unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
                        value = getattr(blm, element['Metric'])()
                    else:
                        preds = ClassificationMetric(dataset, data_all,
                                                     unprivileged_groups=[{att: 0}],
                                                     privileged_groups=[{att: 1}],
                                                     )
                        value = getattr(preds, element['Metric'])()

                    values.append(value)

                    # if (element['Metric']=='mean_difference' or element['Metric']=='disparate_impact'):
                    #     value=getattr(blm, element['Metric'])()
                    # else:
                    #     value=getattr(preds, element['Metric'])()
                    # values.append(value)

                    # tn, fp, fn, tp = confusion_matrix(dataset.labels, data_all.labels).ravel()
                    # temp2={'Attribute':att ,'Accuracy': metrics.accuracy_score(dataset.labels, data_all.labels), 'Precision':metrics.precision_score(dataset.labels, data_all.labels),'Recall': metrics.recall_score(dataset.labels, data_all.labels),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp,'Method':method}
                    # model_info2.append(temp2)

                    if flag2:
                        tn, fp, fn, tp = confusion_matrix(dataset.labels, data_all.labels).ravel()
                        temp2 = {'Accuracy': metrics.accuracy_score(dataset.labels, data_all.labels),
                                 'Precision': metrics.precision_score(dataset.labels, data_all.labels),
                                 'Recall': metrics.recall_score(dataset.labels, data_all.labels), 'True negatives': tn,
                                 'False positives': fp, 'False negatives': fn, 'True positives': tp, 'Method': method}
                        model_info.append(temp2)
                        flag2 = False

                elif (method == "Adversarial Debiasing"):
                    tf.reset_default_graph()
                    sess = tf.Session()

                    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                                          unprivileged_groups=unprivileged_groups,
                                                          scope_name='plain_classifier',
                                                          debias=False,
                                                          sess=sess)

                    debiased_model.fit(dataset_train)
                    y_test_pred = debiased_model.predict(dataset_test)
                    # y_train_pred = debiased_model.predict(dataset_train)
                    whole1 = train.append(test, ignore_index=True)
                    test['Score'] = y_test_pred.labels.reshape(-1, 1)
                    # train['Score']=y_train_pred.labels.reshape(-1,1)
                    whole2 = train.append(test, ignore_index=True)

                    bias = BinaryLabelDataset(df=whole1,
                                              label_names=['Score'],
                                              protected_attribute_names=[att],
                                              favorable_label=1,
                                              unfavorable_label=0,
                                              )

                    debiased = BinaryLabelDataset(df=whole2,
                                                  label_names=['Score'],
                                                  protected_attribute_names=[att],
                                                  favorable_label=1,
                                                  unfavorable_label=0,
                                                  )
                    if (element['Metric'] == 'mean_difference' or element['Metric'] == 'disparate_impact'):
                        blm = BinaryLabelDatasetMetric(debiased, unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
                        value = getattr(blm, element['Metric'])()
                    else:
                        preds = ClassificationMetric(bias, debiased,
                                                     unprivileged_groups=[{att: 0}],
                                                     privileged_groups=[{att: 1}],
                                                     )
                        value = getattr(preds, element['Metric'])()

                    values.append(value)
                    sess.close()

                    # tn, fp, fn, tp = confusion_matrix(bias.labels,debiased.labels).ravel()
                    # temp2={'Attribute':att ,'Accuracy': metrics.accuracy_score(bias.labels,debiased.labels), 'Precision':metrics.precision_score(bias.labels,debiased.labels),'Recall': metrics.recall_score(bias.labels,debiased.labels),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp,'Method':method}
                    # print(temp2)

                    if flag3:
                        tn, fp, fn, tp = confusion_matrix(bias.labels, debiased.labels).ravel()
                        temp2 = {'Accuracy': metrics.accuracy_score(bias.labels, debiased.labels),
                                 'Precision': metrics.precision_score(bias.labels, debiased.labels),
                                 'Recall': metrics.recall_score(bias.labels, debiased.labels), 'True negatives': tn,
                                 'False positives': fp, 'False negatives': fn, 'True positives': tp, 'Method': method}
                        model_info.append(temp2)
                        flag3 = False

                else:

                    dataset2 = BinaryLabelDataset(df=df_preds,
                                                  label_names=['Score'],
                                                  protected_attribute_names=[att],
                                                  favorable_label=1,
                                                  unfavorable_label=0,
                                                  )

                    cost_constraint = "weighted"
                    randseed = 12345679

                    CPP = CalibratedEqOddsPostprocessing(privileged_groups=[{att: 1}],
                                                         unprivileged_groups=[{att: 0}],
                                                         cost_constraint=cost_constraint,
                                                         seed=randseed)

                    CPP = CPP.fit(dataset, dataset2)
                    y_test_pred = CPP.predict(dataset2)

                    if (element['Metric'] == 'mean_difference' or element['Metric'] == 'disparate_impact'):
                        what2 = BinaryLabelDatasetMetric(y_test_pred, unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
                    else:
                        what2 = ClassificationMetric(y_test_pred,
                                                     dataset2,  # i to allo me ta preds
                                                     unprivileged_groups=[{att: 0}],
                                                     privileged_groups=[{att: 1}],
                                                     )

                    value = getattr(what2, element['Metric'])()
                    values.append(value)

                    if flag5:
                        tn, fp, fn, tp = confusion_matrix(y_test_pred.labels, dataset2.labels).ravel()
                        temp2 = {'Accuracy': metrics.accuracy_score(y_test_pred.labels, dataset2.labels),
                                 'Precision': metrics.precision_score(y_test_pred.labels, dataset2.labels),
                                 'Recall': metrics.recall_score(y_test_pred.labels, dataset2.labels),
                                 'True negatives': tn, 'False positives': fp, 'False negatives': fn,
                                 'True positives': tp, 'Method': method}
                        model_info.append(temp2)
                        flag5 = False

            mitigation_json_data['Values'] = values
            temp['Data'].append(mitigation_json_data)
            counter = counter + 1

        mitigation_json.append(temp)

    return mitigation_json, model_info
