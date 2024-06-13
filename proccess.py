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
from sklearn.preprocessing import LabelEncoder
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
from aif360.datasets import BinaryLabelDataset, StandardDataset  # To handle the data
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric  # For calculating metrics
from aif360.explainers import MetricTextExplainer  # For explaining metrics
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover, MetaFairClassifier
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, \
    RejectOptionClassification

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# For the logistic regression model
from sklearn.preprocessing import StandardScaler


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


def preprocess_dataset(df, catCols, numCols, targets):
    # atts=[]

    df = df.fillna(df.mean())

    for col in catCols:
        df[col] = df[col].fillna("Missing value")

    # for col in catCols:
    #     for c in df[col].values:
    #         if (col+":"+c) not in atts:
    #             colname=col+":"+ c
    #             atts.append(colname)

    # for col in numCols:
    #     for c in df[col].values:
    #         if (col+":"+str(c)) not in atts:
    #             colname=col+":"+str(c)
    #             atts.append(colname)            

    for col in catCols:
        df_onehot = pd.concat([df[col], pd.get_dummies(df[col])], axis=1)
        df_onehot = df_onehot.drop([col], axis=1)
        df = df.drop([col], axis=1)
        df = pd.concat([df, df_onehot], axis=1)

    # for col in numCols:
    #     df_onehot = pd.concat([df[col], pd.get_dummies(df[col]).rename(columns=lambda x:col)], axis=1)
    #     df_onehot=df_onehot.drop([col], axis=1)
    #     df=df.drop([col], axis=1)
    #     df=pd.concat([df,df_onehot],axis=1)    

    df2 = pd.concat([targets['Label_value'], df], axis=1)
    df2 = df2.rename(columns={'Label_value': 'Score'})
    df = pd.concat([targets['Score'], df], axis=1)

    return df, df2


def fair_check(metric_value, ideal_value, threshold):
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


def construct_metric_info(metric_name, metric_value, ideal_value, protected_attributes, privileged_group,
                          unprivileged_group, intersectional_attributes=None):
    info = {
        'Metric': metric_name,
        'Protected_Attributes': protected_attributes,
        'Values': metric_value,
        'Ideal_Fairness_Value': ideal_value,
        'Privileged_Group': privileged_group,
        'Unprivileged_Group': unprivileged_group
    }
    if intersectional_attributes:
        info['Intersectional_Attributes'] = intersectional_attributes
    return info


def encode_categorical_variables(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders


def calculate_intersectional_metrics(att, intersection_att, df, label_encoders, threshold):
    results = []
    privileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['privileged']])[0]}]
    unprivileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['unprivileged']])[0]}]

    filtered_df = df[(df[att['attribute']].isin(
        [privileged_groups[0][att['attribute']], unprivileged_groups[0][att['attribute']]])) &
                     (df[intersection_att['attribute']].isin(
                         [label_encoders[intersection_att['attribute']].transform([intersection_att['privileged']])[0],
                          label_encoders[intersection_att['attribute']].transform([intersection_att['unprivileged']])[
                              0]]))]

    binary_label_dataset_intersection = BinaryLabelDataset(
        df=filtered_df,
        label_names=['Label_value'],
        protected_attribute_names=[att['attribute'], intersection_att['attribute']]
    )

    unprivileged_intersection_groups = [{att['attribute']: unprivileged_groups[0][att['attribute']],
                                         intersection_att['attribute']:
                                             label_encoders[intersection_att['attribute']].transform(
                                                 [intersection_att['unprivileged']])[0]}]
    privileged_intersection_groups = [{att['attribute']: privileged_groups[0][att['attribute']],
                                       intersection_att['attribute']:
                                           label_encoders[intersection_att['attribute']].transform(
                                               [intersection_att['privileged']])[0]}]

    metric_intersection = BinaryLabelDatasetMetric(
        binary_label_dataset_intersection,
        unprivileged_groups=unprivileged_intersection_groups,
        privileged_groups=privileged_intersection_groups
    )

    metrics_to_calculate_intersection = {
        "mean_difference": metric_intersection.mean_difference,
        "disparate_impact": metric_intersection.disparate_impact,
    }

    for metric_name, metric_func in metrics_to_calculate_intersection.items():
        metric_value = metric_func()
        if not fair_check(metric_value, 0 if metric_name == 'mean_difference' else 1, threshold):
            results.append(construct_metric_info(
                metric_name,
                metric_value,
                0 if metric_name == 'mean_difference' else 1,
                [att['attribute'], intersection_att['attribute']],
                att['privileged'],
                att['unprivileged'],
                intersectional_attributes=[{
                    'attribute': intersection_att['attribute'],
                    'privileged': intersection_att['privileged'],
                    'unprivileged': intersection_att['unprivileged']
                }]
            ))

    return results


def calculate_standard_metrics(att, df, label_encoders, metrics_to_calculate, threshold):
    results = []
    privileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['privileged']])[0]}]
    unprivileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['unprivileged']])[0]}]

    binary_label_dataset = BinaryLabelDataset(df=df,
                                              label_names=['Label_value'],
                                              protected_attribute_names=[att['attribute']])

    df_pred = df.copy()
    df_pred['Label_value'] = df_pred['Score']

    predicted_dataset = BinaryLabelDataset(df=df_pred,
                                           label_names=['Label_value'],
                                           protected_attribute_names=[att['attribute']])

    metric = BinaryLabelDatasetMetric(binary_label_dataset,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)

    classified_metric = ClassificationMetric(binary_label_dataset, predicted_dataset,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metrics_to_calculate_funcs = {
        'mean_difference': (metric.mean_difference, 0),
        'disparate_impact': (metric.disparate_impact, 1),
        'false_discovery_rate_difference': (classified_metric.false_discovery_rate_difference, 0),
        'false_positive_rate_difference': (classified_metric.false_positive_rate_difference, 0),
        'false_omission_rate_difference': (classified_metric.false_omission_rate_difference, 0),
        'false_negative_rate_difference': (classified_metric.false_negative_rate_difference, 0),
        'average_odds_difference': (classified_metric.average_odds_difference, 0),
        'equal_opportunity_difference': (classified_metric.equal_opportunity_difference, 0)
    }

    for metric_name in metrics_to_calculate:
        if metric_name in metrics_to_calculate_funcs:
            metric_func, ideal_value = metrics_to_calculate_funcs[metric_name]
            metric_value = metric_func()
            if not fair_check(metric_value, ideal_value, threshold):
                results.append(construct_metric_info(
                    metric_name,
                    metric_value,
                    ideal_value,
                    [att['attribute']],
                    att['privileged'],
                    att['unprivileged']
                ))

    return results


def get_fairness_metrics(atts_n_vals, path_to_csv, metrics_to_calculate, threshold):
    df = pd.read_csv(path_to_csv)
    df, label_encoders = encode_categorical_variables(df)
    results = []

    for att in atts_n_vals:
        if 'intersection' in att and att['intersection']:
            for intersection_att in att['intersection']:
                results.extend(calculate_intersectional_metrics(att, intersection_att, df, label_encoders, threshold))
        else:
            results.extend(calculate_standard_metrics(att, df, label_encoders, metrics_to_calculate, threshold))

    return results


# def get_fairness_metrics(atts_n_vals, path_to_csv, metrics_to_calculate,threshold):
#     # Load the dataset
#     df = pd.read_csv(path_to_csv)
#
#
#     # Encode categorical variables
#     label_encoders = {}
#     for column in df.select_dtypes(include=['object']).columns:
#         le = LabelEncoder()
#         df[column] = le.fit_transform(df[column])
#         label_encoders[column] = le
#
#     results = {}
#
#     for att in atts_n_vals:
#         privileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['privileged']])[0]}]
#         unprivileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['unprivileged']])[0]}]
#
#         # Check if intersectional attributes are provided
#         if 'intersection' in att:
#             for intersection_att in att['intersection']:
#                 # Filter the DataFrame for intersectional analysis
#                 filtered_df = df[(df[att['attribute']].isin([privileged_groups[0][att['attribute']], unprivileged_groups[0][att['attribute']]])) &
#                                  (df[intersection_att['attribute']].isin([label_encoders[intersection_att['attribute']].transform([intersection_att['privileged']])[0],
#                                                                           label_encoders[intersection_att['attribute']].transform([intersection_att['unprivileged']])[0]]))]
#
#                 binary_label_dataset_intersection = BinaryLabelDataset(
#                     df=filtered_df,
#                     label_names=['Label_value'],
#                     protected_attribute_names=[att['attribute'], intersection_att['attribute']]
#                 )
#
#                 unprivileged_intersection_groups = [{att['attribute']: unprivileged_groups[0][att['attribute']],
#                                                      intersection_att['attribute']: label_encoders[intersection_att['attribute']].transform([intersection_att['unprivileged']])[0]}]
#                 privileged_intersection_groups = [{att['attribute']: privileged_groups[0][att['attribute']],
#                                                    intersection_att['attribute']: label_encoders[intersection_att['attribute']].transform([intersection_att['privileged']])[0]}]
#
#                 metric_intersection = BinaryLabelDatasetMetric(
#                     binary_label_dataset_intersection,
#                     unprivileged_groups=unprivileged_intersection_groups,
#                     privileged_groups=privileged_intersection_groups
#                 )
#
#                 results[f"{att['attribute']}_{intersection_att['attribute']}_mean_difference"] = metric_intersection.mean_difference()
#                 results[f"{att['attribute']}_{intersection_att['attribute']}_disparate_impact"] = metric_intersection.disparate_impact()
#
#         else:
#             # Create BinaryLabelDataset for true labels
#             binary_label_dataset = BinaryLabelDataset(df=df,
#                                                       label_names=['Label_value'],
#                                                       protected_attribute_names=[att['attribute']])
#
#             # Create a copy of the dataset for predictions
#             df_pred = df.copy()
#             df_pred['Label_value'] = df_pred['Score']
#
#             # Create BinaryLabelDataset for predicted scores
#             predicted_dataset = BinaryLabelDataset(df=df_pred,
#                                                    label_names=['Label_value'],
#                                                    protected_attribute_names=[att['attribute']])
#
#             # Initialize the metric objects
#             metric = BinaryLabelDatasetMetric(binary_label_dataset,
#                                               unprivileged_groups=unprivileged_groups,
#                                               privileged_groups=privileged_groups)
#
#             classified_metric = ClassificationMetric(binary_label_dataset, predicted_dataset,
#                                                      unprivileged_groups=unprivileged_groups,
#                                                      privileged_groups=privileged_groups)
#
#             # Calculate the specified metrics
#             for metric_name in metrics_to_calculate:
#                 if metric_name == 'mean_difference':
#                     results[f"{att['attribute']}_mean_difference"] = metric.mean_difference()
#                 elif metric_name == 'disparate_impact':
#                     results[f"{att['attribute']}_disparate_impact"] = metric.disparate_impact()
#                 elif metric_name == 'false_discovery_rate_difference':
#                     results[f"{att['attribute']}_false_discovery_rate_difference"] = classified_metric.false_discovery_rate_difference()
#                 elif metric_name == 'false_positive_rate_difference':
#                     results[f"{att['attribute']}_false_positive_rate_difference"] = classified_metric.false_positive_rate_difference()
#                 elif metric_name == 'false_omission_rate_difference':
#                     results[f"{att['attribute']}_false_omission_rate_difference"] = classified_metric.false_omission_rate_difference()
#                 elif metric_name == 'false_negative_rate_difference':
#                     results[f"{att['attribute']}_false_negative_rate_difference"] = classified_metric.false_negative_rate_difference()
#                 elif metric_name == 'average_odds_difference':
#                     results[f"{att['attribute']}_average_odds_difference"] = classified_metric.average_odds_difference()
#                 elif metric_name == 'equal_opportunity_difference':
#                     results[f"{att['attribute']}_equal_opportunity_difference"] = classified_metric.equal_opportunity_difference()
#
#     return results
#
# def fair_check(metric,objective,threshold):
#     if objective == 0:
#         if metric >= 0:
#             return  metric  <= (1 - threshold )
#         else:
#             return metric  >= (-1 + threshold )
#     else:
#         return ( (threshold <= metric) and (metric <= (2 - threshold)) )

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
