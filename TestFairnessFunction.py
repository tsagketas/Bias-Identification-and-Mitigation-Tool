import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import json
from collections import defaultdict

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

    # Encode the privileged and unprivileged groups for primary and intersectional attributes
    privileged_primary = label_encoders[att['attribute']].transform([att['privileged']])[0]
    unprivileged_primary = label_encoders[att['attribute']].transform([att['unprivileged']])[0]

    privileged_intersection = label_encoders[intersection_att['attribute']].transform([intersection_att['privileged']])[
        0]
    unprivileged_intersection = \
    label_encoders[intersection_att['attribute']].transform([intersection_att['unprivileged']])[0]

    # Create the BinaryLabelDataset with the entire dataset
    binary_label_dataset_intersection = BinaryLabelDataset(
        df=df,
        label_names=['Label_value'],
        protected_attribute_names=[att['attribute'], intersection_att['attribute']]
    )

    # Define the unprivileged and privileged intersection groups
    unprivileged_intersection_groups = [
        {
            att['attribute']: unprivileged_primary,
            intersection_att['attribute']: unprivileged_intersection
        }
    ]

    privileged_intersection_groups = [
        {
            att['attribute']: privileged_primary,
            intersection_att['attribute']: privileged_intersection
        }
    ]

    # Instantiate the metric object for the intersectional dataset
    metric_intersection = BinaryLabelDatasetMetric(
        binary_label_dataset_intersection,
        unprivileged_groups=unprivileged_intersection_groups,
        privileged_groups=privileged_intersection_groups
    )

    # Define the metrics to calculate
    metrics_to_calculate_intersection = {
        "mean_difference": metric_intersection.mean_difference,
        "disparate_impact": metric_intersection.disparate_impact,
    }

    # Calculate and check each metric
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



# def calculate_intersectional_metrics(att, intersection_att, df, label_encoders, threshold):
#     results = []
#     privileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['privileged']])[0]}]
#     unprivileged_groups = [{att['attribute']: label_encoders[att['attribute']].transform([att['unprivileged']])[0]}]
#
#     filtered_df = df[(df[att['attribute']].isin(
#         [privileged_groups[0][att['attribute']], unprivileged_groups[0][att['attribute']]])) &
#                      (df[intersection_att['attribute']].isin(
#                          [label_encoders[intersection_att['attribute']].transform([intersection_att['privileged']])[0],
#                           label_encoders[intersection_att['attribute']].transform([intersection_att['unprivileged']])[
#                               0]]))]
#
#     binary_label_dataset_intersection = BinaryLabelDataset(
#         df=filtered_df,
#         label_names=['Label_value'],
#         protected_attribute_names=[att['attribute'], intersection_att['attribute']]
#     )
#
#     unprivileged_intersection_groups = [{att['attribute']: unprivileged_groups[0][att['attribute']],
#                                          intersection_att['attribute']:
#                                              label_encoders[intersection_att['attribute']].transform(
#                                                  [intersection_att['unprivileged']])[0]}]
#     privileged_intersection_groups = [{att['attribute']: privileged_groups[0][att['attribute']],
#                                        intersection_att['attribute']:
#                                            label_encoders[intersection_att['attribute']].transform(
#                                                [intersection_att['privileged']])[0]}]
#
#     metric_intersection = BinaryLabelDatasetMetric(
#         binary_label_dataset_intersection,
#         unprivileged_groups=unprivileged_intersection_groups,
#         privileged_groups=privileged_intersection_groups
#     )
#
#     metrics_to_calculate_intersection = {
#         "mean_difference": metric_intersection.mean_difference,
#         "disparate_impact": metric_intersection.disparate_impact,
#     }
#
#     for metric_name, metric_func in metrics_to_calculate_intersection.items():
#         metric_value = metric_func()
#         if not fair_check(metric_value, 0 if metric_name == 'mean_difference' else 1, threshold):
#             results.append(construct_metric_info(
#                 metric_name,
#                 metric_value,
#                 0 if metric_name == 'mean_difference' else 1,
#                 [att['attribute'], intersection_att['attribute']],
#                 att['privileged'],
#                 att['unprivileged'],
#                 intersectional_attributes=[{
#                     'attribute': intersection_att['attribute'],
#                     'privileged': intersection_att['privileged'],
#                     'unprivileged': intersection_att['unprivileged']
#                 }]
#             ))
#
#     return results

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

    return group_results_by_metric(results)

def group_results_by_metric(results):
    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result['Metric']].append({
            'Protected_Attributes': result['Protected_Attributes'],
            'Values': result['Values'],
            'Ideal_Fairness_Value': result['Ideal_Fairness_Value'],
            'Privileged_Group': result['Privileged_Group'],
            'Unprivileged_Group': result['Unprivileged_Group'],
            'Intersectional_Attributes': result.get('Intersectional_Attributes', [])
        })
    return grouped_results

# Define test values
atts_n_vals_picked = [
    {'attribute': 'race', 'privileged': 'White', 'unprivileged': 'Black', 'intersection': [{'attribute': 'gender', 'privileged': 'Male', 'unprivileged': 'Female'}]},
    {'attribute': 'gender', 'privileged': 'Male', 'unprivileged': 'Female', 'intersection': []}
]
path_to_csv = 'adult_predictions_with_original_data.csv'
metrics_to_calculate = [
    'mean_difference',
    'disparate_impact',
    'false_discovery_rate_difference',
    'false_positive_rate_difference',
    'false_omission_rate_difference',
    'false_negative_rate_difference',
    'average_odds_difference',
    'equal_opportunity_difference'
]
threshold = 80

# Call the function with the test values
results_json = get_fairness_metrics(atts_n_vals_picked, path_to_csv, metrics_to_calculate, threshold)

# Print the JSON formatted grouped results
print(results_json,type(results_json))
