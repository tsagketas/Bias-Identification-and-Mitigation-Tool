import pandas as pd
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df, protected_attr_name, label_name):
    # Ensure the protected attribute is binary and create the necessary columns
    if df[protected_attr_name].dtype != 'int64' and df[protected_attr_name].dtype != 'float64':
        df[protected_attr_name] = df[protected_attr_name].astype('category').cat.codes
    df['protected_attribute'] = df[protected_attr_name]
    return df


def get_fairness_metrics(dataset, privileged_groups, unprivileged_groups):
    metrics = {}

    binary_label_dataset = BinaryLabelDataset(df=dataset,
                                              label_names=['Label_value'],
                                              protected_attribute_names=['protected_attribute'])

    metric = BinaryLabelDatasetMetric(binary_label_dataset,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)

    classified_metric = ClassificationMetric(binary_label_dataset, binary_label_dataset,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metrics['mean_difference'] = metric.mean_difference()
    metrics['disparate_impact'] = metric.disparate_impact()
    metrics['false_discovery_rate_difference'] = classified_metric.false_discovery_rate_difference()
    metrics['false_positive_rate_difference'] = classified_metric.false_positive_rate_difference()
    metrics['false_omission_rate_difference'] = classified_metric.false_omission_rate_difference()
    metrics['false_negative_rate_difference'] = classified_metric.false_negative_rate_difference()
    metrics['average_odds_difference'] = classified_metric.average_odds_difference()
    metrics['equal_opportunity_difference'] = classified_metric.equal_opportunity_difference()

    return metrics


if __name__ == "__main__":
    # Load and preprocess the data
    filepath = "adult_predictions_with_original_data.csv"
    df = load_data(filepath)

    # Define the names of the protected attribute and label column
    protected_attr_name = 'Sex_male'  # Change this based on your dataset
    label_name = 'Label_value'  # Change this based on your dataset

    df = preprocess_data(df, protected_attr_name, label_name)

    # Define privileged and unprivileged groups
    privileged_groups = [{'protected_attribute': 1}]
    unprivileged_groups = [{'protected_attribute': 0}]

    # Get fairness metrics
    fairness_metrics = get_fairness_metrics(df, privileged_groups, unprivileged_groups)

    for metric, value in fairness_metrics.items():
        print(f"{metric}: {value}")
