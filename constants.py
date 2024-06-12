# constants.py

global protected_variables, fairness_metrics, mitigation_algorithms, protected_variables_values

fairness_metrics = [
    "mean_difference", "disparate_impact", "false_discovery_rate_difference",
    "false_positive_rate_difference", "false_omission_rate_difference",
    "false_negative_rate_difference"
]
fairness_metrics_descr = [
    "Mean Difference shows if one group is getting better results than another by comparing their average outcomes.",
    "Disparate Impact checks if one group gets favorable results much more often than another.",
    "False Discovery Rate Difference shows if one group is more likely to be wrongly predicted with a positive outcome.",
    "False Positive Rate Difference checks if one group is more often wrongly given positive outcomes.",
    "False Omission Rate Difference shows if one group is more likely to be wrongly predicted with a negative outcome.",
    "False Negative Rate Difference checks if one group is more often wrongly given negative outcomes."
]
fairness_metrics_ideal = [
    "The ideal value is 0.0, meaning no difference between groups.",
    "The ideal value is 1.0. Less than 1.0 means the privileged group benefits more; more than 1.0 means the unprivileged group benefits more.",
    "The ideal value is 0.0, meaning no difference between groups.",
    "The ideal value is 0.0, meaning no difference between groups.",
    "The ideal value is 0.0, meaning no difference between groups.",
    "The ideal value is 0.0, meaning no difference between groups."
]
fairness_example_metrics = [
    "mean_difference", "average_abs_odds_difference", "disparate_impact",
    "false_negative_rate_ratio"
]
fairness_example_metrics_descr = [
    "Mean Difference shows if one group is getting better results than another by comparing their average outcomes.",
    "Average Absolute Odds Difference is the average difference in false positive and true positive rates between groups.",
    "Disparate Impact checks if one group gets favorable results much more often than another.",
    "False Negative Rate Ratio shows if one group is more likely to have true positives missed."
]
fairness_example_metrics_ideal = [
    "The ideal value is 0.0, meaning no difference between groups.",
    "The ideal value is 0.0. Less than 0 means the privileged group benefits more; more than 0 means the unprivileged group benefits more.",
    "The ideal value is 1.0. Less than 1.0 means the privileged group benefits more; more than 1.0 means the unprivileged group benefits more.",
    "The ideal value is 1.0, meaning no difference in the likelihood of missing true positives between groups."
]
mitigation_example_algorithms = [
    "Reweighing", "Adversarial Debiasing", "Calibrated Equality of Odds"
]
mitigation_example_algorithms_descr = [
    "Reweighing adjusts the weights of examples to ensure fairness before making predictions.",
    "Adversarial Debiasing trains a model to be accurate and fair by reducing its ability to guess protected attributes.",
    "Calibrated Equality of Odds adjusts prediction labels to ensure fairness after making predictions."
]
mitigation_algorithms = [
    "Disparate Impact Remover", "Reweighing", "Adversarial Debiasing", "Meta Fair Classifier",
    "Calibrated Equality of Odds"
]
mitigation_algorithms_descr = [
    "Disparate Impact Remover changes feature values to increase fairness while keeping the order of examples within groups.",
    "Reweighing adjusts the weights of examples to ensure fairness before making predictions.",
    "Adversarial Debiasing trains a model to be accurate and fair by reducing its ability to guess protected attributes.",
    "Meta Fair Classifier trains a model to optimize a given fairness metric.",
    "Calibrated Equality of Odds adjusts prediction labels to ensure fairness after making predictions."
]

metrics_picked = []
algorithms_picked = []
values_picked = []
atts_picked = []
