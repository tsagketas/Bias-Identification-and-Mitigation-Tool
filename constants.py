# constants.py

global protected_variables, fairness_metrics, mitigation_algorithms, protected_variables_values

fairness_metrics = [
    "mean_difference", "disparate_impact", "equal_opportunity_difference",
    "average_odds_difference","theil_index"
]
fairness_metrics_descr = [
    "Measures the ratio of favorable outcomes received by the unprivileged group to that of the privileged group. A value close to 1 indicates fairness.",
    "Measures the difference in the rate of favorable outcomes between the unprivileged and privileged groups. A value close to 0 indicates fairness.",
    "Measures the difference in true positive rates between the unprivileged and privileged groups. A value close to 0 indicates fairness.",
    "Measures the average difference in false positive rates and true positive rates between the unprivileged and privileged groups. A value close to 0 indicates fairness.",
    "Measures the inequality in the distribution of predictions. Lower values indicate less inequality and more fairness.",
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
    "mean_difference", "average_abs_odds_difference", "disparate_impact"
    # ,"false_negative_rate_ratio"
]
fairness_example_metrics_descr = [
    "Mean Difference shows if one group is getting better results than another by comparing their average outcomes.",
    "Average Absolute Odds Difference is the average difference in false positive and true positive rates between groups.",
    "Disparate Impact checks if one group gets favorable results much more often than another.",
    # "False Negative Rate Ratio shows if one group is more likely to have true positives missed."
]
fairness_example_metrics_ideal = [
    "The ideal value is 0.0, meaning no difference between groups.",
    "The ideal value is 0.0. Less than 0 means the privileged group benefits more; more than 0 means the unprivileged group benefits more.",
    "The ideal value is 1.0. Less than 1.0 means the privileged group benefits more; more than 1.0 means the unprivileged group benefits more.",
    # "The ideal value is 1.0, meaning no difference in the likelihood of missing true positives between groups."
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
    # "Disparate Impact Remover",
    "Reweighing",
    "Adversarial Debiasing",
    # "Prejudice Remover Impact Remover",
    "Calibrated equalized Odds"
]
mitigation_algorithms_descr = [
    # "Disparate Impact Remover: Modifies the training data to remove disparate impact.",
    "Reweighing: This technique assigns weights to data points to balance the representation of different groups. For example, if a dataset has more data from the privileged group, reweighing can increase the weight of data points from the unprivileged group during training.",
    "Adversarial Debiasing: This approach trains two models simultaneously. One model predicts the outcome, while the other tries to remove bias from the first model's predictions. They compete with each other, ultimately leading to a fairer model.",
    # "Prejudice Remover Impact Remover: Modifies the learning algorithm to reduce prejudice.",
    "Calibrated equalized Odds: This method adjusts the model's output probabilities to ensure equal odds for both privileged and unprivileged groups across different outcome categories."
]
