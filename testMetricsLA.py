import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'adult_predictions_with_original_data.csv'
df = pd.read_csv(file_path)

# Convert the labels and scores to binary values
df['Label_value'] = df['Label_value'].apply(lambda x: 1 if x == '>50K' else 0)
df['Score'] = df['Score'].apply(lambda x: 1 if x == '>50K' else 0)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Check the unique values in 'race', 'gender', 'Label_value', and 'Score'
print("Unique values in 'race':", df['race'].unique())
print("Unique values in 'gender':", df['gender'].unique())
print("Unique values in 'Label_value':", df['Label_value'].unique())
print("Unique values in 'Score':", df['Score'].unique())

# Create BinaryLabelDataset for true labels
binary_label_dataset = BinaryLabelDataset(df=df,
                                          label_names=['Label_value'],
                                          protected_attribute_names=['race', 'gender'])

# Create a copy of the dataset for predictions
df_pred = df.copy()
df_pred['Label_value'] = df_pred['Score']

# Create BinaryLabelDataset for predicted scores
predicted_dataset = BinaryLabelDataset(df=df_pred,
                                       label_names=['Label_value'],
                                       protected_attribute_names=['race', 'gender'])

# Define privileged and unprivileged groups separately for race and gender
privileged_groups = [{'race': 4, 'gender': 1}]  # White male
unprivileged_groups_women = [{'race': 2, 'gender': 0}]  # Black female
unprivileged_groups_men = [{'race': 2, 'gender': 1}]  # Black male

# Initialize metrics dictionary
metrics = {}

# Calculate overall fairness metrics (not intersectional)
metric = BinaryLabelDatasetMetric(binary_label_dataset,
                                  unprivileged_groups=[{'race': 2}],  # Black
                                  privileged_groups=[{'race': 4}])  # White

classified_metric = ClassificationMetric(binary_label_dataset, predicted_dataset,
                                         unprivileged_groups=[{'race': 2}],
                                         privileged_groups=[{'race': 4}])

metrics['mean_difference'] = metric.mean_difference()
metrics['disparate_impact'] = metric.disparate_impact()
metrics['false_discovery_rate_difference'] = classified_metric.false_discovery_rate_difference()
metrics['false_positive_rate_difference'] = classified_metric.false_positive_rate_difference()
metrics['false_omission_rate_difference'] = classified_metric.false_omission_rate_difference()
metrics['false_negative_rate_difference'] = classified_metric.false_negative_rate_difference()
metrics['average_odds_difference'] = classified_metric.average_odds_difference()
metrics['equal_opportunity_difference'] = classified_metric.equal_opportunity_difference()

print("Overall fairness metrics (not intersectional):")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Calculate Fairness Metrics for intersectional groups (White men vs. Black women)
filtered_df_women = df[(df['race'].isin([4, 2])) & (df['gender'] == 0)]
print("Filtered DataFrame for women - shape:", filtered_df_women.shape)  # Check the size of the filtered DataFrame

# Check counts of positive and negative outcomes for each group
print("Counts of outcomes for White women:", filtered_df_women[filtered_df_women['race'] == 4]['Score'].value_counts())
print("Counts of outcomes for Black women:", filtered_df_women[filtered_df_women['race'] == 2]['Score'].value_counts())

binary_label_dataset_white_black_women = BinaryLabelDataset(
    df=filtered_df_women,
    label_names=['Label_value'],
    protected_attribute_names=['race', 'gender'],
    scores_names=['Score']
)

metric_white_black_women = BinaryLabelDatasetMetric(
    binary_label_dataset_white_black_women,
    unprivileged_groups=unprivileged_groups_women,
    privileged_groups=privileged_groups
)

disparate_impact_white_black_women = metric_white_black_women.disparate_impact()
print(f"Disparate Impact (White women vs. Black women): {disparate_impact_white_black_women}")

# Repeat similar steps for other intersectional comparisons, such as White men vs. Black men
filtered_df_men = df[(df['race'].isin([4, 2])) & (df['gender'] == 1)]
print("Filtered DataFrame for men - shape:", filtered_df_men.shape)  # Check the size of the filtered DataFrame

# Check counts of positive and negative outcomes for each group
print("Counts of outcomes for White men:", filtered_df_men[filtered_df_men['race'] == 4]['Score'].value_counts())
print("Counts of outcomes for Black men:", filtered_df_men[filtered_df_men['race'] == 2]['Score'].value_counts())

binary_label_dataset_white_black_men = BinaryLabelDataset(
    df=filtered_df_men,
    label_names=['Label_value'],
    protected_attribute_names=['race', 'gender'],
    scores_names=['Score']
)

metric_white_black_men = BinaryLabelDatasetMetric(
    binary_label_dataset_white_black_men,
    unprivileged_groups=unprivileged_groups_men,
    privileged_groups=privileged_groups
)

disparate_impact_white_black_men = metric_white_black_men.disparate_impact()
print(f"Disparate Impact (White men vs. Black men): {disparate_impact_white_black_men}")
