import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing

# Load the dataset
dataset_path = 'adult_predictions_with_original_data.csv'  # Change the path as needed
df = pd.read_csv(dataset_path)

# Encode the categorical variables
def encode_categorical_variables(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

# Decode the categorical variables
def decode_categorical_variables(df, label_encoders):
    for column, le in label_encoders.items():
        df[column] = le.inverse_transform(df[column].astype(int))
    return df

df_encoded, label_encoders = encode_categorical_variables(df)

# Convert to BinaryLabelDataset
binary_label_dataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df_encoded,
    label_names=['Label_value'],
    protected_attribute_names=['race', 'gender']
)

# Apply DisparateImpactRemover
dir = DisparateImpactRemover(repair_level=1.0)
dir_transformed = dir.fit_transform(binary_label_dataset)

# Convert back to DataFrame
dir_df = dir_transformed.convert_to_dataframe()[0]

# Decode the categorical variables in the DisparateImpactRemover dataset
dir_df = decode_categorical_variables(dir_df, label_encoders)

# Apply Reweighing
rw = Reweighing(unprivileged_groups=[{'race': 0}], privileged_groups=[{'race': 1}])
rw_transformed = rw.fit_transform(binary_label_dataset)

# Convert back to DataFrame
rw_df = rw_transformed.convert_to_dataframe()[0]

# Decode the categorical variables in the Reweighing dataset
rw_df = decode_categorical_variables(rw_df, label_encoders)

# Save the mitigated datasets
dir_df.to_csv('dir_mitigated_dataset.csv', index=False)
rw_df.to_csv('rw_mitigated_dataset.csv', index=False)

print("Disparate Impact Remover mitigated dataset saved to 'dir_mitigated_dataset.csv'")
print("Reweighing mitigated dataset saved to 'rw_mitigated_dataset.csv'")

# Display the first few rows of the mitigated datasets
print("Disparate Impact Remover Mitigated Dataset")
print(dir_df.head())
print("Reweighing Mitigated Dataset")
print(rw_df.head())
