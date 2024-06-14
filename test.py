import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import ClassificationMetric

# Load custom dataset
data_path = "adult.csv"
data = pd.read_csv(data_path)

# Define the label and protected attribute
label_name = "outcome"  # The column we want to predict
protected_attribute = "race"  # Assuming race is the protected attribute
sex_attribute = "sex"  # Assuming sex is an additional protected attribute

# Ensure binary labels in the outcome column
data[label_name] = data[label_name].apply(lambda x: 1 if x == '>50K' else 0)

# Split the data into features and labels
X = data.drop(columns=[label_name])
y = data[label_name]

# Prepare the preprocessing pipeline for numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Convert to DataFrame for BinaryLabelDataset and retain the encoded protected attributes
encoded_feature_names = numerical_features + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features))
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=encoded_feature_names)

# Print columns to debug
print("Encoded columns:", X_preprocessed_df.columns.tolist())

# Extract the encoded protected attribute columns
race_encoded = [col for col in X_preprocessed_df.columns if 'race_' in col]
sex_encoded = [col for col in X_preprocessed_df.columns if 'sex_' in col]

print(race_encoded,sex_encoded);
# Check if the encoded lists are correctly populated
if not race_encoded or not sex_encoded:
    raise ValueError("Encoding of protected attributes failed. Check the column names in the dataset and preprocessing.")

# Convert to AIF360 dataset
binary_label_dataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=pd.concat([X_preprocessed_df, y.reset_index(drop=True)], axis=1),
    label_names=[label_name],
    protected_attribute_names=race_encoded + sex_encoded
)

# Define privileged and unprivileged groups based on the encoded features
privileged_groups = [{race_encoded[0]: 1, sex_encoded[0]: 1}]  # Adjust to match your actual encoded column names
unprivileged_groups = [{race_encoded[0]: 0, sex_encoded[0]: 1}]  # Adjust to match your actual encoded column names

# Split the data into training and testing
train, test = binary_label_dataset.split([0.7], shuffle=True)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(train.features, train.labels.ravel())

# Make predictions
y_pred = model.predict(test.features)
y_pred_proba = model.predict_proba(test.features)[:, 1]

# Add predictions to the test dataset
test_pred = test.copy()
test_pred.labels = y_pred

# Compute metrics before bias mitigation
metric_orig_test = ClassificationMetric(test, test_pred,
                                        privileged_groups=privileged_groups,
                                        unprivileged_groups=unprivileged_groups)
print("Original mean difference:", metric_orig_test.mean_difference())

# Apply Disparate Impact Remover
dir = DisparateImpactRemover(repair_level=1.0)
train_repd = dir.fit_transform(train)
test_repd = dir.fit_transform(test)

# Train a new Logistic Regression model on the transformed data
model_repd = LogisticRegression()
model_repd.fit(train_repd.features, train_repd.labels.ravel())

# Make predictions on the transformed test data
y_pred_repd = model_repd.predict(test_repd.features)
test_repd_pred = test_repd.copy()
test_repd_pred.labels = y_pred_repd

# Compute metrics after bias mitigation
metric_repd_test = ClassificationMetric(test_repd, test_repd_pred,
                                        privileged_groups=privileged_groups,
                                        unprivileged_groups=unprivileged_groups)
print("Repaired mean difference:", metric_repd_test.mean_difference())

# Save the results
with open("results.txt", "w") as f:
    f.write(f"Original mean difference: {metric_orig_test.mean_difference()}\n")
    f.write(f"Repaired mean difference: {metric_repd_test.mean_difference()}\n")
