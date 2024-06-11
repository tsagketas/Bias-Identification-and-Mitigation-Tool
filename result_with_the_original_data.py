import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'adult.csv'
data = pd.read_csv(file_path)

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Save a copy of the original dataset
original_data = data.copy()

# Dropping columns not useful for prediction (Optional: adjust according to your dataset)
# Assuming 'outcome' is the column to predict
predict_column = 'outcome'
if predict_column in data.columns:
    X = data.drop(columns=[predict_column])
    y = data[predict_column]
else:
    raise ValueError("The dataset does not contain the 'outcome' column.")

# Handling missing values
for column in X.columns:
    if X[column].dtype == 'object':
        # For categorical columns, fill missing values with the mode
        X[column].fillna(X[column].mode()[0], inplace=True)
    else:
        # For numerical columns, fill missing values with the median
        X[column].fillna(X[column].median(), inplace=True)

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Retrieve the original test dataset
original_test_data = original_data.iloc[X_test.index]

# Create a DataFrame with the original test data, predictions, and ground truth labels
results = original_test_data.copy()
results['Score'] = y_pred
results['Label_value'] = y_test.values

# Drop the original 'outcome' column
results.drop(columns=['outcome'], inplace=True)

# Save the results to a CSV file
results_file_path = 'adult_predictions_with_original_data.csv'
results.to_csv(results_file_path, index=False)
print(f'Results saved to {results_file_path}')
