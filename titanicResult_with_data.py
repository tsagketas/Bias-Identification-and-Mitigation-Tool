
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
file_path = 'titanic.csv'
titanic_data = pd.read_csv(file_path)

# Handling missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.drop(columns=['Cabin'], inplace=True)

# Encoding categorical variables
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

# Dropping unnecessary columns
titanic_data.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Splitting data into features and target
X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']

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

# Create a DataFrame with the predictions, ground truth labels, and the test set data
X_test['Predicted'] = y_pred
X_test['Ground Truth'] = y_test.values

# Save the results to a CSV file
results_file_path = 'titanic_predictions_with_data.csv'
X_test.to_csv(results_file_path, index=False)
print(f'Results saved to {results_file_path}')
