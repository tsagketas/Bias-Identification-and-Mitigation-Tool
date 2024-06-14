import numpy as np
import pandas as pd

data_path = "adult.csv"
data = pd.read_csv(data_path)

# Define the label
label_name = "outcome"  # The column we want to predict

# Ensure binary labels in the outcome column
data[label_name] = data[label_name].apply(lambda x: 1 if x == '>50K' else 0)

# Save the modified dataset to adult.csv
data.to_csv("adult.csv", index=False)