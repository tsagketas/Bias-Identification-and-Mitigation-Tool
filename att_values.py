# att_values.py

import pandas as pd


class AttsnValues:
    def __init__(self, name):
        self.name = name
        self.values = set()

    def add_value(self, value):
        self.values.add(value)


def dataset_parse(dataset):
    data_to_read = pd.read_csv(dataset, header=0)
    data_to_pass = []

    for col in data_to_read.columns:
        obj = AttsnValues(col)
        for value in data_to_read[col]:
            obj.add_value(value)
        obj.values = sorted(obj.values)
        data_to_pass.append(obj)

    data_to_pass.sort(key=lambda x: x.name)

    return data_to_pass
