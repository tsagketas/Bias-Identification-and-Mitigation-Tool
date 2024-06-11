# We're going to use type hinting
from typing import List, Union, Dict

# Modelling. Warnings will be used to silence various model warnings for tidier output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
import warnings

# Data handling/display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import auc, roc_auc_score, roc_curve

# IBM's fairness tooolbox:
from aif360.datasets import BinaryLabelDataset  # To handle the data
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric  # For calculating metrics
from aif360.explainers import MetricTextExplainer  # For explaining metrics
from aif360.algorithms.preprocessing import Reweighing  # Preprocessing technique

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# For the logistic regression model
from sklearn.preprocessing import StandardScaler

PRIVILEGED_VALUES=[]
UNPRIVILEGED_VALUES=[]
PrivilegedVariables=[]
UnprivilegedVariables=[]
UnprivilegedVariables.append('male')

class SelectCols(TransformerMixin):
    """Select columns from a DataFrame."""
    def __init__(self, cols: List[str]) -> None:
        self.cols = cols

    def fit(self, x: None) -> "SelectCols":
        """Nothing to do."""
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return just selected columns."""
        return x[self.cols]

class LabelEncoder(TransformerMixin):
    """Convert non-numeric columns to numeric using label encoding. 
    Handles unseen data on transform."""
    def fit(self, x: pd.DataFrame) -> "LabelEncoder":
        """Learn encoder for each column."""
        encoders = {}
        for c in x:
            # Make encoder using pd.factorize on unique values, 
            # then convert to a dictionary
            v, k = zip(pd.factorize(x[c].unique()))
            encoders[c] = dict(zip(k[0], v[0]))

        self.encoders_ = encoders

        for x  in encoders['Sex']:
            if x in UnprivilegedVariables :
                # print(encoders['Sex'][x])
                UNPRIVILEGED_VALUES.append(encoders['Sex'][x])
            else:
                PRIVILEGED_VALUES.append(encoders['Sex'][x])
        return self

    def transform(self, x) -> pd.DataFrame:
        """For columns in x that have learned encoders, apply encoding."""
        x = x.copy()
        for c in x:
            # Ignore new, unseen values
            x.loc[~x[c].isin(self.encoders_[c]), c] = np.nan
            # Map learned labels
            x.loc[:, c] = x[c].map(self.encoders_[c])

        # Return without nans
        return x.fillna(-2).astype(int)

class NumericEncoder(TransformerMixin):
    """Remove invalid values from numerical columns, replace with median."""
    def fit(self, x: pd.DataFrame) -> "NumericEncoder":
        """Learn median for every column in x."""
        # Find median for all columns, handling non-NaNs invalid values and NaNs
        # Where all values are NaNs (after coercion) the median value will be a NaN.
        self.encoders_ = {
            c: pd.to_numeric(x[c],
                             errors='coerce').median(skipna=True) for c in x}

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """For each column in x, encode NaN values are learned 
        median and add a flag column indicating where these 
        replacements were made"""

        # Create a list of new DataFrames, each with 2 columns
        output_dfs = []
        for c in x:
            new_cols = pd.DataFrame()
            # Find invalid values that aren't nans (-inf, inf, string)
            invalid_idx = pd.to_numeric(x[c].replace([-np.inf, np.inf],
                                                     np.nan),
                                        errors='coerce').isnull()

            # Copy to new df for this column
            new_cols.loc[:, c] = x[c].copy()
            # Replace the invalid values with learned median
            new_cols.loc[invalid_idx, c] = self.encoders_[c]
            # Mark these replacement in a new column called 
            # "[column_name]_invalid_flag"
            new_cols.loc[:, f"{c}_invalid_flag"] = invalid_idx.astype(np.int8)

            output_dfs.append(new_cols)

        # Concat list of output_dfs to single df
        df = pd.concat(output_dfs,
                       axis=1)

        # Return wtih an remaining NaNs removed. These might exist if the median
        # is a NaN because there was no numeric data in the column at all.
        return df.fillna(0)

class MetricAdditions:
    def explain(self,
                disp: bool=True) -> Union[None, str]:
        """Explain everything available for the given metric."""

        # Find intersecting methods/attributes between MetricTextExplainer and provided metric.
        inter = set(dir(self)).intersection(set(dir(self.metric)))

        # Ignore private and dunder methods
        metric_methods = [getattr(self, c) for c in inter if c.startswith('_') < 1]

        # Call methods, join to new lines
        s = "\n".join([f() for f in metric_methods if callable(f)])

        if disp:
            print(s)
        else:
            return s
        
        
class MetricTextExplainer_(MetricTextExplainer, MetricAdditions):
    """Combine explainer and .explain."""
    pass

# sc = SelectCols(cols=['Sex', 'Survived'])
# sc=sc.transform(train)

# le = LabelEncoder()
# le.fit_transform(train[['Pclass', 'Sex']].sample(5))

# print(le.encoders_)

# ne = NumericEncoder()
# ne.fit_transform(train[['Age', 'Fare']].sample(5))

# print(ne.encoders_)


sns.set()
sns.set_context("talk")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test.loc[:, 'Survived'] = 0

numCols = train.select_dtypes(np.number).columns
catCols = train.select_dtypes(np.object).columns

# numCols= list(numCols)
# catCols= list(catCols)

pp_object_cols = Pipeline([('select', SelectCols(cols=catCols)),
                           ('process', LabelEncoder())])
y=[]
# NumericEncoding fork: Select numeric columns -> numeric encode
pp_numeric_cols = Pipeline([('select', SelectCols(cols=numCols)),
                            ('process', NumericEncoder())])


# We won't use the next part, but typically the pipeline would continue to 
# the model (after dropping 'Survived' from the training data, of course). 

# For example:
pp_pipeline = FeatureUnion([('object_cols', pp_object_cols),
                            ('numeric_cols', pp_numeric_cols)])

model_pipeline = Pipeline([('pp', pp_pipeline),
                           ('mod', LogisticRegression())])

train_, valid = train_test_split(train,
                                 test_size=0.3)

# .fit_transform on train
train_pp = pd.concat((pp_numeric_cols.fit_transform(train_), 
                      pp_object_cols.fit_transform(train_)),
                     axis=1)

# .transform on valid
valid_pp = pd.concat((pp_numeric_cols.transform(valid), 
                      pp_object_cols.transform(valid)),
                     axis=1)

print(valid_pp.sample(5))

test_pp = pd.concat((pp_numeric_cols.transform(test), 
                     pp_object_cols.transform(test)),
                    axis=1)

print(test_pp.sample(5))

target = 'Survived'
x_columns = [c for c in train_pp if c != target]
x_train, y_train = train_pp[x_columns], train_pp[target]
x_valid, y_valid = valid_pp[x_columns], valid_pp[target]
x_test = test_pp[x_columns]

sub = pd.read_csv('gender_submission.csv')
biased_lr = LogisticRegression()

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)

    biased_lr.fit(x_train, y_train)
    
print(f"Logistic regression validation accuracy: {biased_lr.score(x_valid, y_valid)}")

sub.loc[:, 'Survived'] = biased_lr.predict(x_test).astype(int)
sub.to_csv('biased_lr.csv', 
           index=False)



biased_rfc = RandomForestClassifier(n_estimators=100, 
                                    max_depth=4)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    
    biased_rfc.fit(x_train, y_train)
    
print(f"Random forest validation accuracy: {biased_rfc.score(x_valid, y_valid)}")

sub.loc[:, 'Survived'] = biased_rfc.predict(x_test).astype(int)
sub.to_csv('biased_rfc.csv', 
           index=False)


lets=pd.concat((x_test,sub['Survived']),axis=1)
lets.to_csv('input2.csv',index=False)
lets2=lets.copy()
neo=pd.DataFrame()
lets2.loc[1:310,'Survived'] = 0
neo['Score']=lets2['Survived']
neo=pd.concat((lets, neo['Score']), axis=1)
neo.to_csv('DOKIMES.csv',index=False)
train_pp_bld = BinaryLabelDataset(df=pd.concat((x_train, y_train),
                                               axis=1),
                                  label_names=['Survived'],
                                  protected_attribute_names=['Sex'],
                                  favorable_label=1,
                                  unfavorable_label=0)

preds = BinaryLabelDataset(df=lets,
                                  label_names=['Survived'],
                                  protected_attribute_names=['Sex'],
                                  favorable_label=1,
                                  unfavorable_label=0)

lets = BinaryLabelDataset(df=lets2,
                                  label_names=['Survived'],
                                  protected_attribute_names=['Sex'],
                                  favorable_label=1,
                                  unfavorable_label=0)

privileged_groups = [{'Sex': 1}]
unprivileged_groups = [{'Sex': 0}]

metric_train_bld = BinaryLabelDatasetMetric(train_pp_bld,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)


# Create the explainer object
explainer = MetricTextExplainer_(metric_train_bld)
# Explain relevant metrics
explainer.explain()

print("---------------------------------------------------------------------------------------------------------------------------------------------------")
oxaman=ClassificationMetric(lets, preds, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
explainer = MetricTextExplainer_(oxaman)
explainer.explain()
# oxaman.explain()
print("---------------------------------------------------------------------------------------------------------------------------------------------------")
print(oxaman.accuracy())