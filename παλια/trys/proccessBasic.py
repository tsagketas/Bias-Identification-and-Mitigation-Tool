from typing import List, Union, Dict
import os

# Modelling. Warnings will be used to silence various model warnings for tidier output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
import warnings

# Data handling/display
import pandas as pd
import numpy as np

from numpy.random import randint

# IBM's fairness tooolbox:
from aif360.datasets import BinaryLabelDataset,StandardDataset  # To handle the data
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric  # For calculating metrics
from aif360.explainers import MetricTextExplainer  # For explaining metrics

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# For the logistic regression model
from sklearn.preprocessing import StandardScaler



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

        global done
        if done:
            for att in protected_variables_cat:
                for x  in encoders[att]:
                    if x in unprivileged_values_cat:
                        unprivileged_values_coded.append(encoders[att][x])          
            done=False

      
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


def get_fairness_metrics(path_to_csv,protected_variables,protected_variables_values,fairness_metrics):

    global done,protected_variables_cat,protected_variables_num,unprivileged_values_cat,unprivileged_values_num,unprivileged_values_coded,privileged_groups,unprivileged_groups
    protected_variables_cat=[]
    protected_variables_num=[]
    unprivileged_values_cat=[]
    unprivileged_values_num=[]
    unprivileged_values_coded=[]
    privileged_groups = []#[{'Sex': 1}]
    unprivileged_groups = [] #[{'Sex': 0}]
    done=True

    df = pd.read_csv(path_to_csv)

    target_cols=df[['Score','Label_value']]
    to_use=df.drop(['Score', 'Label_value'],axis=1)
    numCols = to_use.select_dtypes(np.number).columns
    catCols = to_use.select_dtypes(np.object).columns

    for i in range(len(protected_variables)):
        if protected_variables[i] in numCols:
            protected_variables_num.append(protected_variables[i])
            unprivileged_values_num.append(float(protected_variables_values[i]))
        else:
            protected_variables_cat.append(protected_variables[i])
            unprivileged_values_cat.append(protected_variables_values[i])


    all_protected=protected_variables_cat+protected_variables_num
    all_protected_values=unprivileged_values_cat+unprivileged_values_num
    all_protected = list(filter(None, all_protected))#check kai gia ta ipoloipa
    all_protected_values = list(filter(None, all_protected_values))



    pp_object_cols = Pipeline([('select', SelectCols(cols=catCols)),
                            ('process', LabelEncoder())])

    # NumericEncoding fork: Select numeric columns -> numeric encode
    pp_numeric_cols = Pipeline([('select', SelectCols(cols=numCols)),
                                ('process', NumericEncoder())])

    # .fit_transform on train
    predictions =pd.concat([pp_numeric_cols.fit_transform(to_use), 
                            pp_object_cols.fit_transform(to_use),
                            target_cols['Score']],
                        axis=1)

    # .transform on valid
    ground_truth = pd.concat([pp_numeric_cols.fit_transform(to_use), 
                            pp_object_cols.fit_transform(to_use),
                            target_cols['Label_value']],
                            axis=1)

    ground_truth.to_csv('gia_des.csv',index=False )

    # edw sinartisi gia ta groups
    for i in range(len(all_protected)):
        unprivileged_groups.append( {all_protected[i] : 0 } )
        privileged_groups.append( {all_protected[i] : 1 } )
        if (unprivileged_values_coded[i] != 0 ):
            predictions.loc[predictions[all_protected[i]] == unprivileged_values_coded[i], all_protected[i]] = 0
            ground_truth.loc[ground_truth[all_protected[i]] == unprivileged_values_coded[i], all_protected[i]] = 0
            predictions.loc[predictions[all_protected[i]] != unprivileged_values_coded[i], all_protected[i]] = 1
            ground_truth.loc[ground_truth[all_protected[i]] != unprivileged_values_coded[i], all_protected[i]] = 1
        
    ground_truth=ground_truth.rename(columns = {'Label_value':'Score'})


    unprivileged_groups = list(filter(None, unprivileged_groups))#check kai gia ta ipoloipa
    privileged_groups = list(filter(None, privileged_groups))#check kai gia ta ipoloipa
    list_of_metrics=[]
    list_of_explainers=[]
    
    for i in range(len(all_protected)):
        print("edw")
        print(all_protected[i],unprivileged_values_coded[i]+1,unprivileged_values_coded[i],privileged_groups)
        predictions1=predictions.copy()
        ground_truth1=ground_truth.copy()
        if (unprivileged_values_coded[i] != 0 ):
            predictions1.loc[predictions[all_protected[i]] == unprivileged_values_coded[i], all_protected[i]] = 0
            ground_truth1.loc[ground_truth[all_protected[i]] == unprivileged_values_coded[i], all_protected[i]] = 0
            predictions1.loc[predictions[all_protected[i]] != unprivileged_values_coded[i], all_protected[i]] = 1
            ground_truth1.loc[ground_truth[all_protected[i]] != unprivileged_values_coded[i], all_protected[i]] = 1

        truth = BinaryLabelDataset(df=ground_truth1,
                                        label_names=['Score'],
                                        protected_attribute_names=all_protected,
                                        favorable_label=1,
                                        unfavorable_label=0,
                                        )
        #loop gia na ftiaksw osa zitaei me to protected_attribute_names.len
        preds = BinaryLabelDataset(df=predictions1,
                                        label_names=['Score'],
                                        protected_attribute_names=all_protected,
                                        favorable_label=1,
                                        unfavorable_label=0,
                                        ) 

        metric=ClassificationMetric(truth, preds, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        
        explainer = MetricTextExplainer_(metric)
        list_of_metrics.append(metric)
        list_of_explainers.append(explainer)

        os.system("pause")
    
    return list_of_metrics,list_of_explainers


fairness_metrics=['mean_difference','generalized_entropy_index']
the_metric,the_explainers = get_fairness_metrics( path_to_csv='check.csv',    protected_variables=['Gender', 'Attrition_Flag', 'Income_Category'],protected_variables_values=['M', 'Attrited Customer', 'Less than $40K'],fairness_metrics=fairness_metrics)
    
for i in range(len(the_explainers)):
    for el in fairness_metrics: 
        a=getattr(the_explainers[i], el)()
        print(el,a)

