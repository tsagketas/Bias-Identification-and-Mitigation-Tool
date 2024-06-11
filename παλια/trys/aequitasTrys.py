import pandas as pd
import csv
import argparse
import matplotlib.pyplot as plt
import aequitas.fairness
import seaborn as sns
from aequitas.bias import Bias
from aequitas.plotting import Plot
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df
from aequitas.fairness import Fairness

FAIR_MAP = {'Equal Parity': {'Statistical Parity'},
            'Proportional Parity': {'Impact Parity'},
            'False Positive Rate Parity': {'FPR Parity'},
            'False Negative Rate Parity': {'FNR Parity'},
            'False Discovery Rate Parity': {'FDR Parity'},
            'False Omission Rate Parity': {'FOR Parity'}}

FAIR_MAP_ORDER = ['Equal Parity', 'Proportional Parity', 'False Positive Rate Parity', 'False Discovery Rate Parity',
                  'False Negative Rate Parity', 'False Omission Rate Parity']

dataset=r'static\Uploaded_files\data4.csv'
data = pd.read_csv(dataset)

print(data.shape)

df, _ = preprocess_input_df(data)

g = Group()
xtab, _ = g.get_crosstabs(df)
absolute_metrics = g.list_absolute_metrics(xtab)
print(xtab)

xtab2=xtab[[col for col in xtab.columns if col not in absolute_metrics]]
print(xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2))
aqp = Plot()
fnr = aqp.plot_group_metric(xtab, 'pprev')
plt.show()

b=Bias()
bdf = b.get_disparity_predefined_groups(xtab, original_df=df, ref_groups_dict={'race':'Caucasian', 'sex':'Male', 'age_cat':'25 - 45'}, alpha=0.05, mask_significance=True)
calculated_disparities = b.list_disparities(bdf)
disparity_significance = b.list_significance(bdf)

print(bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance])

hbdf = b.get_disparity_predefined_groups(xtab, original_df=df,
                                         ref_groups_dict={'race':'Caucasian', 'sex':'Male', 'age_cat':'25 - 45'},
                                         alpha=0.05,
                                         mask_significance=False)                                       

print(hbdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance])