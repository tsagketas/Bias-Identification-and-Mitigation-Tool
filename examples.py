from matplotlib import pyplot as plt
import json 
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from aif360.datasets import AdultDataset,GermanDataset,CompasDataset,BankDataset,BinaryLabelDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.explainers import MetricJSONExplainer
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def example(dataset,atts_picked):
    
    train_datasets=[]
    test_datasets=[]
    the_b_datasets=[]
    datasets=[]
    privileged_grps=[]
    unprivileged_grps=[]
    b_metrics=[]
    truth=[]
    cl_metrics=[]

    if dataset == "Adult" :
        for element in atts_picked:
            if element['name'] == "sex":
                data=(load_preproc_data_adult(['sex']))
                privileged_groups=[{"sex": 1}]
                unprivileged_groups=[{"sex": 0}]
            else:
                data=(load_preproc_data_adult(['race']))
                privileged_groups=[{"race": 1}]
                unprivileged_groups=[{"race": 0}]
            
            datasets.append(data)
            privileged_grps.append(privileged_groups)
            unprivileged_grps.append(unprivileged_groups)

            dataset_orig_train, dataset_orig_vt = data.split([0.7], shuffle=True)
            train_datasets.append(dataset_orig_train)
            test_datasets.append(dataset_orig_vt)
            scale_orig = StandardScaler()

            X_train = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()

            X_test = scale_orig.fit_transform(dataset_orig_vt.features)
            y_test = dataset_orig_vt.labels.ravel()

            lmod = LogisticRegression()
            lmod.fit(X_train, y_train)
            y_train_pred = lmod.predict(X_test)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_train_pred).ravel()
            details=[{'Accuracy': metrics.accuracy_score(y_test, y_train_pred), 'Precision':metrics.precision_score(y_test, y_train_pred),'Recall': metrics.recall_score(y_test, y_train_pred),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp}]
        
            dataset_orig_test_pred = dataset_orig_vt.copy()
            dataset_orig_test_pred.labels = y_train_pred

            the_b_datasets.append(dataset_orig_test_pred)

            b=BinaryLabelDatasetMetric(dataset_orig_test_pred,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            t=BinaryLabelDatasetMetric(dataset_orig_vt,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            m=ClassificationMetric(dataset_orig_test_pred,dataset_orig_vt,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            b_metrics.append(b)
            truth.append(t)
            cl_metrics.append(m)
        
    elif dataset == "German":
        for element in atts_picked:
            if element['name'] == "sex":
                data=(load_preproc_data_german(['sex']))
                privileged_groups=[{"sex": 1}]
                unprivileged_groups=[{"sex": 0}]
            else:
                data=load_preproc_data_german(['age'])
                privileged_groups=[{"age": 1}]
                unprivileged_groups=[{"age": 0}]

            datasets.append(data)
            privileged_grps.append(privileged_groups)
            unprivileged_grps.append(unprivileged_groups)

            dataset_orig_train, dataset_orig_vt = data.split([0.7], shuffle=True)
            train_datasets.append(dataset_orig_train)
            test_datasets.append(dataset_orig_vt)
            scale_orig = StandardScaler()

            X_train = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()

            X_test = scale_orig.fit_transform(dataset_orig_vt.features)
            y_test = dataset_orig_vt.labels.ravel()

            lmod = LogisticRegression()
            lmod.fit(X_train, y_train)
            y_train_pred = lmod.predict(X_test)

            dataset_orig_test_pred = dataset_orig_vt.copy()
            dataset_orig_test_pred.labels = y_train_pred
            the_b_datasets.append(dataset_orig_test_pred)

            tn, fp, fn, tp = confusion_matrix(y_test, y_train_pred).ravel()
            details=[{'Accuracy': metrics.accuracy_score(y_test, y_train_pred), 'Precision':metrics.precision_score(y_test, y_train_pred),'Recall': metrics.recall_score(y_test, y_train_pred),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp}]

            b=BinaryLabelDatasetMetric(dataset_orig_test_pred,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            t=BinaryLabelDatasetMetric(dataset_orig_vt,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            m=ClassificationMetric(dataset_orig_test_pred,dataset_orig_vt,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            b_metrics.append(b)
            truth.append(t)
            cl_metrics.append(m)    
    else :
        for element in atts_picked:
            if element['name'] == "sex":
                data=load_preproc_data_compas(['sex'])
                privileged_groups=[{"sex": 0}]
                unprivileged_groups=[{"sex": 1}]
            else:
               data=load_preproc_data_compas(['race'])
               privileged_groups=[{"race": 1}]
               unprivileged_groups=[{"race": 0}]

            datasets.append(data)
            privileged_grps.append(privileged_groups)
            unprivileged_grps.append(unprivileged_groups)

            dataset_orig_train, dataset_orig_vt = data.split([0.7], shuffle=True)
            train_datasets.append(dataset_orig_train)
            test_datasets.append(dataset_orig_vt)
            scale_orig = StandardScaler()

            X_train = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()

            X_test = scale_orig.fit_transform(dataset_orig_vt.features)
            y_test = dataset_orig_vt.labels.ravel()

            lmod = LogisticRegression()
            lmod.fit(X_train, y_train)
            y_train_pred = lmod.predict(X_test)

            dataset_orig_test_pred = dataset_orig_vt.copy()
            dataset_orig_test_pred.labels = y_train_pred
            
            the_b_datasets.append(dataset_orig_test_pred)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_train_pred).ravel()
            details=[{'Accuracy': metrics.accuracy_score(y_test, y_train_pred), 'Precision':metrics.precision_score(y_test, y_train_pred),'Recall': metrics.recall_score(y_test, y_train_pred),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp}]

            b=BinaryLabelDatasetMetric(dataset_orig_test_pred,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            t=BinaryLabelDatasetMetric(dataset_orig_vt,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            m=ClassificationMetric(dataset_orig_test_pred,dataset_orig_vt,privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
            b_metrics.append(b)
            truth.append(t)
            cl_metrics.append(m)
   

    return b_metrics,truth,cl_metrics,datasets,privileged_grps,unprivileged_grps,the_b_datasets,train_datasets,test_datasets,details

def mitigation_examples(to_json,methods,data,privileged_groups,unprivileged_groups,the_biased_b_datasets,train_datasets,test_datasets): 
    mitigation_json=[]
    model_info=[]
    temp=[]
    counter=0
    flag1=True
    flag2=True
    flag3=True
    for method in methods:
        temp={'Method': method, 'Data':[]}
        values=[]
        for element in to_json:
            values=[]
            mitigation_json_data={'Metric': element['Metric'], 'ID':counter, 'Protected_Attributes':element['Protected_Attributes'],'Values':values,'Biased_values':element['Values'], 'Atts_to_show':element['Atts_to_show']}
            for i in range(0, len(data)):
                if (method=="Reweighing"):
                    
                    RW = Reweighing(unprivileged_groups=unprivileged_groups[i],privileged_groups=privileged_groups[i])
                    RW.fit(train_datasets[i])
                    dataset_transf_train = RW.transform(train_datasets[i])

                    scale_transf = StandardScaler()
                    scale_orig = StandardScaler()

                    X_train = scale_transf.fit_transform(dataset_transf_train.features)
                    y_train = dataset_transf_train.labels.ravel()

                    X_test = scale_orig.fit_transform(test_datasets[i].features)
                    y_test = test_datasets[i].labels.ravel()

                    lmod = LogisticRegression()
                    lmod.fit(X_train, y_train,
                            sample_weight=dataset_transf_train.instance_weights)

                    y_train_pred = lmod.predict(X_test)

                    dataset_orig_test_pred = test_datasets[i].copy()
                    dataset_orig_test_pred.labels = y_train_pred

                    if (element['Metric']=='mean_difference' or element['Metric']=='disparate_impact'):
                        blm=BinaryLabelDatasetMetric(dataset_orig_test_pred,unprivileged_groups=unprivileged_groups[i],privileged_groups=privileged_groups[i])
                        value=getattr(blm, element['Metric'])()
                    else:
                        preds=ClassificationMetric(dataset_orig_test_pred,the_biased_b_datasets[i],  
                                            unprivileged_groups=privileged_groups[i], 
                                            privileged_groups=unprivileged_groups[i],
                                            )
                        value=getattr(preds, element['Metric'])()
                    
                    if flag1:
                        tn, fp, fn, tp = confusion_matrix(y_test, y_train_pred).ravel()
                        temp2={'Accuracy': metrics.accuracy_score(y_test, y_train_pred), 'Precision':metrics.precision_score(y_test, y_train_pred),'Recall': metrics.recall_score(y_test, y_train_pred),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp,'Method':method}
                        model_info.append(temp2)
                        flag1=False

                    values.append(value)

                elif (method=="Adversarial Debiasing"):
                    
                    tf.reset_default_graph()
                    sess = tf.Session()
                    plain_model = AdversarialDebiasing(privileged_groups = privileged_groups[i],
                                            unprivileged_groups = unprivileged_groups[i],
                                            scope_name='plain_classifier',
                                            debias=False,
                                            sess=sess)

                    plain_model.fit(train_datasets[i])
                    debiased_preds = plain_model.predict(test_datasets[i])

                    if (element['Metric']=='mean_difference' or element['Metric']=='disparate_impact'):
                        blm=BinaryLabelDatasetMetric(debiased_preds,unprivileged_groups=unprivileged_groups[i],privileged_groups=privileged_groups[i])
                        value=getattr(blm, element['Metric'])()
                    else:
                        preds=ClassificationMetric(debiased_preds,the_biased_b_datasets[i],  
                                            unprivileged_groups=privileged_groups[i], 
                                            privileged_groups=unprivileged_groups[i],
                                            )
                        value=getattr(preds, element['Metric'])()
                    
                    if flag2:
                        tn, fp, fn, tp = confusion_matrix(debiased_preds.labels, test_datasets[i].labels).ravel()
                        temp2={'Accuracy': metrics.accuracy_score(debiased_preds.labels, test_datasets[i].labels), 'Precision':metrics.precision_score(debiased_preds.labels, test_datasets[i].labels),'Recall': metrics.recall_score(debiased_preds.labels, test_datasets[i].labels),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp,'Method':method}
                        model_info.append(temp2)
                        flag2=False

                    sess.close()
                    values.append(value)
                else:
                    randseed = 12345679
                    cost_constraint = "weighted"
                    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups[i],
                                     unprivileged_groups = unprivileged_groups[i],
                                     cost_constraint=cost_constraint,
                                     seed=randseed)

                    cpp = cpp.fit(test_datasets[i], the_biased_b_datasets[i])
                    debiased_preds = cpp.predict(the_biased_b_datasets[i])

                    if (element['Metric']=='mean_difference' or element['Metric']=='disparate_impact'):
                        preds=BinaryLabelDatasetMetric(debiased_preds,unprivileged_groups=unprivileged_groups[i],privileged_groups=privileged_groups[i])
                    else:
                        preds=ClassificationMetric(debiased_preds, 
                                                the_biased_b_datasets[i], #i to allo me ta preds  
                                                unprivileged_groups=unprivileged_groups[i], 
                                                privileged_groups=privileged_groups[i],
                                                )

                    value=getattr(preds, element['Metric'])()
                    values.append(value)
                    if flag3:
                        tn, fp, fn, tp = confusion_matrix(debiased_preds.labels, the_biased_b_datasets[i].labels).ravel()
                        temp2={'Accuracy': metrics.accuracy_score(debiased_preds.labels, the_biased_b_datasets[i].labels), 'Precision':metrics.precision_score(debiased_preds.labels, the_biased_b_datasets[i].labels),'Recall': metrics.recall_score(debiased_preds.labels, the_biased_b_datasets[i].labels),'True negatives': tn,'False positives': fp,'False negatives': fn,'True positives': tp,'Method':method}
                        model_info.append(temp2)
                        flag3=False   

            mitigation_json_data['Values']=values
            temp['Data'].append(mitigation_json_data)
            counter=counter+1

                
        mitigation_json.append(temp)    
    
    return mitigation_json,model_info


def save_conf(y_test, y_pred,method):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="PuBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(bottom=.05,left=.15)
    if method=="Reweighing":
        fig.savefig('static/Reweighing_Example.png')
    elif method=="Adversarial Debiasing":
        fig.savefig('static/AdversarialDebiasing_Example.png')
    else: 
        fig.savefig('static/CalibratedEqualityofOdds_Example.png')            
