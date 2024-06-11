import os
import pandas as pd
import csv
import argparse
from flask import Flask, render_template, url_for, redirect, request, flash,make_response,jsonify
import json
from werkzeug.utils import secure_filename
import proccess as prcs

app = Flask(__name__)
app.config['FILES_UPLOADS'] = r'C:\Users\ceid6\Desktop\diplomatiki\static\Uploaded_files'
global protected_variables,fairness_metrics,mitigation_algorithms,protected_variables_values
fairness_metrics=["mean_difference","disparate_impact","false_discovery_rate","false_positive_rate","false_omission_rate","false_negative_rate"]
fairness_metrics_descr=["Statistical imparity is the difference between the probability that a random individual drawn from unprivileged is labeled 1 and the probability that a random individual from privileged is labeled 1.","This is the ratio of probability of favorable outcomes between the unprivileged and privileged groups.","The FDR is the rate that features called significant are truly null.","False positive rate parity is achieved if the false positive rates (division of false positives with all negatives) False positive rate parity is achieved if the false positive rates (division of false positives with all negatives)","The false omission rate is the proportion of the individuals with a negative test result for which the true condition is positive","False negative rate parity is achieved if the false negative rates (division of false negatives with all positives) in the subgroups are close to each other. "]
fairness_example_metrics=["mean_difference","average_abs_odds_difference","disparate_impact","false_negative_rate_ratio"]
fairness_example_metrics_descr=["Statistical imparity is the difference between the probability that a random individual drawn from unprivileged is labeled 1 and the probability that a random individual from privileged is labeled 1.","Average of absolute difference in FPR and TPR for unprivileged and privileged groups","This is the ratio of probability of favorable outcomes between the unprivileged and privileged groups."," Also called the miss rate – is the probability that a true positive will be missed by the test."]
protected_variables=[]
protected_variables_values=[]
mitigation_algorithms=[]
metrics_picked=[]

class AttsnValues:

    def __init__(self, name):
        self.name = name
        self.values = []

    def addValue(self, value):
        if not value in self.values:
            self.values.append(value)
            self.values.sort
        else:
            pass


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/selection', methods=['GET', 'POST'])
def selection():
    return render_template("selection.html")


@app.route('/chooseAtt', methods=['GET', 'POST'])
def chooseAtt():
    return render_template("chooseAtt.html")

@app.route('/metric/<dataset>', methods=['GET', 'POST'])
def metric(dataset):

    if dataset=="Compass" or dataset=="Adult" or dataset=="German":
        pass
    else:
        pass

    # if request.method == 'POST':
    #     attributes = request.form.getlist('attribute')
    #     for element in attributes:
    #         protected_variables.append(element)
    #         protected_variables_values.append(request.form.get(element)) 

    return render_template("metric.html",metrics=fairness_example_metrics,description=fairness_example_metrics_descr)

@app.route('/graphsgo', methods=['GET','POST'])
def graphsgo():
    # fairness_metrics=['mean_difference','false_discovery_rate','average_odds_difference','disparate_impact']
    the_metrics,the_explainers = prcs.get_fairness_metrics( path_to_csv= app.config["UPLOADED_FILE"],    protected_variables=protected_variables,protected_variables_values=protected_variables_values)
    df2=prcs.prepare_df(0.2,the_metrics,metrics_picked,protected_variables)
    data=prcs.get_data(the_metrics,metrics_picked,protected_variables_values)
    return render_template("graphs.html",fairness_metrics=metrics_picked,protected_variables=json.dumps(protected_variables),data=data,data2=json.dumps(data))
    
@app.route('/graphs', methods=['GET','POST'])
def graphs():
    if request.method == "POST": 
      metrics = request.get_json()
      for val in metrics['name']:
          metrics_picked.append(val)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST' and request.files['file']:

        file_to_upload = request.files['file']

        file_to_upload.save(os.path.join(
            app.config["FILES_UPLOADS"], file_to_upload.filename))
        app.config["UPLOADED_FILE"] = app.config["FILES_UPLOADS"] + \
            "/" + file_to_upload.filename
        data = dataset_parse(app.config["UPLOADED_FILE"])
    #edw tha kanw call ti sinartisi process logika me oles tis times i apla tis pernaw to json 


    return render_template("metric.html", metrics=fairness_metrics,description=fairness_metrics_descr)


def dataset_parse(dataset):

    data_to_pass = []
    data_to_read = pd.read_csv(dataset, header=0)

    for col in data_to_read.columns:
        obj = AttsnValues(col)
        if not ( col=="Score" or col=="Label_value"):
            for value in data_to_read[col]:
                obj.addValue(value)
            data_to_pass.append(obj)

    return data_to_pass



if __name__ == "__main__":
    app.run(debug=True)
