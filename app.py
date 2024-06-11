import os
import pandas as pd
import csv
import argparse
from flask import Flask, render_template, url_for, redirect, request, flash, make_response, jsonify
import json
from werkzeug.utils import secure_filename
import proccess as prcs
import examples as expls

app = Flask(__name__, template_folder='templates')
app.config['FILES_UPLOADS'] = r'C:\Users\ceid6\Desktop\diplomatiki\static\Uploaded_files'
TEMPLATES_AUTO_RELOAD = True

global protected_variables, fairness_metrics, mitigation_algorithms, protected_variables_values
fairness_metrics = ["mean_difference", "disparate_impact", "false_discovery_rate_difference",
                    "false_positive_rate_difference", "false_omission_rate_difference",
                    "false_negative_rate_difference"]
fairness_metrics_descr = [
    "Statistical parity difference or Mean Difference is the difference between the probability that a random individual drawn from unprivileged is labeled 1 and the probability that a random individual from privileged is labeled 1.",
    "This is the ratio of probability of favorable outcomes between the unprivileged and privileged groups.",
    "The FDR is the rate that features called significant are truly null.",
    "False positive rate parity is achieved if the false positive rates (division of false positives with all negatives) False positive rate parity is achieved if the false positive rates (division of false positives with all negatives)",
    "The false omission rate is the proportion of the individuals with a negative test result for which the true condition is positive",
    "False negative rate parity is achieved if the false negative rates (division of false negatives with all positives) in the subgroups are close to each other. "]
fairness_metrics_ideal = ["The ideal value of this metric is 0.0",
                          "The ideal value of this metric is 1.0 A value < 1 implies higher benefit for the privileged group and a value >1 implies a higher benefit for the unprivileged group.",
                          "The ideal value of this metric is 0.0", "The ideal value of this metric is 0.0",
                          "The ideal value of this metric is 0.0", "The ideal value of this metric is 0.0"]
fairness_example_metrics = ["mean_difference", "average_abs_odds_difference", "disparate_impact",
                            "false_negative_rate_ratio"]
fairness_example_metrics_descr = [
    "Statistical parity difference or Mean Difference is the difference between the probability that a random individual drawn from unprivileged is labeled 1 and the probability that a random individual from privileged is labeled 1.",
    "Average of absolute difference in FPR and TPR for unprivileged and privileged groups",
    "This is the ratio of probability of favorable outcomes between the unprivileged and privileged groups.",
    " Also called the miss rate – is the probability that a true positive will be missed by the test."]
fairness_example_metrics_ideal = ["The ideal value of this metric is 0.0",
                                  "The ideal value of this metric is 0.0.  A value of < 0 implies higher benefit for the privileged group and a value > 0 implies higher benefit for the unprivileged group.",
                                  "The ideal value of this metric is 1.0 A value < 1 implies higher benefit for the privileged group and a value >1 implies a higher benefit for the unprivileged group.",
                                  "The ideal value of this metric is 1.0"]
mitigation_example_algorithms = ["Reweighing", "Adversarial Debiasing", "Calibrated Equality of Odds"]
mitigation_example_algorithms_descr = [
    "Reweighing is a preprocessing technique that Weights the examples in each (group, label) combination differently to ensure fairness before classification",
    "Adversarial debiasing is an in-processing technique that learns a classifier to maximize prediction accuracy and simultaneously reduce an adversary’s ability to determine the protected attribute from the predictions. This approach leads to a fair classifier as the predictions cannot carry any group discrimination information that the adversary can exploit.",
    "Calibrated equalized odds postprocessing is a post-processing technique that optimizes over calibrated classifier score outputs to find probabilities with which to change output labels with an equalized odds objective"]
mitigation_algorithms = ["Disparate Impact Remover", "Reweighing", "Adversarial Debiasing", "Meta Fair Classifier",
                         "Calibrated Equality of Odds"]
mitigation_algorithms_descr = [
    "Disparate impact remover is a preprocessing technique that edits feature values increase group fairness while preserving rank-ordering within groups",
    "Reweighing is a preprocessing technique that Weights the examples in each (group, label) combination differently to ensure fairness before classification",
    "Adversarial debiasing is an in-processing technique that learns a classifier to maximize prediction accuracy and simultaneously reduce an adversary’s ability to determine the protected attribute from the predictions. This approach leads to a fair classifier as the predictions cannot carry any group discrimination information that the adversary can exploit.",
    "The meta algorithm here takes the fairness metric as part of the input and returns a classifier optimized w.r.t. that fairness metric",
    "Calibrated equalized odds postprocessing is a post-processing technique that optimizes over calibrated classifier score outputs to find probabilities with which to change output labels with an equalized odds objective"]
metrics_picked = []
algorithms_picked = []
values_picked = []
atts_picked = []


class AttsnValues:

    def __init__(self, name):
        self.name = name
        self.values = []

    def addValue(self, value):
        if not value in self.values:
            self.values.append(value)
        else:
            pass


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/algorithms', methods=['GET', 'POST'])
def algorithms():
    if app.config["DATASET"] == "upload":
        return render_template("algorithms.html", algorithms=mitigation_algorithms,
                               description=mitigation_algorithms_descr)
    else:
        return render_template("algorithms.html", algorithms=mitigation_example_algorithms,
                               description=mitigation_example_algorithms_descr)


@app.route('/selection', methods=['GET', 'POST'])
def selection():
    return render_template("selection.html")


@app.route('/fairness_report', methods=['GET', 'POST'])
def fairness_report():
    if app.config["DATASET"] == "upload":
        the_metrics, the_b_metrics, datasets, label_data, score_data, app.config[
            "b_details"] = prcs.get_fairness_metrics(app.config["atts_n_vals"], path_to_csv=app.config["UPLOADED_FILE"])
        app.config["Label_data"] = label_data
        app.config["Score_data"] = score_data
        data, flag = prcs.get_data(the_metrics, the_b_metrics, metrics_picked, atts_picked, values_picked,
                                   app.config['Threshold'], app.config["DATASET"])
    else:
        the_b_metrics, truth, the_metrics, datasets, app.config["privileged_groups"], app.config["unprivileged_groups"], \
        app.config["the_b_datasets"], app.config["train_datasets"], app.config["test_datasets"], app.config[
            "b_details"] = expls.example(app.config["DATASET"], app.config["atts"])
        app.config["Datasets"] = datasets
        data, flag = prcs.get_data(the_metrics, the_b_metrics, metrics_picked, atts_picked, values_picked, 0,
                                   app.config["DATASET"])

    app.config["DATA"] = data
    if app.config["DATASET"] == "upload":
        ideal = fairness_metrics_ideal
    else:
        ideal = fairness_example_metrics_ideal

    return render_template("fairness_report.html", fairness_metrics=metrics_picked, data=data, data2=json.dumps(data),
                           ideal=ideal, threshold=app.config['Threshold'], flag=flag, details=app.config["b_details"])


@app.route('/mitigation_report', methods=['GET', 'POST'])
def mitigation_report():
    if app.config["DATASET"] == "upload":
        unbiased_data, app.config["ub_details"] = prcs.mitigation_all(app.config["DATA"], app.config["Label_data"],
                                                                      app.config["Score_data"], algorithms_picked)
    else:
        unbiased_data, app.config["ub_details"] = expls.mitigation_examples(app.config["DATA"], algorithms_picked,
                                                                            app.config["Datasets"],
                                                                            app.config["privileged_groups"],
                                                                            app.config["unprivileged_groups"],
                                                                            app.config["the_b_datasets"],
                                                                            app.config["train_datasets"],
                                                                            app.config["test_datasets"])

    return render_template("mitigation_report.html", data=unbiased_data, threshold=app.config['Threshold'],
                           ideal=fairness_metrics_ideal, biased_details=app.config["b_details"],
                           unbiased_details=app.config["ub_details"])


@app.route('/graphs', methods=['GET', 'POST'])
def graphs():
    if request.method == "POST":
        metrics = request.get_json()
        app.config['Threshold'] = metrics['threshold']
        for val in metrics['name']:
            metrics_picked.append(val)

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/mit_graphs', methods=['GET', 'POST'])
def mit_graphs():
    if request.method == "POST":
        algs = request.get_json()
        for val in algs['name']:
            algorithms_picked.append(val)

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/Choose_att', defaults={'dataset': None}, methods=['GET', 'POST'])
@app.route('/Choose_att/<dataset>', methods=['GET', 'POST'])
def Choose_att(dataset):
    if not dataset:
        if request.method == 'POST' and request.files['file']:
            file_to_upload = request.files['file']

            file_to_upload.save(os.path.join(
                app.config["FILES_UPLOADS"], file_to_upload.filename))
            app.config["UPLOADED_FILE"] = app.config["FILES_UPLOADS"] + \
                                          "/" + file_to_upload.filename
            app.config["DATASET"] = "upload"
            data = dataset_parse(app.config["UPLOADED_FILE"])

        return render_template("chooseAtt.html", data=data)
    else:
        data = []
        app.config["DATASET"] = dataset
        if dataset == "Adult":
            obj = AttsnValues("sex")
            obj.addValue(" Male is considered privileged (value = 1) and Female is considered unprivileged (value = 0)")
            obj1 = AttsnValues("race")
            obj1.addValue(
                "White is considered privileged (value = 1) and Non-white is considered unprivileged (value = 0)")
        elif dataset == "German":
            obj = AttsnValues("sex")
            obj.addValue("Male is  considered privileged (value = 1) and Female is considered unprivileged (value = 0)")
            obj1 = AttsnValues("age")
            obj1.addValue(
                " age >= 25 is considered privileged (value = 1) and age < 25 is considered unprivileged (value = 0)")
        else:
            obj = AttsnValues("sex")
            obj.addValue("Female is considered privileged (value = 1) and Male is considered unprivileged (value = 0)")
            obj1 = AttsnValues("race")
            obj1.addValue(
                "Caucasian is considered privileged (value = 1) and African-American is considered unprivileged (value = 0)")
        data.append(obj)
        data.append(obj1)
        return render_template("chooseAtt.html", data=data, dataset=dataset)
        # return render_template("metric.html",metrics=fairness_example_metrics,description=fairness_example_metrics_descr,dataset=dataset)


@app.route('/getAtts_n_vals', methods=['GET', 'POST'])
def getAtts_n_vals():
    if request.method == "POST":
        app.config["atts_n_vals"] = request.get_json()
        for element in app.config["atts_n_vals"]:
            atts_picked.append(element['name'])
            values_picked.append(element['value'])

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/getAtts', methods=['GET', 'POST'])
def getAtts():
    if request.method == "POST":
        app.config["atts"] = request.get_json()
        for element in app.config["atts"]:
            atts_picked.append(element['name'])
            values_picked.append(1)

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/metric', methods=['GET', 'POST'])
def metric():
    if app.config["DATASET"] == "upload":
        return render_template("metric.html", metrics=fairness_metrics, description=fairness_metrics_descr)
    else:
        return render_template("metric.html", metrics=fairness_metrics, description=fairness_metrics_descr,
                               dataset="upload")


def dataset_parse(dataset):
    data_to_pass = []
    data_to_read = pd.read_csv(dataset, header=0)

    for col in data_to_read.columns:
        obj = AttsnValues(col)
        for value in data_to_read[col]:
            obj.addValue(value)
        data_to_pass.append(obj)

    for col in data_to_pass:
        col.values.sort()

    return data_to_pass


if __name__ == "__main__":
    app.run(debug=True)
