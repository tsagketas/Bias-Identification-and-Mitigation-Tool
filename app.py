import os
import pandas as pd
import csv
import argparse
from flask import Flask, render_template, url_for, redirect, request, flash, make_response, jsonify, Response
import json
from werkzeug.utils import secure_filename
import proccess as prcs
import examples as expls
import att_values
import constants

app = Flask(__name__, template_folder='templates')
app.config['FILES_UPLOADS'] = r'C:\Users\ceid6\Desktop\diplomatiki\static\Uploaded_files'
TEMPLATES_AUTO_RELOAD = True

global protected_variables, fairness_metrics, mitigation_algorithms, protected_variables_values


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/selection', methods=['GET', 'POST'])
def selection():
    return render_template("selection.html")


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
            data = att_values.dataset_parse(app.config["UPLOADED_FILE"])

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


@app.route('/metric', methods=['GET', 'POST'])
def metric():
    if request.method == "POST":
        app.config['atts_n_values_picked'] = request.get_json()

    return render_template("metric.html", metrics=constants.fairness_metrics,
                           description=constants.fairness_metrics_descr)
@app.route('/fairness_report', methods=['GET', 'POST'])
def fairness_report():
    if app.config["DATASET"] == "upload":

        metrics = request.form.getlist('metrics')
        threshold = request.form['threshold']
        app.config['Threshold'] = threshold
        app.config['Metrics'] = metrics

        data = prcs.get_fairness_metrics(app.config["atts_n_values_picked"], path_to_csv=app.config["UPLOADED_FILE"],
                                         metrics_to_calculate=app.config['Metrics'], threshold=app.config['Threshold'])
    else:
        the_b_metrics, truth, the_metrics, datasets, app.config["privileged_groups"], app.config["unprivileged_groups"], \
            app.config["the_b_datasets"], app.config["train_datasets"], app.config["test_datasets"], app.config[
            "b_details"] = expls.example(app.config["DATASET"], app.config["atts"])
        app.config["Datasets"] = datasets
        data, flag = prcs.get_data(the_metrics, the_b_metrics, metrics_picked, atts_picked, values_picked, 0,
                                   app.config["DATASET"])

    app.config["DATA"] = data

    return render_template("fairness_report.html", data=data, threshold=app.config['Threshold'])


@app.route('/algorithms', methods=['GET', 'POST'])
def algorithms():
    if app.config["DATASET"] == "upload":
        return render_template("algorithms.html", algorithms=mitigation_algorithms,
                               description=mitigation_algorithms_descr)
    else:
        return render_template("algorithms.html", algorithms=mitigation_example_algorithms,
                               description=mitigation_example_algorithms_descr)


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


@app.route('/mit_graphs', methods=['GET', 'POST'])
def mit_graphs():
    if request.method == "POST":
        algs = request.get_json()
        for val in algs['name']:
            algorithms_picked.append(val)

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


if __name__ == "__main__":
    app.run(debug=True)
