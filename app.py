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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

global protected_variables, fairness_metrics, mitigation_algorithms, protected_variables_values


@app.route('/')
def index():
    import flask
    print(flask.__version__)
    return render_template('index.html')


@app.route('/selection', methods=['GET', 'POST'])
def selection():
    return render_template("selection.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'file' in request.files:
        file_to_upload = request.files['file']
        if file_to_upload.filename != '':
            file_path = os.path.join(app.config["FILES_UPLOADS"], file_to_upload.filename)
            file_to_upload.save(file_path)
            app.config["UPLOADED_FILE"] = file_path
            app.config["DATASET"] = "upload"

            # Check if 'outcome' column exists
            import pandas as pd
            df = pd.read_csv(file_path)
            if 'outcome' in df.columns:
                return jsonify({"success": True, "message": "File uploaded successfully !"})
            else:
                return jsonify({"success": False,
                                "message": "No column 'outcome' found. Please rename your prediction column to 'outcome'."})
    return jsonify({"success": False, "message": "No file uploaded."})


@app.route('/Choose_att', methods=['GET', 'POST'])
def Choose_att():
    if request.method == 'POST':
        selected_models = request.form.get('model')  # Update the form field name to 'model'
        if selected_models:
            selected_models = selected_models.strip('[]').strip('"')
            app.config['SELECTED_MODEL'] = selected_models

    data = att_values.dataset_parse(app.config.get("UPLOADED_FILE"))
    return render_template("chooseAtt.html", data=data)


@app.route('/example/<dataset>', methods=['GET', 'POST'])
def example(dataset):
    app.config["DATASET"] = "example"
    return render_template('example_att.html', data=expls.get_example_attributes(dataset))


@app.route('/metric', methods=['GET', 'POST'])
def metric():
    if request.method == "POST":
        app.config['atts_n_values_picked'] = request.get_json()

    if app.config["DATASET"] == "upload":
        metrics = constants.fairness_metrics
        description = constants.fairness_metrics_descr
    else:
        metrics = constants.fairness_example_metrics
        description = constants.fairness_example_metrics_descr

    return render_template("metric.html", metrics=metrics,
                           description=description)


@app.route('/fairness_report', methods=['GET', 'POST'])
def fairness_report():
    metrics = request.form.getlist('metrics')
    threshold = request.form['threshold']
    app.config['Threshold'] = threshold
    app.config['Metrics'] = metrics

    if app.config["DATASET"] == "upload":

        model_metrics, data = prcs.train_and_evaluate(atts_n_vals_picked=app.config["atts_n_values_picked"],
                                                      path_to_csv=app.config["UPLOADED_FILE"],
                                                      metrics_to_calculate=app.config['Metrics'],
                                                      threshold=app.config['Threshold'],
                                                      model_name=app.config["SELECTED_MODEL"])
    else:
        model_metrics, data = expls.train_and_evaluate(dataset=pp.config["DATASET"],
                                                       atts_n_vals_picked=app.config["atts_n_values_picked"],
                                                       metrics_to_calculate=app.config['Metrics'])

    app.config["DATA"] = data
    app.config["MODEL_DATA"] = model_metrics

    return render_template("fairness_report.html", data=data, threshold=app.config['Threshold'],
                           model_metrics=model_metrics)


@app.route('/algorithms', methods=['GET', 'POST'])
def algorithms():
    if app.config["DATASET"] == "upload":
        return render_template("algorithms.html", algorithms=constants.mitigation_algorithms,
                               description=constants.mitigation_algorithms_descr)
    else:
        return render_template("algorithms.html", algorithms=mitigation_example_algorithms,
                               description=mitigation_example_algorithms_descr)


@app.route('/mitigation_report', methods=['GET', 'POST'])
def mitigation_report():
    app.config['Algorithms'] = request.form.getlist('algorithms')

    data = prcs.get_mitigated_results(app.config['UPLOADED_FILE'],
                                      app.config['SELECTED_MODEL'],
                                      app.config['atts_n_values_picked'],
                                      app.config['Algorithms'],
                                      app.config['DATA'],
                                      app.config['MODEL_DATA'],
                                      app.config['Threshold'])

    return render_template("mitigation_report.html", data=data, threshold=app.config['Threshold'])


if __name__ == "__main__":
    app.run(debug=True)
