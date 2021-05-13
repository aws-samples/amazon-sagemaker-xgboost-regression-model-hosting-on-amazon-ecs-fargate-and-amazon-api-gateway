"""
Copyright 2021 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import flask
from flask import Flask, request
from io import StringIO
import json
from logging.config import dictConfig
import os
import pandas as pd
import pickle
import xgboost as xgb


dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': os.environ.get('FLASK_SERVER_LOG_LEVEL'),
        'handlers': ['wsgi']
    }
})


## Initialize the Flask app
app = Flask(__name__)


## Load the model object from the pickle file
model_pickle_file_path = os.environ.get('MODEL_PICKLE_FILE_PATH')
app.logger.debug('Loading the model object from the pickle file \'{}\'...'.format(model_pickle_file_path))
with open(model_pickle_file_path, 'rb') as model_pickle_file:
    model = pickle.load(model_pickle_file)
app.logger.debug('Completed loading the model object from the pickle file.')


## Perform prediction
def predict(pred_x_csv):
    app.logger.debug('Performing prediction...')
    response_status_code = '200'
    try:
        pred_x_df = pd.read_csv(StringIO(pred_x_csv), sep=',', header=None)
        pred_x = xgb.DMatrix(pred_x_df.values)
        response_content_body = str(model.predict(pred_x)[0])
        app.logger.info('Predicted value = {}'.format(response_content_body))
    except pd.errors.EmptyDataError:
        response_status_code = '400'
        response_content_body = 'Empty string in x columns'
        app.logger.info('Missing data :: {}'.format(response_content_body))
    except ValueError:
        response_status_code = '400'
        response_content_body = 'Invalid data in x columns'
        app.logger.info('Invalid data :: {}'.format(response_content_body))
    app.logger.debug('Completed performing prediction.')
    return response_status_code, response_content_body


## Parse input data (supported response content-type - 'text/plain' or 'application/json'):
def parse_request_data(request):
    app.logger.debug('Parsing request data...')
    request_content_type = request.content_type
    request_data = json.loads(request.get_data(as_text=True))
    # Parse the response_content_type value
    try:
        response_content_type = request_data['response_content_type']
    except KeyError:
        # Set default
        response_content_type = 'application/json'
    # Parse the pred_x_csv value
    try:
        pred_x_csv = request_data['pred_x_csv']
    except KeyError:
        # Set to empty string
        pred_x_csv = ''
    app.logger.info('Request content type = {}'.format(request_content_type))
    app.logger.info('Response content type = {}'.format(response_content_type))
    app.logger.info('Prediction x columns CSV = {}'.format(pred_x_csv))
    app.logger.debug('Completed parsing request data.')
    return response_content_type, pred_x_csv


## Format the response based on the specified content-type ('text/plain' or 'application/json')
def format_response_data(response_status_code, response_content_type, response_content_body):
    app.logger.debug('Formatting response data...')
    if response_content_type != 'text/plain':
        response_content_type = 'application/json'
        if response_status_code == '200':
            response_content_body = json.dumps({"Predicted value": response_content_body})
        else:
            response_content_body = json.dumps({"Error": response_content_body})
    response = flask.make_response(response_content_body)
    response.headers['Content-Type'] = response_content_type
    response.status = response_status_code
    app.logger.debug('Completed formatting response data.')
    return response


## Process healthcheck request
@app.route('/healthcheck', methods=['GET'])
def process_health_check():
    app.logger.debug('Executing the process_health_check() function...')
    response = flask.make_response('OK')
    response.headers['Content-Type'] = 'text/plain'
    response.status = '200'
    app.logger.debug('Completed executing the process_health_check() function.')
    return response


## Process regular request
@app.route('/', methods=['POST'])
def handler():
    app.logger.debug('Executing the handler() function...')
    # Parse the request data
    response_content_type, pred_x_csv = parse_request_data(request)
    # Perform prediction
    response_status_code, response_content_body = predict(pred_x_csv)
    # Format the response data
    response = format_response_data(response_status_code, response_content_type, response_content_body)
    app.logger.debug('Completed executing the handler() function.')
    return response


## The main function
if __name__ == '__main__':
    app.logger.debug('Starting the California Housing XGBoost Regression Inference app...')
    app.run(host=os.environ.get('FLASK_SERVER_HOSTNAME'),
            port=int(os.environ.get('FLASK_SERVER_PORT')),
            debug=bool(os.environ.get('FLASK_SERVER_DEBUG')))
