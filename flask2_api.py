from flask import Flask, request
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from keras.models import load_model

app = Flask(__name__)
Swagger(app)

model = load_model('model.h5')


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict', methods=["Get"])
def predict_note_authentication():
    """Let's Authenticate the Banks Note [Authenticate Note : 1 , Not Authenticate Note: 0]
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = model.predict_classes([[variance, skewness, curtosis, entropy]])
    # print(prediction)
    return str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note [Authenticate Note : 1 , Not Authenticate Note: 0]
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = model.predict_classes(df_test)

    return str(prediction)


if __name__ == '__main__':
    app.run()
    # host = '0.0.0.0', port = 8000