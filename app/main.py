import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from model import deep_regression

from joblib import load
from flask import Flask, request, jsonify

import json

app = Flask(__name__)

@app.route("/", methods=['POST'])
def index():

    try:
        # fetching request data
        data = request.get_json()
        
        # composing numpy array
        X = np.array([[
            data['Category'],
            data['Accident_type'],
            data['Prev_Year_Value'],
        ]])
    except Exception as e:
        return jsonify({
            "prediction": f"404: {e}"
        })

    # loading tranforms
    ft = load('assets/feature_tranform.pkl')
    lt = load('assets/label_tranform.pkl')

    # tranforming data
    X_test = ft.transform(X)

    # loading model
    model = deep_regression()

    # loading trained weights
    model.load_weights('trained_weights/best_model.h5')

    # predicting
    Y_pred = model.predict(X_test)          # model predictions
    Y_pred = lt.inverse_transform(Y_pred)   # tranforming

    # float numpy array to int
    pred = Y_pred.astype(int).item()

    return jsonify({
        "prediction": pred
    })
