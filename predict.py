import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling TF debugging info

import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from model import deep_regression

import argparse
from joblib import load

def parse_args():
    parser = argparse.ArgumentParser(
        description='Predictions with Trained Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    # input CSV path
    parser.add_argument(
        '-i', '--input',
        type=str, 
        help='Path to input CSV file',
        required=True
    )

    # path to trained weights
    parser.add_argument(
        '-w', '--weight_path',
        type=str, 
        default="trained_weights/",
        help='Path to trained weights directory'
    )

    # output CSV filename
    parser.add_argument(
        '-o', '--output',
        type=str, 
        help='output CSV filename',
        required=True
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # fetching arguments
    config = parse_args()

    # loading input csv 
    df = pd.read_csv(config.input)

    # fetching required columns
    X = df[["Category", "Accident_type", "Prev_Year_Value"]]

    # loading tranforms
    ft = load('assets/feature_tranform.pkl')
    lt = load('assets/label_tranform.pkl')


    X_test = ft.transform(X.values)

    # loading model
    model = deep_regression()

    # loading trained weights
    model.load_weights(os.path.join(config.weight_path, 'best_model.h5'))

    # predicting 
    Y_pred = model.predict(X_test)          # model predictions 
    Y_pred = lt.inverse_transform(Y_pred)   # tranforming

    # adding predictions as column
    df["Value"] = Y_pred.astype(int)

    # writing output as CSV
    df.to_csv(config.output)