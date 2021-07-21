import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling TF debugging info

import numpy as np
import tensorflow as tf 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import argparse
from joblib import load

from model import logistic_regression
from utils import plot_loss_curve, compare_plot

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # saved weights output path
    parser.add_argument(
        '-d', '--output_dir',
        type=str, 
        default="trained_weights/",
        help='Output directory.'
    )

    # learning rate 
    parser.add_argument(
        '-lr', '--learning_rate', 
        type=float, 
        default=0.001,
        help="Optimizer's learning rate"
    )

    # epochs 
    parser.add_argument(
        '-e', '--epochs', 
        type=int, 
        default=100,
        help="Total training iterations"
    )

    # batch size 
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, 
        default=16,
        help="Total batch size"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # fetching arguments
    config = parse_args()
    
    # loading data 
    data = load('dataset/transformed_dataset.joblib')
    X_train, Y_train = data["train"]
    X_valid, Y_valid = data["valid"]
    X_test, Y_test = data["test"]

    # loading data tranforms
    ft = load('assets/feature_tranform.joblib')
    lt = load('assets/label_tranform.joblib')

    # loading model
    model = logistic_regression(input_shape=(7,))

    # compiling model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate
    )

    model.compile(optimizer=optimizer, loss="mse")
    
    # defining model callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.output_dir, 'best_model.h5'), 
            save_best_only=True, 
            save_weights_only=True, 
            verbose=1, 
            monitor='val_loss', 
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')
    ]

    # training model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        batch_size=config.batch_size, 
        epochs=config.epochs, 
        callbacks=callbacks
    )
    
    # plotting loss curves 
    plot_loss_curve(history)

    # loading best weights 
    print("\nLOADING SAVED WEIGHTS...")
    model.load_weights(os.path.join(config.output_dir, 'best_model.h5'))

    # evaluating on test set 
    print("\nEVALUATING ON TEST SET...")
    test_loss = model.evaluate(x=X_test, y=Y_test)
    print(f"Test Loss: {test_loss}")

    # compare plot 
    Y_pred = model.predict(X_test)          # model predictions 
    Y_pred = lt.inverse_transform(Y_pred)   # tranforming
    compare_plot(Y_true=Y_test, Y_pred=Y_pred)