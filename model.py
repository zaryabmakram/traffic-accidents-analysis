from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras import Model
import tensorflow.keras.backend as K

def deep_regression(input_shape):
    # clearing keras session
    K.clear_session()

    # input placeholder
    X_input = Input(input_shape, name='input')

    # Fully-Connected -> ReLU
    X = Dense(units=8, activation='relu', name='fc0')(X_input)
    
    # Fully-Connected -> ReLU
    X = Dense(units=4, activation='relu', name='fc1')(X)

    # output layer
    X = Dense(units=1, activation=None, name='output')(X)

    # create model
    model = Model(inputs=X_input, outputs=X, name='Logistic_Regression_Model')

    return model