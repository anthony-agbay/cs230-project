import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import optimizers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data preparation functions  
def lopo_train_test_split(protein, curr_data):
    """ Splits data into train/test splits by leaving one protein out of training data
    """
    train_data = curr_data[curr_data.protein != protein].drop(['protein', 'pdb', 'resnum'], axis=1)
    test_data = curr_data[curr_data.protein == protein].drop(['protein', 'pdb', 'resnum'], axis=1)
    
    y_train = train_data.type
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    y_train_oh = np_utils.to_categorical(encoded_y_train)
    
    scaler_train = StandardScaler()
    x_train = train_data.drop(['type'], axis=1)
    x_columns = x_train.columns
    x_train = scaler_train.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=x_columns)
    
    y_test = test_data.type
    encoder = LabelEncoder()
    encoder.fit(y_test)
    encoded_y_test = encoder.transform(y_test)
    y_test_oh = np_utils.to_categorical(encoded_y_test)
    
    scaler_test = StandardScaler()
    x_test = test_data.drop(['type'], axis=1)
    x_test = scaler_test.fit_transform(x_test)
    x_test = pd.DataFrame(x_test, columns=x_columns)

    return x_train, y_train_oh, x_test, y_test_oh

# Model definitions
def nn_model(num_layers, num_nodes):
    model = Sequential()
    inputs = Input(shape=(107,))
    x = Dense(num_nodes, activation=tf.nn.relu, kernel_regularizer='l2')(inputs)
    x = Dropout(0.2)(x)
    for layers in range(num_layers-1):
        x = Dense(num_nodes, activation=tf.nn.relu, kernel_regularizer='l2')(x)
        x = Dropout(0.2)(x)
    outputs = Dense(3, activation=tf.nn.softmax)(x)
    opt = optimizers.Adam(learning_rate = 0.001)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return(model)


def main():
    # Reading in Data
    data_path = 'data/upsampled_data.csv'
    data = pd.read_csv(data_path)
    data = data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

    # Setup hyperparameter search
    num_hl = [2, 4, 6, 8, 10]
    num_nodes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Other Variables to define
    protein = 'Kka2'
    column_list = ['Hidden Layers', 'Number Nodes', 'Loss', 'Accuracy', 'Precision', 'Recall']

    # Evaluation Metric Storage
    eval_metrics = pd.DataFrame(columns=column_list)

    # Model Callbacks
    my_callbacks = [EarlyStopping(patience=3)]

    for hl in num_hl:
        for nodes in num_nodes:
            # Generate the Split Training Set
            data_final = data
            x_train, y_train, x_test, y_test = lopo_train_test_split(protein, data_final)

            # Build the Model
            print("Current Model: HL - {} | Nodes - {}".format(hl, nodes))
            curr_model = nn_model(hl, nodes)
            curr_model.fit(x_train, y_train, epochs = 100, batch_size = 64, callbacks = my_callbacks, verbose=1, validation_data=(x_test, y_test))
            # Calculate Evaluation Metrics
            loss, acc, prec, rec = curr_model.evaluate(x_test, y_test)

            # Append to Eval Storage
            eval_metrics = eval_metrics.append(pd.DataFrame([[hl, nodes, loss, acc, prec, rec]], columns=column_list))
            eval_metrics.to_csv('hyperparameter-upsampled-eval.csv')
                

if __name__ == '__main__':
    main()
