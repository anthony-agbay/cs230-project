import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import optimizers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Data preparation functions
def label_type(row, thresh_offset):
    """ Converts continuous label to categorical label
    """
    if row['scaled_effect'] < 1 - thresh_offset:
        return('Deleterious')
    elif row['scaled_effect'] > 1 + thresh_offset:
        return('Beneficial')
    else:
        return('Netural')
    
def lopo_train_test_split(protein, curr_data):
    """ Splits data into train/test splits by leaving one protein out of training data
    """
    train_data = curr_data[curr_data.protein != protein].drop(['protein', 'pdb', 'resnum'], axis=1)
    test_data = curr_data[curr_data.protein == protein].drop(['protein', 'pdb', 'resnum'], axis=1)
    
    # Set up Training Data
    ## Need to one-hot encode labels
    y_train = train_data.type
    encoder_train = LabelEncoder()
    encoder_train.fit(y_train)
    y_train = to_categorical(encoder_train.transform(y_train))
    
    x_train = train_data.drop(['type'], axis=1)
    
    # Set up Tresting Data
    ## Need to one-hot encode labels
    y_test = test_data.type
    encoder_test = LabelEncoder()
    encoder_test.fit(y_test)
    y_test = to_categorical(encoder_test.transform(y_test))
    
    x_test = test_data.drop(['type'], axis=1)

    return x_train, y_train, x_test, y_test

# Model definitions
def nn_model(num_layers, num_nodes):
    model = Sequential()
    inputs = Input(shape=(969,))
    x = Dense(num_nodes, activation=tf.nn.relu)(inputs)
    for layers in range(num_layers-1):
        x = Dense(num_nodes, activation=tf.nn.relu)(x)
    outputs = Dense(3, activation=tf.nn.softmax)(x)
    opt = optimizers.Adam(learning_rate = 0.1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return(model)


def main():
    # Reading in Data
    data_path = 'data/merged.csv'
    data = pd.read_csv(data_path)

    # Setup hyperparameter search
    num_hl = [2, 4, 8, 16, 32]
    num_nodes = [25, 100, 400, 800]
    label_thresholds = [.025, .05, .1, .2]

    # Other Variables to define
    protein = 'Kka2'
    column_list = ['Hidden Layers', 'Number Nodes', 'Threshold Offset', 'Loss', 'Accuracy', 'Precision', 'Recall']

    # Evaluation Metric Storage
    eval_metrics = pd.DataFrame(columns=column_list)

    for hl in num_hl:
        for nodes in num_nodes:
            for thresholds in label_thresholds:
                # Generate the Split Training Set
                data['type'] = data.apply(lambda row: label_type(row, thresholds), axis = 1)
                data_final = data
                x_train, y_train, x_test, y_test = lopo_train_test_split(protein, data_final)
                
                # Build the Model
                print("Current Model: HL - {} | Nodes - {} | Threshold Offset - {}".format(hl, nodes, thresholds))
                curr_model = nn_model(hl, nodes)
                curr_model.fit(x_train, y_train, epochs = 20, batch_size = 10, verbose=1)
                # Calculate Evaluation Metrics
                loss, acc, prec, rec = curr_model.evaluate(x_test, y_test)
                
                # Append to Eval Storage
                eval_metrics = eval_metrics.append(pd.DataFrame([[hl, nodes, thresholds, loss, acc, prec, rec]], columns=column_list))
                eval_metrics.to_csv('hyperparameter-eval.csv')
                

if __name__ == '__main__':
    main()
