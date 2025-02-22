import tensorflow
import sklearn.model_selection as sk
from matplotlib import pyplot as plt
import os
import os.path
import glob
import numpy as np
# lstm model
import random
from numpy  import mean
from numpy  import std
from numpy  import dstack
from pandas import read_csv
import pandas as pd
import seaborn as sns
from scipy import stats
from IPython.display import display, HTML
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History
from sklearn.metrics import mean_squared_error
from datetime import datetime
 
from matplotlib   import pyplot
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

#%%%%%%%%%%%%%%%%%%%Initialization information%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
print('Tensorflow version:' , tensorflow.__version__)
# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', tensorflow.keras.__version__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def import_data(path):
    files = glob.glob(path +'/*.csv')
    list_of_dfs = {}
    dfs = []
    for i in range(0,len(files)):
        dfs.append(os.path.basename(files[i])[:-4])
    for df, file in zip(dfs, files):        
        list_of_dfs[df] = read_csv(file, header = None)
    print("Loaded data successfully!")
    return list_of_dfs
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
def show_confusion_matrix(validations, predictions): 
    matrix = metrics.confusion_matrix(validations, predictions)        
    matrix1 = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)
    plt.figure(figsize=(6, 4))
    ss = sns.heatmap(matrix1,
                cmap=plt.cm.Blues,
                linecolor='white',
                linewidths=1,
                annot=matrix,
                cbar=False,
                xticklabels=LABELS,
                yticklabels=LABELS,                
                fmt='')
    figure = ss.get_figure()    
    figure.savefig('confusion-matrix.eps', dpi=400)
    plt.title('Confusion Matrix')    
    tick_marks = np.arange(n_outputs) + 0.5
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS, rotation=0)
    plt.show()
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
class myCallback(tensorflow.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}):   
      if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):   
        print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
        self.model.stop_training = True
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
def build_model(trainX, trainY, validationX, validationY):
    history = History()
    model = Sequential()
    #model.add(Bidirectional(LSTM(lstm_output_units, return_sequences=False), input_shape=(n_timesteps,n_features)))    
    model.add(LSTM(lstm_output_units, return_sequences=False, input_shape=(n_timesteps,n_features)))    
    #model.add(LSTM(lstm_output_units))
    for a in range(hidden_layers):
      model.add(Dropout(dropout))
      model.add(Dense(hidden_units, activation='tanh'))   
    model.add(Dense(n_outputs, activation='softmax'))

    callbacks = myCallback()
    optimizer = optimizers.Adam(lr=learning_rate)    

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    #Fit network    
    history = model.fit(trainX, trainY, validation_data=(validationX, validationY), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[callbacks])
    print(history.history['val_accuracy'])
    maxAccuracy = max(history.history['val_accuracy'])    
    mEpoch = (history.history['val_accuracy']).index(maxAccuracy)
    print("Max validation accuracy: %2.2f%% at epoch number %d, batch size: %d, hidden units: %d" %(maxAccuracy*100, mEpoch, batch_size, hidden_units))
    print("================================================================================")
    logfile.write("Max validation accuracy: %2.2f%% at epoch number %d, batch size: %d, hidden units: %d \r\n" %(maxAccuracy*100, mEpoch, batch_size, hidden_units))
    logfile.write("================================================================================\r\n")
    return model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def setup_data():
    # load data
    data = import_data(path)

    #Get training data:
    trainX = data[training_data].values
    trainX = np.reshape(trainX,(trainX.shape[0],n_timesteps, n_features))   
    trainY = data[training_labels][0].values 
    trainY = np.asarray(pd.get_dummies(trainY), dtype = np.float32)
    
    #Get validation data:
    validationX = data[validation_data].values
    validationX = np.reshape(validationX,(validationX.shape[0], n_timesteps, n_features))    
    validationY = data[validation_labels][0].values 
    validationY = np.asarray(pd.get_dummies(validationY), dtype = np.float32)

    #Get testing data:
    testX = data[testing_data].values
    testX = np.reshape(testX,(testX.shape[0], n_timesteps, n_features))    
    testY = data[testing_labels][0].values
    testY = np.asarray(pd.get_dummies(testY), dtype = np.float32)

    print("Shape of training data:")    
    print(trainX.shape)
    print(trainY.shape)
    print("Shape of validation data:")    
    print(validationX.shape)
    print(validationY.shape)
    print("Shape of testing data:")
    print(testX.shape)
    print(testY.shape)
    
    logfile.write(str(trainX.shape))      
    logfile.write("\r\n")
    logfile.write(str(trainY.shape))
    logfile.write("\r\n")

    return trainX, validationX, testX, trainY, validationY, testY      
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def save_tflite_model(model):
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    #saving converted model in "converted_model.tflite" file
    open(path+"unimib.tflite", "wb").write(tflite_model)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def save_tflite_quantized_model(model):
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()    
    open(path+"unimib-quantized.tflite", "wb").write(tflite_quant_model)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def start_model_training():
    trainX, validationX, testX, trainY, validationY, testY = setup_data()    
    model = build_model(trainX, trainY, validationX, validationY)    
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    score = accuracy * 100.0
    print('Accuracy: %.3f' % (score))
    logfile.write('Accuracy: %.3f\r\n' % (score))
 
    y_pred_test = model.predict(testX)
    #predict test samples (select the class with the highest probability):
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(testY, axis=1)
 
    show_confusion_matrix(max_y_test, max_y_pred_test)
    print(classification_report(max_y_test, max_y_pred_test))
    model.save(path+model_name+".h5")
    print("Saved model to disk")
    return model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LABELS = ['StandingUpFromSitting',
          'StandingUpFromLaying',
          'Walking',
          'Running',
          'GoingUpStairs',
          'Jumping',
          'GoingDownStairs',
          'LyingDownFromSittng',
          'SittingDown',
          'F-Forward',
          'F-Right',
          'F-Backward',
          'F-HittingObstacle',
          'F-ProtectionStrategy',
          'F-BackSittingChair',
          'F-Syncope',
          'F-Left']
#configure all parameters, files, split, augmentation here:
path = "/content/drive/MyDrive/HAR/unimib/" #folder containing input data
filename = path+"logs/"+datetime.now().strftime("d%d-%m-%Y-t%H-%M-%S.txt")

logfile = open(filename, "w")
logfile.write("Start training the model!\r\n")

n_timesteps, n_features, n_outputs = 151, 3, 17

#hyperparameters:    
lstm_output_units = 512
hidden_layers=1
hidden_units = 512
dropout = 0.5
learning_rate = 0.001
batch_size = 64
epochs = 400
verbose=2
ACCURACY_THRESHOLD = 0.74


print("lstm_output_units: %d hidden_layers: %d hidden_units: %d, dropout: %1.1f learning_rate: %1.5f batch_size: %d epochs: %d" %(lstm_output_units, hidden_layers, hidden_units, dropout, learning_rate, batch_size, epochs))
logfile.write("hidden_layers: %d hidden_units: %d, dropout: %1.1f learning_rate: %1.5f batch_size: %d epochs: %d\r\n" %(hidden_layers, hidden_units, dropout, learning_rate, batch_size, epochs))

training_data = 'acc_train_data'      
training_labels = 'acc_train_labels'    
validation_data = 'acc_val_data'
validation_labels = 'acc_val_labels'
testing_data = 'acc_test_data'
testing_labels = 'acc_test_labels'

model_name = "retrain/unimib-loo"
model = start_model_training()
#save_tflite_model(model)
#save_tflite_quantized_model(model)
