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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def save_tflite_model(model):
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    #saving converted model in "converted_model.tflite" file
    open(path+user+"unimib.tflite", "wb").write(tflite_model)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def save_tflite_quantized_model(model):
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()    
    open(path+user+"unimib-quantized.tflite", "wb").write(tflite_quant_model)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_lite_model(lite_model, testX, testY):
  interpreter = tensorflow.lite.Interpreter(model_path=lite_model)
  interpreter.allocate_tensors()
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()  
  predictions = []
  for sample in testX:      
    sample = sample.astype(np.float32)
    sample = np.expand_dims(sample, axis=0)    
    interpreter.set_tensor(input_details[0]['index'], sample)        
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data)        
    result = np.where(output_data == np.amax(output_data))
    predictions.append(int(result[1]) + 1)
    #print(predictions[0])
    correct_pred = 0
  print(len(testY))
  print(len(predictions))
  for i in range(len(testY)):    
    if int(np.argmax(testY[i])+1) == predictions[i]:
      correct_pred = correct_pred + 1
    #print(correct_pred)
  accuracy = correct_pred / len(testY)
  print ('Accuracy: %.2f' % (accuracy*100))    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def show_confusion_matrix(validations, predictions, tag): 
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
                fmt='d')
    figure = ss.get_figure()    
    figure.savefig(path+'CM-Model-Testing' + tag + '.eps')
    plt.title('Confusion Matrix')    
    tick_marks = np.arange(n_outputs) + 0.5
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS, rotation=0)
    plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def setup_data():
    # load data
    data = import_data(path+"data/")

    #Get user data samples:
    userData = data[user_data].values
    userData = np.reshape(userData,(userData.shape[0],n_timesteps, n_features))   
    ##Get Labels:
    userLabels = data[user_labels][0].values
    userLabels = np.asarray(pd.get_dummies(userLabels), dtype = np.float32)
    
    trainX, testX, trainY, testY = sk.train_test_split(userData,userLabels,test_size=0.5,random_state = 1, stratify=userLabels)
    if split_size != 0:
      trainX, dumpX, trainY, dumpY = sk.train_test_split(trainX,trainY,test_size=split_size,random_state = 1, stratify=trainY)
    
    print("Shape of retraining data:")
    print(trainX.shape)
    print(trainY.shape)
    print("Shape of testing data:")
    print(testX.shape)
    print(testY.shape)
    logfile.write(str(trainX.shape))      
    logfile.write("\r\n")
    logfile.write(str(trainY.shape))
    logfile.write("\r\n")

    return trainX, testX, trainY, testY      
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
def start_model_retraining():
    trainX, testX, trainY, testY = setup_data()    
    model = tensorflow.keras.models.load_model(path+model_name)
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    score = accuracy * 100.0
    print('Initial Accuracy: %.3f' % (score))
    logfile.write('Initial Accuracy: %.3f\r\n' % (score)) 

    y_pred_test = model.predict(testX)
    #predict test samples (select the class with the highest probability):
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(testY, axis=1)
    tag="_initial_user"+user
    show_confusion_matrix(max_y_test, max_y_pred_test, tag)
    print(classification_report(max_y_test, max_y_pred_test))

    print(model.summary())

    #Fit network     
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    score = accuracy * 100.0
    print('Accuracy after retraining: %.3f' % (score))

    if split_size == 0.0:
      y_pred_test = model.predict(testX)
      #predict test samples (select the class with the highest probability):
      max_y_pred_test = np.argmax(y_pred_test, axis=1)
      max_y_test = np.argmax(testY, axis=1)
      tag="_retrained_user_"+user    
      show_confusion_matrix(max_y_test, max_y_pred_test, tag)
      print(classification_report(max_y_test, max_y_pred_test))
      model.save(path+"unimib_user_"+user+".h5")
      print("Saved model to disk")
    return model, testX, testY
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LABELS = ['StandingUpFS',
          'StandingUpFL',
          'Walking',
          'Running',
          'GoingUpS',
          'Jumping',
          'GoingDownS',
          'LyingDownFS',
          'SittingDown',
          'FallingForw',
          'FallingRight',
          'FallingBack',
          'HittingObstacle',
          'FallingWithPS',
          'FallingBackSC',
          'Syncope',
          'FallingLeft']
#configure all parameters, files, split, augmentation here:
path = "/content/drive/MyDrive/HAR/unimib/retrain/" #folder containing input data
filename = path+"logs/"+datetime.now().strftime("d%d-%m-%Y-t%H-%M-%S.txt")

logfile = open(filename, "w")
logfile.write("Start training the model!\r\n")

n_timesteps, n_features, n_outputs = 151, 3, 17

batch_size = 64
epochs = 20
verbose=2
user = '15'

user_data = 'test_data_user_' + user
user_labels = 'test_label_user_' + user

model_name="unimib-loo.h5"

for split_size in np.arange(0.1, 1, 0.1):
  print(split_size)
  model, testX, testY = start_model_retraining()



#save_tflite_model(model)
#save_tflite_quantized_model(model)

#test_lite_model(path+ user+'unimib.tflite', testX, testY)
#test_lite_model(path+ user +'unimib-quantized.tflite', testX, testY)
