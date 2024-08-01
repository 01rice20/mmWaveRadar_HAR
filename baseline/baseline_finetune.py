import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from livelossplot.keras import PlotLossesCallback
import h5py, glob, re
import numpy as np
import itertools, random
from cnn_utils import *
from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import classification_report, confusion_matrix
from time import time
from numpy.random import seed
from keras import backend as K
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

time1 = time()
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(2)
    np.random.seed(1)
    random.seed(1254)
    tf.random.set_seed(89)

reset_random_seeds()

### dataset setting
all_data = []
all_data.append('./dataset/10GHz.hdf5')
all_data.append('./dataset/24GHz.hdf5')
all_data.append('./dataset/77GHz.hdf5')
all_data.append('./dataset/all.hdf5')
print('Real Dataset: '+str(all_data))
data_10 = h5py.File(all_data[0], "r")
data_24 = h5py.File(all_data[1], "r")
data_77 = h5py.File(all_data[2], "r")
data_all = h5py.File(all_data[3], "r")

X_train_10 = np.array(data_10["train_img"])
Y_train_10 = np.array(data_10["train_labels"])
X_test_10 = np.array(data_10["test_img"])
Y_test_10 = np.array(data_10["test_labels"])
print('10 ghz Dataset''s Number of training samples: ', len(Y_train_10))
print('10 ghz Dataset''s Number of test samples: ', len(Y_test_10))
X_train_24 = np.array(data_24["train_img"])
Y_train_24 = np.array(data_24["train_labels"])
X_test_24 = np.array(data_24["test_img"])
Y_test_24 = np.array(data_24["test_labels"])
print('24 ghz Dataset''s Number of training samples: ', len(Y_train_24))
print('24 ghz Dataset''s Number of test samples: ', len(X_test_24))
X_train_77 = np.array(data_77["train_img"])
Y_train_77 = np.array(data_77["train_labels"])
X_test_77 = np.array(data_77["test_img"])
Y_test_77 = np.array(data_77["test_labels"])
print('77 ghz Dataset''s Number of training samples: ', len(Y_train_77))
print('77 ghz Dataset''s Number of test samples: ', len(X_test_77))
X_train_all = np.array(data_all["train_img"])
Y_train_all = np.array(data_all["train_labels"])
X_test_all = np.array(data_all["test_img"])
Y_test_all = np.array(data_all["test_labels"])
print('All Dataset''s Number of training samples: ', len(X_train_all))
print('All. Dataset''s Number of test samples: ', len(X_test_all))
data_10.close()
data_24.close()
data_77.close()
data_all.close()

num_class = 11

x_train_10 = X_train_10/255.
x_test_10 = X_test_10/255.
y_train_10 = convert_to_one_hot(Y_train_10, num_class).T
y_test_10 = convert_to_one_hot(Y_test_10, num_class).T
print ("X_train_10 shape: " + str(x_train_10.shape))
print ("Y_train_10 shape: " + str(y_train_10.shape))
print ("X_test_10 shape: " + str(x_test_10.shape))
print ("Y_test_10 shape: " + str(y_test_10.shape)+"\n\n")

x_train_24 = X_train_24/255.
x_test_24 = X_test_24/255.
y_train_24 = convert_to_one_hot(Y_train_24, num_class).T
y_test_24 = convert_to_one_hot(Y_test_24, num_class).T
print ("X_train_24 shape: " + str(x_train_24.shape))
print ("Y_train_24 shape: " + str(y_train_24.shape))
print ("X_test_24 shape: " + str(x_test_24.shape))
print ("Y_test_24 shape: " + str(y_test_24.shape)+"\n\n")

x_train_77 = X_train_77/255.
x_test_77 = X_test_77/255.
y_train_77 = convert_to_one_hot(Y_train_77, num_class).T
y_test_77 = convert_to_one_hot(Y_test_77, num_class).T
print ("X_train_77 shape: " + str(x_train_77.shape))
print ("Y_train_77 shape: " + str(y_train_77.shape))
print ("X_test_77 shape: " + str(x_test_77.shape))
print ("Y_test_77 shape: " + str(y_test_77.shape)+"\n\n")

X_train_all = X_train_all/255.
x_test_all = X_test_all/255.
y_train_all = convert_to_one_hot(Y_train_all, num_class).T
y_test_all = convert_to_one_hot(Y_test_all, num_class).T
print ("X_train_all shape: " + str(X_train_all.shape))
print ("y_train_all shape: " + str(y_train_all.shape))
print ("x_test_all shape: " + str(x_test_all.shape))
print ("y_test_all shape: " + str(y_test_all.shape)+"\n\n")

# set flag: '0' for 10ghz, '1' for 24ghz, '2' for 77ghz and '3' for 4GHz USRP TOBB Data
def select_data(radar):
    reset_random_seeds()
    if radar == 0:
        X_Train = x_train_10
        Y_Train = y_train_10
        print("10 GHz Dataset")
        X_Test = x_test_10
        Y_Test = y_test_10
    elif radar == 1:
        X_Train = x_train_24
        Y_Train = y_train_24
        X_Test = x_test_24
        Y_Test = y_test_24
        print("24 GHz Dataset")
    elif radar == 2: 
        X_Train = x_train_77
        Y_Train = y_train_77
        X_Test = x_test_77
        Y_Test = y_test_77
        print("77 GHz Dataset")
    elif radar==3:
        X_Train = X_train_all
        Y_Train = y_train_all
        X_Test = x_test_all
        Y_Test = y_test_all

    return X_Train, Y_Train, X_Test, Y_Test

### Load Model & Weight
radar = 0 # '0' 10 ghz, '1' 24 ghz, or '2' 77, '3' all
model_addr = []
model_addr.append('./weights/10GHz.json')
model_addr.append("./weights/24GHz.json")
model_addr.append('./weights/77GHz.json')
model_addr.append('./weights/111GHz.json')

weight_addr = []
weight_addr.append('./weights/10GHz.h5')
weight_addr.append("./weights/24GHz.h5")
weight_addr.append("./weights/77GHz.h5")
weight_addr.append('./weights/111GHz.h5')
print(model_addr)
print(weight_addr)
model_file = model_addr[radar]
weights_file = weight_addr[radar]

json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_file)
print(model_file+' & '+weights_file+ " loaded model from disk")
loaded_model.summary()

(X_train, Y_train, X_test, Y_test) = select_data(radar)
epochs = [100]
batch_size = [8, 16]
dense_1 = [64, 64, 128, 128, 256, 256]
dense_2 = [32, 64, 64, 128, 128, 256]
learn_rate = [0.0002, 0.0001]
acc_hist = []
hist_hist = []
drop = [0.5]
    
depth = int((len(loaded_model.layers)-2)/2) # for CAE
print(depth)
# layer_name = 'max_pooling2d_6'
# model2 = Model(inputs=loaded_model.input , outputs=loaded_model.get_layer(layer_name).output)
# model2 = Model(inputs=loaded_model.input , outputs=loaded_model.layers[depth].output) # for CAE

for i in range(len(epochs)):
    for j in range(len(batch_size)):
        for k in range(len(dense_1)):
            for m in range(len(learn_rate)):
                for n in range(len(drop)):
                    
                    model2 = Model(inputs=loaded_model.input , outputs=loaded_model.layers[depth].output)
                    model = Sequential()

                    model.add(model2)
                    model.add(Flatten())
                    model.add(Dense(dense_1[k], activation='relu'))
                    model.add(Dropout(drop[n]))

                    model.add(Dense(dense_2[k], activation='relu'))
                    model.add(Dropout(drop[n]))

                    model.add(Dense(num_class))
                    model.add(Activation('softmax'))

                    optim = Adam(learning_rate=learn_rate[m], decay = 1e-06)
                    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
                    history = model.fit(X_train, Y_train,
                                    batch_size=batch_size[j],
                                    epochs=epochs[i],
                                    verbose=0,
                                    validation_data=(X_test, Y_test),
                                    shuffle=False
                                         )
                    y_pred = model.predict(X_test)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_true = np.argmax(Y_test, axis=1)
                    accuracy = accuracy_score(y_true, y_pred_classes)
                    precision = precision_score(y_true, y_pred_classes, average='weighted')
                    recall = recall_score(y_true, y_pred_classes, average='weighted')
                    f1 = f1_score(y_true, y_pred_classes, average='weighted')

                    acc = history.history['val_accuracy'][-1]
                    acc_hist.append(acc)
                    hist_hist.append(history.history)
                    print('Params for '+str(radar) +' GHz: epochs= '+str(epochs[i])+', batch_size= '+str(batch_size[j])+
                          ', dense_1= '+str(dense_1[k])+', dense_2= '+str(dense_2[k])+', learn_rate= '+str(learn_rate[m])+
                          ', Acc='+str(acc)+', Accuracy=' +str(accuracy)+', Precision='+str(precision)+', Recall='+str(recall)+', F1-score='+str(f1))
                
time2 = time()
print("Total cost time: ", (time2 - time1)/60)