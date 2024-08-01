import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers import Input, ZeroPadding2D,concatenate, Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, UpSampling2D, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from livelossplot.keras import PlotLossesCallback
import h5py, glob
import numpy as np
import itertools
from cnn_utils import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

np.random.seed(1)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

def select_data(radar):

    datapath = '../../dataset/baseline/*.hdf5'
    all_data = glob.glob(datapath)
    print('Real Dataset: '+str(all_data))
    data_10 = h5py.File(all_data[0], "r")
    data_24 = h5py.File(all_data[1], "r")
    data_77 = h5py.File(all_data[2], "r")
    data_all = h5py.File(all_data[3], "r")

    print('Real Dataset: '+str(all_data))
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
    print('24 ghz Dataset''s Number of training samples: ', len(X_train_24))
    print('24 ghz Dataset''s Number of test samples: ', len(X_test_24))

    X_train_77 = np.array(data_77["train_img"])
    Y_train_77 = np.array(data_77["train_labels"])
    X_test_77 = np.array(data_77["test_img"])
    Y_test_77 = np.array(data_77["test_labels"])
    print('77 ghz Dataset''s Number of training samples: ', len(X_train_77))
    print('77. Dataset''s Number of test samples: ', len(X_test_77))

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

    if radar==10:
        X_Train = x_train_10
        Y_Train = y_train_10
        X_Test = x_test_10
        Y_Test = y_test_10
    elif radar==24:
        X_Train = x_train_24
        Y_Train = y_train_24
        X_Test = x_test_24
        Y_Test = y_test_24
    elif radar==77: 
        X_Train = x_train_77
        Y_Train = y_train_77
        X_Test = x_test_77
        Y_Test = y_test_77
    elif radar==111:
        X_Train = X_train_all
        Y_Train = y_train_all
        X_Test = x_test_all
        Y_Test = y_test_all
   
    return X_Train, Y_Train, X_Test, Y_Test

def encoderX(input_img,depth,num_filter):
    #encoder
    #input = 128 x 128 x 3 (wide and thin) 128x128x3
    
    for i in range(depth):
        conv01 = Conv2D(num_filter, (3, 3), strides=(1, 1), activation='relu', padding='same')(input_img) # 128x128x8
        conv02 = Conv2D(num_filter, (9, 9), activation='relu', padding='same')(input_img)
        out01  = concatenate([conv01,conv02],axis=-1)
        pool = MaxPooling2D(pool_size=(2, 2))(out01) #14 x 14 x 32, 64x64x8
        input_img = pool
    
    return input_img

def decoderX(pool,depth,num_filter):    
    #decoder
    for i in range(depth):
        conv9 = Conv2D(num_filter, (3, 3), activation='relu', padding='same')(pool) # 16x16x16
        conv10 = Conv2D(num_filter, (9, 9), activation='relu', padding='same')(pool)
        out5  = concatenate([conv9,conv10], axis=-1)
        up   = UpSampling2D((2,2))(out5) # 64x64x16
        pool = up
    
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(up) # 128 x 128 x 3 / Conv2D(1, (3, 3)
    
    return decoded


### CAE hyp
batch_size = 16
epochs = 100
inChannel = 3
x, y = 128, 128 # height, width
lr = 0.001
optim = Adam(lr=lr)
radars = [77]
input_img = Input(shape = (x, y, inChannel))
num_class = 11
depth = [5]
num_filter = [64]

for i in range(len(radars)):
    acc_hist = []
    radar = radars[i]
    (X_train, Y_train, X_test, Y_test) = select_data(radar)
    for d in range(len(depth)):
        for f in range(len(num_filter)):
            
            autoencoder = Model(input_img, decoderX(encoderX(input_img,depth[d],num_filter[f]),depth[d],num_filter[f]))
            autoencoder.compile(loss='mean_squared_error', optimizer = optim, metrics=['accuracy'])
            autoencoder_train = autoencoder.fit(X_train, X_train, validation_data = (X_test,X_test), batch_size=batch_size,epochs=epochs, verbose=0) 
            
            model2 = Model(input_img, outputs=autoencoder.layers[4*depth[d]].output)
            model = Sequential()

            model.add(model2)
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))

            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))

            model.add(Dense(num_class))
            model.add(Activation('softmax'))

            
            model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
            history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(X_test, Y_test)
                                 )
            acc = history.history['val_accuracy'][-1]
            acc_hist.append(acc)
                    
            MODEL_FILE = "./"+str(radars[i])+"GHz.json" 
            WEIGHT_FILE ="./"+str(radars[i])+"GHz.h5" 

            model_json = autoencoder.to_json()
            with open(MODEL_FILE, "w") as json_file:
                json_file.write(model_json)
            acc = history.history['val_accuracy'][-1]
            acc_hist.append(acc)
            autoencoder.save_weights(WEIGHT_FILE)
            
            print('Parameters for '+str(radar) +' GHz: Depth = '+str(depth[d])+', Num_filter = '+str(num_filter[f])+
                  ', Accuracy = '+str(acc))