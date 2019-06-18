# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 21:54:52 2019

@author: winado

Created based on the following tutorial: http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""

#%% IMPORT NECESSARY PACKAGES

#pip install keras
#conda remove keras
#pip install numpy --upgrade

import pywt
import time
import scipy
import matplotlib
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
history = History()

import keras
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import History 
from keras.layers import LeakyReLU

#Check to see whether TF is utilizing GPU or CPU
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#First run HAR_utils.py to compile utility functions

#%% IMPORT HAR DATASETS

#folder_ucihar = 'C:/Users/winado/Desktop/Python/UCIHAR/UCI HAR Dataset/' 
#train_signals, train_labels, test_signals, test_labels = load_ucihar_data(folder_ucihar)

#INPUT_SIGNAL_FILES = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt', 
#                     'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
#                     'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

#[no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
#[no_signals_test, no_steps_test, no_components_test] = np.shape(test_signals)
#no_labels = len(np.unique(train_labels[:])) #number of labels (activities)

#print("Training dataset contains {} signals, each one of length {} and {} different components ".format(no_signals_train, no_steps_train, no_components_train))
#print("Training labels have the following distribution:\n {}".format(Counter(train_labels[:])))
#print("Test dataset contains {} signals, each one of length {} and {} different components ".format(no_signals_test, no_steps_test, no_components_test))
#print("Test labels have the following distribution:\n {}".format(Counter(test_labels[:])))

#print("Activity labels originally have the following unique values:\n {}".format(unique(test_labels)))
#Need to ensure activity labels start at zero instead of one for later use in keras
#train_labels[:] = [x - 1 for x in train_labels]
#test_labels[:] = [x - 1 for x in test_labels]
#print("Activity labels now have the following unique values:\n {}".format(unique(test_labels)))

#Define activity label key
#activities_description = {
#    0: 'walking',
#    1: 'walking upstairs',
#    2: 'walking downstairs',
#    3: 'sitting',
#    4: 'standing',
#    5: 'laying'
#}

#Shuffle data while retaining mapping with labels
#train_signals, train_labels = randomize(train_signals, np.array(train_labels))
#test_signals, test_labels = randomize(test_signals, np.array(test_labels))

#%% EXTRACT AND PLOT SPECTROGRAMS FOR CHOSEN RECORD (EACH RECORDING CONSISTS OF 9 SENSORS)

t0 = 0
N = 128
f_s = 50
t_n = 2.56
T = t_n / N
dt = 1 / f_s
time = np.arange(0, N) * dt + t0

def plot_spectrogram(ax, time, signal, waveletname = 'morl', cmap = plt.cm.seismic):
    dt = time[1] - time[0]
    scales = np.arange(1, 128)
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)    
    ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    plt.axis('off')
    
record_num = 1
signals = train_signals[record_num, : , : ]
label = train_labels[record_num]
activity_name = activities_description[label]

axtitles = ['Body Accel X', 'Body Accel Y', 'Body Accel Z',
            'Body Gyro X', 'Body Gyro Y', 'Body Gyro Z',
            'Total Accel X', 'Total Accel Y', 'Total Accel Z']

print("Training Record Chosen: {}".format(record_num))
fig = plt.figure(figsize = (8, 8))
suptitle = "All Spectrograms for Chosen Record [Activity: {}]"
fig.suptitle(suptitle.format(activity_name), fontsize=16)
for sig_comp in range(0,9):
    signal = signals[ : , sig_comp ]
    ax = fig.add_subplot(3, 3, sig_comp+1 )
    axtitle = axtitles[sig_comp]
    ax.set_title(axtitle, fontsize=12)
    #ax.set_xlabel('Seconds', fontsize=12)
    plot = plot_spectrogram(ax, time, signal)
fig.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.25)
fig.show()
print("Training Record Chosen: {}".format(record_num))


#%%EXTRACT 9D SPECTROGRAMS FOR (SUBSET OF) ALL RECORDS

scales = range(1,128)
waveletname = 'morl'

#Take subset of data when training with CPU
train_size = 250
Xtrain_CWT = np.ndarray(shape=(train_size, 127, 127, 9))

for ii in range(0,train_size):
    if ii % 50 == 0:
        print(ii)
    for jj in range(0,9):
        signal = train_signals[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:127]
        Xtrain_CWT[ii, :, :, jj] = coeff_

test_size = 100
Xtest_CWT = np.ndarray(shape=(test_size, 127, 127, 9))
for ii in range(0,test_size):
    if ii % 10 == 0:
        print(ii)
    for jj in range(0,9):
        signal = test_signals[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:127]
        Xtest_CWT[ii, :, :, jj] = coeff_
        
#%%PREP DATA FOR CNN USING KERAS

y_train = list(train_labels[:train_size])
y_test = list(test_labels[:test_size])

#Specify spectrogram image dimensions
img_x = 127
img_y = 127
img_z = 9
num_classes = 6

#reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
#because the spectrograms are essentially grayscale, we only have a single channel - RGB colour images would have 3
input_shape = (img_x, img_y, img_z)

#convert the data to the right type
#x_train = x_train.reshape(x_train.shape[0], img_x, img_y, img_z)
#x_test = x_test.reshape(x_test.shape[0], img_x, img_y, img_z)
Xtrain_CWT = Xtrain_CWT.astype('float32')
Xtest_CWT = Xtest_CWT.astype('float32')

print('Xtrain_CWT shape:', Xtrain_CWT.shape)
print(Xtrain_CWT.shape[0], 'train samples')
print(Xtest_CWT.shape[0], 'test samples')

#convert class vectors to binary class matrices - this is for use in categorical_crossentropy loss below
Ytrain_CWT = keras.utils.to_categorical(y_train, num_classes)
Ytest_CWT = keras.utils.to_categorical(y_test, num_classes)

#%%BUILD AND TRAIN CNN MODEL USING ONLY CWT FEATURES (SPECTROGRAM IMAGES)

#Specify hyperparameters (may need to tune to optimize performance)
def train_CNN(batch_size=64, epochs=100, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.025))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.025))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.Adam(lr=lr, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

    model.fit(Xtrain_CWT, Ytrain_CWT, batch_size=batch_size, 
              epochs=epochs, verbose=1, 
              validation_data=(Xtest_CWT, Ytest_CWT), 
              callbacks=[history])

    train_score = model.evaluate(Xtrain_CWT, Ytrain_CWT, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    test_score = model.evaluate(Xtest_CWT, Ytest_CWT, verbose=0)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

#Plot model performance over all epochs
def plot_epochs(epochs):
    fig, axarr = plt.subplots(figsize=(12,6), ncols=2)
    axarr[0].plot(range(1, epochs+1), history.history['acc'], label='train score')
    axarr[0].plot(range(1, epochs+1), history.history['val_acc'], label='test score')
    axarr[0].set_xlabel('Number of Epochs', fontsize=18)
    axarr[0].set_ylabel('Accuracy', fontsize=18)
    axarr[0].set_ylim([0,1])
    axarr[1].plot(range(1, epochs+1), history.history['acc'], label='train score')
    axarr[1].plot(range(1, epochs+1), history.history['val_acc'], label='test score')
    axarr[1].set_xlabel('Number of Epochs', fontsize=18)
    axarr[1].set_ylabel('Accuracy', fontsize=18)
    axarr[1].set_ylim([0.75,1])
    plt.legend()
    plt.show()

train_CNN(batch_size=32, epochs=5, lr=0.001)
plot_epochs(epochs=5)
