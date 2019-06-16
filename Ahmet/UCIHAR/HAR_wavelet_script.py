# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:34:12 2019

@author: nadolsw

Created based on the following tutorial: http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""
#%% IMPORT NECESSARY PACKAGES

#pip install keras
#conda remove keras

import pywt
import scipy
import matplotlib
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter

import keras
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import History 
from keras.layers import LeakyReLU

#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
history = History()

#Check that GPU is being utilized by TF
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#Check that GPU is being utilized by Keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

#%% IMPORT HAR DATA

folder_ucihar = 'C:/Users/nadolsw/Desktop/Tech/Data Science/Python/Ahmet/wavelet/UCI HAR Dataset/' 
train_signals, train_labels, test_signals, test_labels = load_ucihar_data(folder_ucihar)

#Need to ensure labels start at zero instead of one for later use in keras
train_labels[:] = [x - 1 for x in train_labels]
test_labels[:] = [x - 1 for x in test_labels]

unique(train_labels)
unique(test_labels)

activities_description = {
    0: 'walking',
    1: 'walking upstairs',
    2: 'walking downstairs',
    3: 'sitting',
    4: 'standing',
    5: 'laying'
}

[no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
[no_signals_test, no_steps_test, no_components_test] = np.shape(test_signals)
no_labels = len(np.unique(train_labels[:]))

print("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train, no_steps_train, no_components_train))
print("The train dataset contains {} labels, with the following distribution:\n {}".format(np.shape(train_labels)[0], Counter(train_labels[:])))
#print("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test, no_steps_test, no_components_test))
#print("The test dataset contains {} labels, with the following distribution:\n {}".format(np.shape(test_labels)[0], Counter(test_labels[:])))

#Shuffle data while retaining mapping with labels
train_signals, train_labels = randomize(train_signals, np.array(train_labels))
test_signals, test_labels = randomize(test_signals, np.array(test_labels))

#%% PLOT SENSOR READINGS FOR CHOSEN SIGNAL #

#Data consists of 10,000 recordings (7352 training and 2947 test cases) of 6 different activities (labels)
#Each recording is composed of 9 different smartphone sensors capturing 128 samples over 2.56 seconds (sampling rate of 50hz)

t0 = 0
N = 128
f_s = 50
t_n = 2.56
T = t_n / N
dt = 1 / f_s
time = np.arange(0, N) * dt + t0

ylabel = 'Sensor Value' 
xlabel = 'Time [Seconds]'
colors = ['r', 'g', 'b']
labels = ['x-component', 'y-component', 'z-component']
axtitles = [ ['Acceleration', 'Gyroscope', 'Total Acceleration'] ]

signal_no = 100 #Record number index (from 0 to 7351)
signals = train_signals[signal_no, :, :]
label = train_labels[signal_no]
activity_name = activities_description[label]

f, axarr = plt.subplots(nrows=3, ncols=1, figsize=(10,8))
suptitle = "Different signals for the activity: {}"
f.suptitle(suptitle.format(activity_name), fontsize=20)
for row_no in range(0,1): 
    if row_no == 0:
        for comp_no in range(0,9):
            col_no = comp_no // 3
            plot_no = comp_no % 3
            color = colors[plot_no]
            label = labels[plot_no]     
            axtitle  = axtitles[row_no][col_no]
            value_retriever = [get_values][row_no]    
            ax = axarr[col_no]
            ax.set_title(axtitle, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)     
            signal_component = signals[:, comp_no]
            x_values, y_values = value_retriever(signal_component, T, N, f_s)
            ax.plot(x_values, y_values, linestyle='-', color=color, label=label)
            ax.legend(loc='best', bbox_to_anchor=(1, 0.5)) 
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.5)
plt.show()

#%% EXTRACT AND PLOT SPECTROGRAMS FOR CHOSEN RECORD (EACH RECORDING CONSISTS OF 9 SENSORS)

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
    
record_num = 100
signals = train_signals[record_num, : , : ]
label = train_labels[record_num]
activity_name = activities_description[label]

axtitles = ['Body Accel X', 'Body Accel Y', 'Body Accel Z',
            'Body Gyro X', 'Body Gyro Y', 'Body Gyro Z',
            'Total Accel X', 'Total Accel Y', 'Total Accel Z']

fig = plt.figure(figsize = (8, 8))
suptitle = "All Spectrograms for Chosen Record. [Activity: {}]"
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


#%%EXTRACT 9D SPECTROGRAMS FOR ALL RECORDS

#uci_har_signals_train = train_signals
#uci_har_signals_test = test_signals

scales = range(1,128)
waveletname = 'morl'

train_size = 5000
train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 9))

for ii in range(0,train_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0,9):
        signal = train_signals[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:127]
        train_data_cwt[ii, :, :, jj] = coeff_

test_size = 2000
test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 9))
for ii in range(0,test_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0,9):
        signal = test_signals[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:127]
        test_data_cwt[ii, :, :, jj] = coeff_
        
#%%PREP DATA FOR CNN USING KERAS

x_train = train_data_cwt
x_test = test_data_cwt

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
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices - this is for use in categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%% SEARCH FOR SUITABLE HYPERPARAMETERS

epochs = 1

def search(batch_size, lr):
    model = Sequential()
    #model.add(Dropout(0.10)) #Dropout of the input layer only seems to aid when data is overly noisy/dirty
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

    model.fit(x_train, y_train, batch_size=batch_size, 
              epochs=epochs, verbose=1, 
              validation_data=(x_test, y_test), 
              callbacks=[history])
    
    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Batch Size: {}'.format(batch_size))
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    keras.backend.clear_session()

#Want higher LR for smaller batch sizes
search(256, 0.001)    #84.65%
search(128, 0.001)    #90.65%
search(64, 0.0001)    #92.45%
search(32, 0.0001)    #91.90%
search(16, 0.0001)    #92.05%
search(8, 0.0001)     #92.40%
search(4, 0.0001)     #90.80%
search(2, 0.00001)    #91.00%

#%%BUILD AND TRAIN CNN MODEL USING CWT

#Specify hyperparameters
batch_size = 64
epochs = 100
lr = 0.0001

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

model.fit(x_train, y_train, batch_size=batch_size, 
          epochs=epochs, verbose=1, 
          validation_data=(x_test, y_test), 
          callbacks=[history])

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

#Plot model performance over all epochs
fig, axarr = plt.subplots(figsize=(12,6), ncols=2)
axarr[0].plot(range(1, 101), history.history['acc'], label='train score')
axarr[0].plot(range(1, 101), history.history['val_acc'], label='test score')
axarr[0].set_xlabel('Number of Epochs', fontsize=18)
axarr[0].set_ylabel('Accuracy', fontsize=18)
axarr[0].set_ylim([0,1])
axarr[1].plot(range(1, 101), history.history['acc'], label='train score')
axarr[1].plot(range(1, 101), history.history['val_acc'], label='test score')
axarr[1].set_xlabel('Number of Epochs', fontsize=18)
axarr[1].set_ylabel('Accuracy', fontsize=18)
axarr[1].set_ylim([0.9,1])
plt.legend()
plt.show()


#%% NOW TAKE DIFFERENT APPROACH AND USE DWT TO EXTRACT FEATURES PER SCALE/SUBBAND

#RE-IMPORT DATA
train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data(folder_ucihar)

#Shuffle data while retaining mapping with labels
train_signals_ucihar, train_labels_ucihar = randomize(train_signals_ucihar, np.array(train_labels_ucihar))
test_signals_ucihar, test_labels_ucihar = randomize(test_signals_ucihar, np.array(test_labels_ucihar))

#VISUALIZE WAVELET DECOMPOSITION FOR CHOSEN SIGNAL (MULTIRESOLUTION ANALYSIS)
record_num=0
sig_comp_num=0
num_decomp_lvls=5
wavelet='sym5'

Z = train_signals_ucihar[record_num,:,sig_comp_num]
plot_wvlt_decomp(Z, num_decomp_lvls, wavelet)

#LOOP OVER ALL RECORDS AND SIGNAL COMPONENTS AND EXTRACT FEATURES FOR EACH SCALE OF EACH SIGNAL COMPONENT 
def get_uci_har_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            list_of_scales = pywt.wavedec(signal, waveletname, level=5)
            for scale in list_of_scales:
                features += get_features(scale)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y
#TOTAL OF 10,000 RECORDINGS OF LENGTH 128 OVER 9 SIGNAL COMPONENTS EACH WITH 10 LEVELS OF SCALE DECOMPOSITION
#EXTRACTING A TOTAL OF 16 FEATURES PER SCALE (SEE ASSOCIATED UTIL FILE)

#train_signals_ucihar=train_signals_ucihar[:100,:,:]
#test_signals_ucihar=test_signals_ucihar[:10,:,:]
#train_labels_ucihar=train_labels_ucihar[:100]
#test_labels_ucihar=test_labels_ucihar[:10]

X_train_ucihar, Y_train_ucihar = get_uci_har_features(train_signals_ucihar, train_labels_ucihar, 'rbio3.1')
X_test_ucihar, Y_test_ucihar = get_uci_har_features(test_signals_ucihar, test_labels_ucihar, 'rbio3.1')

#DETERMINE CLASSIFIERS TO USE
dict_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(), 
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XG Boost": XGBClassifier()
}

#Takes as input the X, Y matrices of the Train and Test set and fit them on all classifiers specified in dict_classifier.
def batch_classify(X_train, Y_train, X_test, Y_test, num_classifiers = 10, verbose = True):
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:num_classifiers]:
        t_start = time.process_time()
        classifier.fit(X_train, Y_train)
        t_end = time.process_time()
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models

#Display training progress and results
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    display(df_.sort_values(by=sort_by, ascending=False))

#%% TRAIN MODELS USING WAVELET-DERIVED STATISTICS
    
models = batch_classify(X_train_ucihar, Y_train_ucihar, X_test_ucihar, Y_test_ucihar, num_classifiers = 10)
display_dict_models(models)

cls = GradientBoostingClassifier(n_estimators=2000)
cls.fit(X_train_ucihar, Y_train_ucihar)
train_score = cls.score(X_train_ucihar, Y_train_ucihar)
test_score = cls.score(X_test_ucihar, Y_test_ucihar)
print("Train Score for the UCI-HAR dataset is about: {}".format(train_score))
print("Test Score for the UCI-HAR dataset is about: {}".format(test_score))






#%% TRY ADDING BASIC STATS DERIVED FROM RAW SIGNAL VALUES AS WELL

X4_train = np.concatenate((X1_train, X2_train, X_train_ucihar), axis=1)
X4_test = np.concatenate((X1_test, X2_test, X_test_ucihar), axis=1)
Y4_train = train_labels
Y4_test = test_labels

models = batch_classify(X4_train, Y4_train, X4_test, Y4_test, num_classifiers = 10)
display_dict_models(models)


pca = PCA()  
PCA_train = pca.fit_transform(X4_train)  
PCA_test = pca.transform(X4_test) 
print(pca.explained_variance_ratio_)

models = batch_classify(PCA_train, Y4_train, PCA_test, Y4_test, num_classifiers = 10)
display_dict_models(models)