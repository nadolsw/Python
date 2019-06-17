# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:34:12 2019

@author: nadolsw

Created based on the following tutorial: http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""
#%% IMPORT NECESSARY PACKAGES

import pywt
import time
import scipy
import matplotlib
import numpy as np
from collections import defaultdict, Counter

#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

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

#%% USE DISCRETE WAVELET TRANSFORM (DWT) TO EXTRACT DETAIL COEFFICIENTS FOR EACH SCALE/SUBBAND

#VISUALIZE WAVELET DECOMPOSITION FOR CHOSEN SIGNAL (MULTIRESOLUTION ANALYSIS)
record_num=100
sig_comp_num=0
num_decomp_lvls=5
wavelet='sym5'

Z = train_signals[record_num,:,sig_comp_num]
print("Training Record Chosen: {} - Signal Component: {}".format(signal_no, sig_comp_num))
plot_DWT_decomp(Z, num_decomp_lvls, wavelet)
print("Training Record Chosen: {} - Signal Component: {}".format(signal_no, sig_comp_num))

#LOOP OVER ALL RECORDS AND SIGNAL COMPONENTS AND EXTRACT DWT FEATURES FOR EACH SCALE OF EACH SIGNAL COMPONENT 
def calculate_statistics(list_values):
    min_ = np.nanmin(list_values)
    max_ = np.nanmax(list_values)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    skew = scipy.stats.skew(list_values)
    kurt = scipy.stats.kurtosis(list_values)
    return [min_, max_, median, mean, std, skew, kurt]

def calc_DWT_features(list_values):
    #entropy = calculate_entropy(list_values)
    #crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return statistics

def extract_DWT_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            list_of_scales = pywt.wavedec(signal, waveletname, level=5) #level= determines number of scales to extract
            for scale in list_of_scales:
                features += calc_DWT_features(scale)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y
#TOTAL OF 10,299 RECORDINGS OF LENGTH 128 OVER 9 SIGNAL COMPONENTS EACH WITH 10 LEVELS OF SCALE DECOMPOSITION
#EXTRACTING A TOTAL OF 7 FEATURES PER SCALE (SEE ASSOCIATED UTIL FILE) - 6 scales for each signal component (level=5)

Xtrain_DWT, Ytrain_DWT = extract_DWT_features(train_signals, train_labels, 'rbio3.1')
Xtest_DWT, Ytest_DWT = extract_DWT_features(test_signals, test_labels, 'rbio3.1')

#Expecting a total of: 9 x 6 x 7 = 378 features for each record
print("Output Xtrain dataset has dimensions:")
np.shape(Xtrain_DWT)

#%% TRAIN MODELS USING DWT-DERIVED STATISTICS
    
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

models = batch_classify(Xtrain_DWT, Ytrain_DWT, Xtest_DWT, Ytest_DWT, num_classifiers = 10)
display_dict_models(models)
