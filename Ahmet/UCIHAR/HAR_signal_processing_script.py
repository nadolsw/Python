# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:33:34 2019

@author: nadolsw

Created based on the following tutorial: http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
"""

#%% IMPORT NECESSARY PACKAGES

#pip install siml
#pip install pandas
#pip install seaborn
#pip install sklearn
#pip install scipy
#pip install xgboost

import os
import time
import numpy as np
import pandas as pd
import scipy as scipy
import scipy.io as sio
import matplotlib.pyplot as plt

from siml import *
from sklearn import *
from scipy import *
from scipy.stats import mode
from collections import defaultdict, Counter
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier

#DEFINE UTILITY FUNCTIONS

def read_signals(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels(filename):        
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [sample_rate * kk for kk in range(0,len(y_values))]
    return x_values, y_values

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, f_s/2, N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def get_psd_values(y_values, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

#%% IMPORT DATASETS

INPUT_FOLDER_TRAIN = 'C:/Users/nadolsw/Desktop/Tech/Data Science/Python/Ahmet/signal/UCI HAR Dataset/train/Inertial Signals/'
INPUT_FOLDER_TEST = 'C:/Users/nadolsw/Desktop/Tech/Data Science/Python/Ahmet/signal/UCI HAR Dataset/test/Inertial Signals/'

LABELFILE_TRAIN = 'C:/Users/nadolsw/Desktop/Tech/Data Science/Python/Ahmet/signal/UCI HAR Dataset/train/y_train.txt'
LABELFILE_TEST = 'C:/Users/nadolsw/Desktop/Tech/Data Science/Python/Ahmet/signal/UCI HAR Dataset/test/y_test.txt'

INPUT_FILES_TRAIN = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt', 
                     'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                     'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

INPUT_FILES_TEST = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt', 
                     'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                     'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

activities_description = {
    1: 'walking',
    2: 'walking upstairs',
    3: 'walking downstairs',
    4: 'sitting',
    5: 'standing',
    6: 'laying'
}

train_signals, test_signals = [], []

for input_file in INPUT_FILES_TRAIN:
    signal = read_signals(INPUT_FOLDER_TRAIN + input_file)
    train_signals.append(signal)
train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

for input_file in INPUT_FILES_TEST:
    signal = read_signals(INPUT_FOLDER_TEST + input_file)
    test_signals.append(signal)
test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

train_labels = read_labels(LABELFILE_TRAIN)
test_labels = read_labels(LABELFILE_TEST)

[no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
[no_signals_test, no_steps_test, no_components_test] = np.shape(train_signals)
no_labels = len(np.unique(train_labels[:])) #number of labels (activities)

print("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train, no_steps_train, no_components_train))
print("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test, no_steps_test, no_components_test))

#print("The train dataset contains {} labels, with the following distribution:\n {}".format(np.shape(train_labels)[0], Counter(train_labels[:])))
#print("The test dataset contains {} labels, with the following distribution:\n {}".format(np.shape(test_labels)[0], Counter(test_labels[:])))

#train_signals, train_labels = randomize(train_signals, np.array(train_labels))
#test_signals, test_labels = randomize(test_signals, np.array(test_labels))

#%% PLOT SENSOR READINGS FOR CHOSEN SIGNAL #

#Data consists of 10,000 recordings (7352 training and 2947 test cases) of 6 different activities (labels)
#Each recording is composed of 9 different smartphone sensors capturing 128 samples over 2.56 seconds (sampling rate of 50hz)

N = 128
f_s = 50
t_n = 2.56
T = t_n / N
sample_rate = 1 / f_s
denominator = 10

ylabel = 'Sensor Value' 
xlabel = 'Time [Seconds]'
colors = ['r', 'g', 'b']
labels = ['x-component', 'y-component', 'z-component']
axtitles = [ ['Acceleration', 'Gyroscope', 'Total Acceleration'] ]

signal_no = 0 #Record number index (from 0 to 7351)
signals = train_signals[signal_no, :, :]
label = train_labels[signal_no]
activity_name = activities_description[label]

f, axarr = plt.subplots(nrows=3, ncols=1, figsize=(12,10))
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


#%%EXTRACT COMMON TIME SERIES FEATURES

def calculate_basic_stats(list_values):
    min_ = np.nanmin(list_values)
    median = np.nanpercentile(list_values, 50)
    max_ = np.nanmax(list_values)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    return [min_, median, max_, mean, std]

def get_uci_har_stats(dataset, labels):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, : , signal_comp]
            features += calculate_basic_stats(signal)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y

X1_train, Y1_train = get_uci_har_stats(train_signals, train_labels)
X1_test, Y1_test = get_uci_har_stats(test_signals, test_labels)

#%% PERFORM CLASSIFICATION IN BATCH USING FOLLOWING METHODS

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

#%% TRAIN MODELS USING TRADITIONAL STATISTICS
    
models = batch_classify(X1_train, Y1_train, X1_test, Y1_test, num_classifiers = 10)
display_dict_models(models)

#%% ADD SOME ADDITIONAL STATS (RMS, ZCR, PCTILES, ENTROPY)

def calculate_dist_stats(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    skew = scipy.stats.skew(list_values)
    kurt = scipy.stats.kurtosis(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, skew, kurt, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    num_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    num_mean_crossings = len(mean_crossing_indices)
    return [num_zero_crossings, num_mean_crossings]

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def get_features(list_values):
    statistics = calculate_dist_stats(list_values)
    crossings = calculate_crossings(list_values)
    entropy = calculate_entropy(list_values)
    return statistics + crossings + [entropy]

def get_uci_har_features(dataset, labels):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, : , signal_comp]
            features += get_features(signal)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y

X2_train, Y2_train = get_uci_har_features(train_signals, train_labels)
X2_test, Y2_test = get_uci_har_features(test_signals, test_labels)

#%% TRAIN MODELS USING ADDITIONAL STATISTICS
    
models = batch_classify(X2_train, Y2_train, X2_test, Y2_test, num_classifiers = 10)
display_dict_models(models)

#%%COMPUTE FFT, PSD, AUTOCORR FOR CHOSEN SIGNAL

ylabel = 'Amplitude'
xlabels = ['Time [Seconds]', 'Frequency [Hz]', 'Frequency [Hz]', 'Time Lag [Seconds]']
axtitles = [
            ['Acceleration', 'Gyro', 'Total Acceleration'],
            ['FFT Acc', 'FFT Gyro', 'FFT Tot Acc'],
            ['PSD Acc', 'PSD Gyro', 'PSD Tot Acc'],
            ['Autocorr Acc', 'Autocorr Gyro', 'Autocorr Tot Acc']
           ]
 
list_functions = [get_values, get_fft_values, get_psd_values, get_autocorr_values]
 
signal_no = 0
signals = train_signals[signal_no, : , :]
label = train_labels[signal_no]
activity_name = activities_description[label]
 
f, axarr = plt.subplots(nrows=4, ncols=3, figsize=(12,12))
f.suptitle(suptitle.format(activity_name), fontsize=16)
for row_no in range(0,4):
    for comp_no in range(0,9):
        col_no = comp_no // 3
        plot_no = comp_no % 3
        color = colors[plot_no]
        label = labels[plot_no]
 
        axtitle  = axtitles[row_no][col_no]
        xlabel = xlabels[row_no]
        value_retriever = list_functions[row_no]
 
        ax = axarr[row_no, col_no]
        ax.set_title(axtitle, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        if col_no == 0:
            ax.set_ylabel(ylabel, fontsize=12)
 
        signal_component = signals[:, comp_no]
        x_values, y_values = value_retriever(signal_component, T, N, f_s)
        ax.plot(x_values, y_values, linestyle='-', color=color, label=label)
        if row_no > 0:
            max_peak_height = 0.1 * np.nanmax(y_values)
            indices_peaks = detect_peaks(y_values, mph=max_peak_height)
            ax.scatter(x_values[indices_peaks], y_values[indices_peaks], c=color, marker='*', s=60)
        if col_no == 2:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.6)
plt.show()

#%% EXTRACT FFT & AUTOCORR FEATURES (PEAKS) FOR ALL SIGNALS - CURRENTLY EXTRACTING TOP 3 X & Y VALS FOR EACH

X3_train, Y3_train = extract_features_labels(train_signals, train_labels, T, N, f_s, denominator)
X3_test, Y3_test = extract_features_labels(test_signals, test_labels, T, N, f_s, denominator)

#%% TRAIN MODELS USING ONLY DFT & AUTOCORR FEATURES
    
models = batch_classify(X3_train, Y3_train, X3_test, Y3_test, num_classifiers = 10)
display_dict_models(models)

#%% COMBINE FEATURES FROM ALL METHODS

#Assumes labels/records have not been randomized across datasets
X4_train = np.concatenate((X1_train, X2_train, X3_train), axis=1)
X4_test = np.concatenate((X1_test, X2_test, X3_test), axis=1)
Y4_train = train_labels
Y4_test = test_labels

models = batch_classify(X4_train, Y4_train, X4_test, Y4_test, num_classifiers = 10)
display_dict_models(models)

#%% OPTIMIZE HYPERPARAMETERS FOR CHAMPION MODEL

GDB_params = {
    'n_estimators': [50, 100, 250, 500, 1000],
    'learning_rate': [0.5, 0.25, 0.1, 0.05, 0.01],
    'criterion': ['friedman_mse', 'mse']
}

for n_est in GDB_params['n_estimators']:
    for lr in GDB_params['learning_rate']:
        for crit in GDB_params['criterion']:
            clf = GradientBoostingClassifier(n_estimators=n_est, learning_rate = lr, criterion = crit)
            clf.fit(X4_train, Y4_train)
            train_score = clf.score(X4_train, Y4_train)
            test_score = clf.score(X4_test, Y4_test)
            print("For ({}, {}, {}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(
                n_est, lr, crit[:4], train_score, test_score)
                 )