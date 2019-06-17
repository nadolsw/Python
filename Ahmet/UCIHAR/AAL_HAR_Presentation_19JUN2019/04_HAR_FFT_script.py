# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:45:17 2019

@author: winado

Created based on the following tutorial: http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""

import time

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

#%% INSTEAD OF BASIC DECRIPTIVE STATS - COMPUTE FFT, PSD, AUTOCORR FOR CHOSEN SIGNAL AND PLOT OUTPUT

ylabel = 'Amplitude'
xlabels = ['Time [Seconds]', 'Frequency [Hz]', 'Frequency [Hz]', 'Time Lag [Seconds]']
colors = ['r', 'g', 'b']
labels = ['x-component', 'y-component', 'z-component']
axtitles = [
            ['Raw Acceleration', 'Raw Gyro', 'Raw Total Acceleration'],
            ['FFT Acc', 'FFT Gyro', 'FFT Tot Acc'],
            ['PSD Acc', 'PSD Gyro', 'PSD Tot Acc'],
            ['Autocorr Acc', 'Autocorr Gyro', 'Autocorr Tot Acc']
           ]

N = 128 #Total number of samples per signal
f_s = 50 #Sampling frequency (number of samples per second)
t_n = 2.56 #Total length of each signal (in seconds)
T = t_n / N #Uniform time increment between samples (sampling period in seconds)
 
list_functions = [get_values, get_fft_values, get_psd_values]
 
signal_no = 50
signals = train_signals[signal_no, : , :]
label = train_labels[signal_no]
activity_name = activities_description[label]
 
print("Training Record Chosen: {}".format(signal_no))
f, axarr = plt.subplots(nrows=3, ncols=3, figsize=(12,10))
suptitle = "All Signal Components for Activity: {}"
f.suptitle(suptitle.format(activity_name), fontsize=16)
for row_no in range(0,3):
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
plt.subplots_adjust(top=0.90, hspace=0.50)
plt.show()
print("Training Record Chosen: {}".format(signal_no))

#%% EXTRACT FFT & PSD FEATURES (PEAKS) FOR ALL SIGNALS - CURRENTLY EXTRACTING TOP 3 X & Y VALS FOR EACH

Xtrain_FFT, Ytrain_FFT = extract_FFT_features(train_signals, train_labels, T, N, f_s)
Xtest_FFT, Ytest_FFT = extract_FFT_features(test_signals, test_labels, T, N, f_s)

print("Output Xtrain dataset has dimensions:")
np.shape(Xtrain_FFT)

#%% TRAIN MODELS USING ONLY FFT & PSD FEATURES
    
models = batch_classify(Xtrain_FFT, Ytrain_FFT, Xtest_FFT, Ytest_FFT, num_classifiers = 10)
display_dict_models(models)

#%% OPTIMIZE HYPERPARAMETERS FOR CHAMPION MODEL (OPTIONAL)

GDB_params = {
    'n_estimators': [50, 100, 250, 500, 1000],
    'learning_rate': [0.5, 0.25, 0.1, 0.05, 0.01],
    'criterion': ['friedman_mse', 'mse']
}

for n_est in GDB_params['n_estimators']:
    for lr in GDB_params['learning_rate']:
        for crit in GDB_params['criterion']:
            clf = GradientBoostingClassifier(n_estimators=n_est, learning_rate = lr, criterion = crit)
            clf.fit(Xtrain_FFT, Ytrain_FFT)
            train_score = clf.score(Xtrain_FFT, Ytrain_FFT)
            test_score = clf.score(Xtest_FFT, Ytest_FFT)
            print("For ({}, {}, {}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(
                n_est, lr, crit[:4], train_score, test_score)
                 )
    
#%% 