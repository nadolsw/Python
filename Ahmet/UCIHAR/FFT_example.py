# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:05:11 2019

@author: nadolsw

Created based on the following tutorial: http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
"""

#%% IMPORT NECESSARY PACKAGES

#pip install siml
#pip install pandas
#pip install seaborn

import numpy as np
import matplotlib.pyplot as plt

from siml import *
from collections import defaultdict, Counter
from mpl_toolkits.mplot3d import Axes3D
from siml.sk_utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.signal import welch
from scipy.fftpack import fft
from scipy import signal
from scipy.fftpack import fft

#DEFINE UTILITY FUNCTIONS
def get_fft_values(y_values, nsamples, samp_freq):
    f_values = np.linspace(0.0, samp_freq/2, nsamples//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/nsamples * np.abs(fft_values_[0:nsamples//2])
    return f_values, fft_values

def get_psd_values(y_values, nsamples, samp_freq):
    f_values, psd_values = welch(y_values, fs=samp_freq)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, samp_period, nsamples, samp_freq):
    autocorr_values = autocorr(y_values)
    x_values = np.array([samp_period * jj for jj in range(0, nsamples+1)])
    return x_values, autocorr_values

#%% BASIC DEMONSTRATION OF FAST FOURIER TRANSFORM IN PYTHON

#Specify signal parameters
samp_duration = 10 #Total length of each signal (in seconds)
nsamples = 1000 #Total number of samples per signal
samp_period = samp_duration/nsamples #Uniform time increment between samples (sampling period in seconds)
samp_freq = 1/samp_period #Sampling frequency (number of samples per second)

x_value = np.linspace(0,samp_duration,nsamples+1)
amplitudes = [4, 6, 8, 10, 14]
frequencies = [6.5, 5, 3, 1.5, 1]

#Explicitly create input signals used to construct composite signal
y0 = [amplitudes[0]*np.sin(2*np.pi*frequencies[0]*x_value)]
y1 = [amplitudes[1]*np.sin(2*np.pi*frequencies[1]*x_value)]
y2 = [amplitudes[2]*np.sin(2*np.pi*frequencies[2]*x_value)]
y3 = [amplitudes[3]*np.sin(2*np.pi*frequencies[3]*x_value)]
y4 = [amplitudes[4]*np.sin(2*np.pi*frequencies[4]*x_value)]

y0_array = np.transpose(np.asarray(y0))
y1_array = np.transpose(np.asarray(y1))
y2_array = np.transpose(np.asarray(y2))
y3_array = np.transpose(np.asarray(y3))
y4_array = np.transpose(np.asarray(y4))

#Plot each simulated signal
def plot_fig(y,n,a,f,c):
    plt.figure(figsize=(12,4))
    plt.plot(x_value, y, linestyle='-', color=c)
    plt.title('Signal {}'.format(n) + ': Amplitude={}'.format(a) + ' Frequency={}'.format(f), fontsize=14)
    plt.xlabel('Time [Seconds]', fontsize=10)
    plt.ylabel('Signal Value', fontsize=10)
    plt.show()
    
plot_fig(y0_array,0,4,6.5,'blue')
plot_fig(y1_array,1,6,5,'red')
plot_fig(y2_array,2,8,3,'green')
plot_fig(y3_array,3,10,1.5,'orange')
plot_fig(y4_array,4,14,1,'purple')

#SIMULATE A COMPOSITE SIGNAL COMPOSED OF MULTIPLE SINE WAVES
y_values = [amplitudes[ii]*np.sin(2*np.pi*frequencies[ii]*x_value) for ii in range(0,len(amplitudes))]
composite_signal = np.sum(y_values, axis=0)
composite_array = np.transpose(np.asarray(composite_signal))
composite_matrix = np.column_stack((x_value, composite_array))
df_composite = pd.DataFrame({'Time':composite_matrix[:,0],'Signal Value':composite_matrix[:,1]})

plt.figure(figsize=(12,4))
plt.plot(x_value, composite_signal, linestyle='-', color='black')
plt.title("Composite Signal", fontsize=12)
plt.xlabel('Signal Value', fontsize=12)
plt.ylabel('Time [Seconds]', fontsize=12)
plt.show()

#PLOT FFT OUTPUT
f_values, fft_values = get_fft_values(composite_signal, nsamples, samp_freq)
FFT_matrix = np.column_stack((f_values, fft_values))
df_FFT = pd.DataFrame({'Frequency':FFT_matrix[:,0],'Amplitude':FFT_matrix[:,1]})

plt.figure(figsize=(12,4))
plt.plot(f_values, fft_values, linestyle='-', color='black')
plt.title("FFT: Frequency domain of the composite signal", fontsize=12)
plt.xlabel('Frequency [Hz]', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.xticks(frequencies)
plt.xlim([0,8])
plt.show()

#EXTRACT AND PLOT POWER SPECTRAL DENSITY VALUES
f_values, psd_values = get_psd_values(composite_signal, nsamples, samp_freq)
PSD_matrix = np.column_stack((f_values, psd_values))
df_PSD = pd.DataFrame({'Frequency':PSD_matrix[:,0],'Amplitude':PSD_matrix[:,1]})
 
plt.figure(figsize=(12,4))
plt.plot(f_values, psd_values, linestyle='-', color='black')
plt.title("FFT: Power Spectral Density of the composite signal", fontsize=12)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2 / Hz]')
plt.xticks(frequencies)
plt.xlim([0,8])
plt.show()

#COMPUTE AUTOCORRELATION VALUES OF SIGNAL
t_values, autocorr_values = get_autocorr_values(composite_signal, samp_period, nsamples, samp_freq)
autocorr_matrix = np.column_stack((t_values, autocorr_values))
df_autocorr = pd.DataFrame({'Time':autocorr_matrix[:,0],'Amplitude':autocorr_matrix[:,1]})

plt.figure(figsize=(12,4))
plt.plot(t_values, autocorr_values, linestyle='-', color='black')
plt.title("Autocorrelation of the composite signal", fontsize=12)
plt.xlabel('Time Delay [Seconds]')
plt.ylabel('Autocorrelation Amplitude')
plt.show()
