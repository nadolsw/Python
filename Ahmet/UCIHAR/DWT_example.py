# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:35:07 2019

@author: nadolsw

Created based on the following tutorial: http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
"""

#%% IMPORT NECESSARY PACKAGES

#pip install pywt
#pip install pywavelets
#pip install scaleogram

import pywt
import pywt.data
import numpy as np
import pandas as pd
import scaleogram as scg
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#%% SIMULATE TWO SIGNALS - A PERIODIC AND CHIRP SIGNAL

#Specify signal parameters
samp_duration = 1 #Total length of each signal (in seconds)
nsamples = 100000 #Total number of samples per signal
samp_period = samp_duration / nsamples #Uniform time increment between samples (sampling period in seconds)
samp_freq = 1/samp_period #Sampling frequency (number of samples per second)

frequencies = [4, 30, 60, 90] 
xa = np.linspace(0, samp_duration, num=nsamples)
xb = np.linspace(0, samp_duration/4, num=nsamples/4)

y1a, y1b = np.sin(2*np.pi*frequencies[0]*xa), np.sin(2*np.pi*frequencies[0]*xb)
y2a, y2b = np.sin(2*np.pi*frequencies[1]*xa), np.sin(2*np.pi*frequencies[1]*xb)
y3a, y3b = np.sin(2*np.pi*frequencies[2]*xa), np.sin(2*np.pi*frequencies[2]*xb)
y4a, y4b = np.sin(2*np.pi*frequencies[3]*xa), np.sin(2*np.pi*frequencies[3]*xb)
 
periodic_signal = y1a + y2a + y3a + y4a
orig_chirp_signal = np.concatenate([y1b, y2b, y3b, y4b])

def get_fft_values_modified(y_values, samp_period, nsamples, samp_freq):
    f_values = np.linspace(0.0, 1.0/(2.0*samp_period), nsamples//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/nsamples * np.abs(fft_values_[0:nsamples//2])
    return f_values, fft_values

#GET FFT COEFS FOR EACH SIGNAL
f_values1, fft_values1 = get_fft_values_modified(periodic_signal, samp_period, nsamples, samp_freq)
f_values2, fft_values2 = get_fft_values_modified(orig_chirp_signal, samp_period, nsamples, samp_freq)

#PLOT EACH SIGNAL 
fig, axarr = plt.subplots(nrows=2, figsize=(12,8))
axarr[0].title.set_text('Periodic Signal')
axarr[0].plot(xa, periodic_signal)
axarr[1].title.set_text('Chirp Signal')
axarr[1].plot(xa, orig_chirp_signal)
for ax in axarr.flat:
    ax.set(xlabel='Time [Seconds]', ylabel='Signal Value')
plt.subplots_adjust(top=1, hspace=0.3)
plt.show()

#PLOT FFT COEFFICIENTS
fig, axarr = plt.subplots(nrows=2, figsize=(12,8))
axarr[0].title.set_text('Periodic Signal')
axarr[0].plot(f_values1, fft_values1)
axarr[1].title.set_text('Chirp Signal')
axarr[1].plot(f_values2, fft_values2)
for ax in axarr.flat:
    ax.set(xlabel='Frequency [Hz]', ylabel='Amplitude')
    ax.set_xlim([0,120])
plt.subplots_adjust(top=1, hspace=0.3)
plt.show()

#%% PRINT MOTHER WAVELETS & WAVELET FAMILY INFO

wavelet_families = pywt.families(short=False)
discrete_mother_wavelets = pywt.wavelist(kind='discrete')
continuous_mother_wavelets = pywt.wavelist(kind='continuous')

print("PyWavelets contains the following families: ")
print(wavelet_families)
print()
print("PyWavelets contains the following Continuous families: ")
print(continuous_mother_wavelets)
print()
print("PyWavelets contains the following Discrete families: ")
print(discrete_mother_wavelets)
print()
for family in pywt.families():
    print("    * The {} family contains: {}".format(family, pywt.wavelist(family)))
    
#Visualize a number of the wavelets
    
discrete_wavelets = ['db5', 'sym5', 'coif5']
continuous_wavelets = ['mexh', 'morl', 'cgau5']
list_list_wavelets = [discrete_wavelets, continuous_wavelets]
list_funcs = [pywt.Wavelet, pywt.ContinuousWavelet]

fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
for ii, list_wavelets in enumerate(list_list_wavelets):
    func = list_funcs[ii]
    row_no = ii
    for col_no, waveletname in enumerate(list_wavelets):
        wavelet = func(waveletname)
        family_name = wavelet.family_name
        biorthogonal = wavelet.biorthogonal
        orthogonal = wavelet.orthogonal
        symmetry = wavelet.symmetry
        if ii == 0:
            _ = wavelet.wavefun()
            wavelet_function = _[0]
            x_values = _[-1]
        else:
            wavelet_function, x_values = wavelet.wavefun()
        if col_no == 0 and ii == 0:
            axarr[row_no, col_no].set_ylabel("Discrete Wavelets", fontsize=16)
        if col_no == 0 and ii == 1:
            axarr[row_no, col_no].set_ylabel("Continuous Wavelets", fontsize=16)
        axarr[row_no, col_no].set_title("{}".format(family_name), fontsize=16)
        axarr[row_no, col_no].plot(x_values, wavelet_function)
        axarr[row_no, col_no].set_yticks([])
        axarr[row_no, col_no].set_yticklabels([])
plt.tight_layout()
plt.show()

#%% EXAMINE WAVELET DECOMPOSITION FUNCTIONALITY

#PLOT WAVELET DECOMPOSITION
x = np.linspace(0, 1, num=2048)
new_chirp_signal = np.sin(250 * np.pi * x**2) 

def plot_wvlt_decomp(data, dset_name, num_levels, waveletname): 
    fig, ax = plt.subplots(figsize=(12,2))
    ax.set_title("Input: {}".format(dset_name))
    ax.plot(data)
    plt.show()

    fig, axarr = plt.subplots(nrows=num_levels, ncols=2, figsize=(10,12))
    for ii in range(num_levels):
        (data, coeff_d) = pywt.dwt(data, waveletname)
        axarr[ii, 0].plot(data, 'r')
        axarr[ii, 1].plot(coeff_d, 'g')
        axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
        axarr[ii, 0].set_yticklabels([])
        if ii == 0:
            axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
            axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
        axarr[ii, 1].set_yticklabels([])
    plt.tight_layout()
    plt.show()

plot_wvlt_decomp(new_chirp_signal, 'Exponential Chirp Signal', 5, 'sym5')
plot_wvlt_decomp(periodic_signal, 'Periodic Signal', 12, 'sym5')
plot_wvlt_decomp(orig_chirp_signal, 'Original Chirp Signal', 12, 'sym5')

#Determine maximum useful level of decomposition
w = pywt.Wavelet('sym5')
m = pywt.dwt_max_level(data_len=100000, filter_len=w.dec_len)
print(m)

#%% USING PYWT.DWT on ECG SAMPLE DATA TO RECONSTRUCT A SIGNAL

#Plot Original ECG Signal
x = pywt.data.ecg()
plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('ECG Signal')
plt.plot(x)

#Perform Wavelet Decomposition
w = pywt.Wavelet('sym5')
plt.plot(w.dec_lo)
coeffs = pywt.wavedec(x, w, level=6)

def reconstruction_plot(coeffs):
    plt.plot(np.linspace(0, 1, len(coeffs)), coeffs, alpha=0.75)
    
#Compare Level 0 Reconstruction to Full Reconstruction
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure(figsize=(12,3))
plt.title('ECG Reconstruction from DWT Decomposition')
reconstruction_plot(pywt.waverec(coeffs, w)) # full reconstruction 
reconstruction_plot(pywt.waverec(coeffs[:-6] + [None] * 6, w)) # leaving out all detail coefficients = reconstruction using lvl1 approximation only
plt.legend(['Full Reconstruction', 'Using Only Level 1 Approximation'])

#Compare Level 0-2 Reconstructions
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure(figsize=(12,3))
plt.title('ECG Reconstruction from DWT Decomposition')
reconstruction_plot(pywt.waverec(coeffs[:-6] + [None] * 6, w)) # leaving out all detail coefficients
reconstruction_plot(pywt.waverec(coeffs[:-5] + [None] * 5, w)) # leaving out detail coefficients up to lvl 1
reconstruction_plot(pywt.waverec(coeffs[:-4] + [None] * 4, w)) # leaving out detail coefficients up to lvl 2
plt.legend(['Level 0 Approx', 'Level 1 Approx', 'Level 2 Approx'])

#Plot all approximations for comparison
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure(figsize=(14,6))
plt.title('ECG Reconstruction from DWT Decomposition')
reconstruction_plot(pywt.waverec(coeffs, w)) # full reconstruction 
reconstruction_plot(pywt.waverec(coeffs[:-6] + [None] * 6, w)) # leaving out all detail coefficients
reconstruction_plot(pywt.waverec(coeffs[:-5] + [None] * 5, w)) # leaving out detail coefficients up to lvl 1
reconstruction_plot(pywt.waverec(coeffs[:-4] + [None] * 4, w)) # leaving out detail coefficients up to lvl 2
reconstruction_plot(pywt.waverec(coeffs[:-3] + [None] * 3, w)) # leaving out detail coefficients up to lvl 3
reconstruction_plot(pywt.waverec(coeffs[:-2] + [None] * 2, w)) # leaving out detail coefficients up to lvl 4
reconstruction_plot(pywt.waverec(coeffs[:-1] + [None] * 1, w)) # leaving out detail coefficients up to lvl 5
plt.legend(['Full reconstruction','Level 0', 'Level 1','Level 2','Level 3','Level 4','Level 5','Level 6'])


#%% USE PYWT TO DENOISE SIGNAL (NASA FEMTO BEARING DATA)

#Data: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto

DATA_FOLDER = 'C:/Users/nadolsw/Desktop/Tech/Data Science/Python/Ahmet/wavelet/Bearing1_1/'
filename = 'acc_01210.csv'
df = pd.read_csv(DATA_FOLDER + filename, header=None)
bearing_signal = df[4].values

#Plot raw signal
x = np.linspace(0, 2560, num=2560)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, bearing_signal, color="b", alpha=0.5, label='original signal')
ax.legend()
ax.set_title('Noisy Bearing Signal', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample Number', fontsize=16)
plt.show()

def lowpassfilter(signal, thresh = 0.50, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

#Plot denoised signal using three different thresholds
smoothed_signal = lowpassfilter(bearing_signal, 0.25)
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(bearing_signal, color="b", alpha=0.5, label='original signal')
ax.plot(smoothed_signal, 'k', label='DWT smoothing', linewidth=2)
ax.legend()
ax.set_title('Removing High Frequency Noise with DWT (thresh=0.25)', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample No', fontsize=16)
plt.show()

smoothed_signal = lowpassfilter(bearing_signal, 0.50)
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(bearing_signal, color="b", alpha=0.5, label='original signal')
ax.plot(smoothed_signal, 'k', label='DWT smoothing', linewidth=2)
ax.legend()
ax.set_title('Removing High Frequency Noise with DWT (thresh=0.50)', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample No', fontsize=16)
plt.show()

smoothed_signal = lowpassfilter(bearing_signal, 0.75)
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(bearing_signal, color="b", alpha=0.5, label='original signal')
ax.plot(smoothed_signal, 'k', label='DWT smoothing', linewidth=2)
ax.legend()
ax.set_title('Removing High Frequency Noise with DWT (thresh=0.75)', fontsize=18)
ax.set_ylabel('Signal Amplitude', fontsize=16)
ax.set_xlabel('Sample No', fontsize=16)
plt.show()

#%% EXAMINE SCALEOGRAM FUNCTIONALITY ON EL NINO DATASET

dataset = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
df_nino = pd.read_csv(dataset, header=None)

N = df_nino.shape[0]
t0=1871 #starting year of the dataset
dt=0.25 #sampling period (quarterly)
fs = 1/dt #sampling rate (per year)
time = np.arange(0, N) * dt + t0
elnino = df_nino.values.squeeze()

#How many scales of interest?
w = pywt.Wavelet('sym5')
m = pywt.dwt_max_level(data_len=504, filter_len=w.dec_len)
print(m)

#Plot Raw Data
plt.figure(figsize=(12,6))
plt.plot(time, elnino, linestyle='-', color='blue')
plt.title("El Nino Weather Data", fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Standardized Sea-Surface Temperature', fontsize=12)
plt.show()

#Perform Simple Moving Average to Smooth Data
def get_avg_values(xvalues, yvalues, w):
    signal_length = len(xvalues)
    if signal_length % w == 0:
        padding_length = 0
    else:
        padding_length = w - signal_length//w % w
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//w, w)
    yarr.resize(signal_length//w, w)
    xarr_reshaped = xarr.reshape((-1,w))
    yarr_reshaped = yarr.reshape((-1,w))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def plot_signal_plus_avg(ax, time, signal, window):
    time_ave, signal_avg = get_avg_values(time, signal, window)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_avg, label = 'time average (w={})'.format(window))
    ax.set_xlim([time[0], time[-1]])
    ax.set_title('El Nino: Raw Signal & Moving Average', fontsize=16)
    ax.set_ylabel('Standardized Sea-Surface Temperature', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.legend(loc='best')

#Plot Raw Signal & Smoothed Values
fig, ax = plt.subplots(figsize=(12,6))
plot_signal_plus_avg(ax, time, elnino, window = 5)
plt.show()

#Perform FFT & Extract Resulting Coefs
def get_fft_values(y_values, T, N, f_s):
    N2 = 2 ** (int(np.log2(N)) + 1) # round up to next highest power of 2
    f_values = np.linspace(0.0, 1.0/(2.0*T), N2//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N2 * np.abs(fft_values_[0:N2//2])
    return f_values, fft_values

def plot_fft_plus_power(ax, time, signal, plot_direction='horizontal', yticks=None, ylim=None, f_or_p='f'):  
    variance = np.std(signal)**2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    p_values = 1/f_values
    fft_power = variance * abs(fft_values) ** 2
    ax.set_ylabel('Amplitude', fontsize=12)
    if plot_direction == 'horizontal':
        if f_or_p == 'f':
            ax.set_title('El Nino: DFT & PSD Values: By Frequency', fontsize=16)
            ax.set_xlabel('Frequency [Cycles/Year]', fontsize=12)
            ax.plot(f_values, fft_values, 'r-', label='Fourier Transform')
            ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
        elif f_or_p == 'p':
            ax.set_title('El Nino: DFT & PSD Values: By Period', fontsize=16)
            ax.set_xlabel('Periodicity [Years/Cycle]', fontsize=12)
            ax.plot(p_values, fft_values, 'r-', label='Fourier Transform')
            ax.plot(p_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
            ax.set_xlim(0, 25)            
    elif plot_direction == 'vertical':
        scales = 1./f_values
        scales_log = np.log2(scales)
        ax.plot(fft_values, scales_log, 'r-', label='Fourier Transform')
        ax.plot(fft_power, scales_log, 'k--', linewidth=1, label='FFT Power Spectrum')
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ax.set_ylim(ylim[0], -1)
    ax.legend()

#Plot FFT & PSD Coefs (Frequency Domain)
fig, ax = plt.subplots(figsize=(12,6))
plot_fft_plus_power(ax=ax, time=time, signal=elnino, f_or_p='f')
#Plot FFT & PSD Coefs (Period Domain)
fig, ax = plt.subplots(figsize=(12,6))
plot_fft_plus_power(ax=ax, time=time, signal=elnino, f_or_p='p')

#CREATE SPECTROGRAM OF EL NINO DATA
def plot_spectrogram(ax, time, signal, waveletname = 'cmor', cmap = plt.cm.seismic):
    #dt = time[1] - time[0]
    scales = np.arange(1, 128)
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    ax.set_title('Spectrogram Representation of Amplitude for El Nino Data', fontsize=20)
    ax.set_ylabel('Period (Years)', fontsize=16)
    ax.set_xlabel('Time', fontsize=16)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()
    
#PLOT SPECTROGRAM
fig, ax = plt.subplots(figsize=(10, 10))
plot_spectrogram(ax=ax, time=time, signal=elnino)
plt.show()

#PLOT HIGHER RESOLUTION SCALEOGRAM USING SCALEOGRAM PACKAGE
scales = np.logspace(1, 2.4, num=200, dtype=np.int32)
#scales = np.arange(4,400,8) #For a coarser decomp (adjust last param to change resolution)
ax = scg.cws(time, elnino, scales, figsize=(14,7), ylabel="Period [Years]", xlabel='Year', yscale='log')
ticks = ax.set_yticks([2,4,8, 16,32])
ticks = ax.set_yticklabels([2,4,8, 16,32])


