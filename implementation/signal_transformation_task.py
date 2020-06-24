# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 06:48:19 2019

@author: Pritam
"""

import numpy as np
import math
import cv2


def add_noise(signal, noise_amount):
    """ 
    adding noise
    """
    noise = np.random.normal(1, noise_amount, np.shape(signal)[0])
    noised_signal = signal+noise
    return noised_signal
    
def add_noise_with_SNR(signal, noise_amount):
    """ 
    adding noise
    created using: https://stackoverflow.com/a/53688043/10700812 
    """
    
    target_snr_db = noise_amount #20
    x_watts = signal ** 2                       # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)   # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))     # Generate an sample of white noise
    noised_signal = signal + noise_volts        # noise added signal

    return noised_signal 

def scaled(signal, factor):
    """"
    scale the signal
    """
    scaled_signal = signal * factor
    return scaled_signal
 

def negate(signal):
    """ 
    negate the signal 
    """
    negated_signal = signal * (-1)
    return negated_signal

    
def hor_filp(signal):
    """ 
    flipped horizontally 
    """
    hor_flipped = np.flip(signal)
    return hor_flipped


def permute(signal, pieces):
    """ 
    signal: numpy array (batch x window)
    pieces: number of segments along time    
    """
    pieces       = int(np.ceil(np.shape(signal)[0]/(np.shape(signal)[0]//pieces)).tolist())
    piece_length = int(np.shape(signal)[0]//pieces)
    
    sequence = list(range(0,pieces))
    np.random.shuffle(sequence)
    
    permuted_signal = np.reshape(signal[:(np.shape(signal)[0]//pieces*pieces)], (pieces, piece_length)).tolist() + [signal[(np.shape(signal)[0]//pieces*pieces):]]
    permuted_signal = np.asarray(permuted_signal)[sequence]
    permuted_signal = np.hstack(permuted_signal)
        
    return permuted_signal

     
    
def time_warp(signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
    """ 
    signal: numpy array (batch x window)
    sampling freq
    pieces: number of segments along time
    stretch factor
    squeeze factor
    """
    
    total_time = np.shape(signal)[0]//sampling_freq
    segment_time = total_time/pieces
    sequence = list(range(0,pieces))
    stretch = np.random.choice(sequence, math.ceil(len(sequence)/2), replace = False)
    squeeze = list(set(sequence).difference(set(stretch)))
    initialize = True
    for i in sequence:
        orig_signal = signal[int(i*np.floor(segment_time*sampling_freq)):int((i+1)*np.floor(segment_time*sampling_freq))]
        orig_signal = orig_signal.reshape(np.shape(orig_signal)[0],1)
        if i in stretch:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*stretch_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
        elif i in squeeze:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*squeeze_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
    return time_warped
   

