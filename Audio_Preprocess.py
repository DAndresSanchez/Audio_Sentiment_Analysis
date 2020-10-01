from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os

# Calculate Energy
def energy_calc(signal: np.array, segment_length: int) -> np.array:
    """
    Calculates energy of the audio segment. Normalised with segment legth.
    """
    energy = []
    for i in range(int(len(signal)/segment_length)):
        segment = signal[i*segment_length:(i+1)*segment_length]# try except error ...
        energy.append(np.sum(np.square(segment)) / segment_length)
        if energy[-1] < 0:
            print(i)
    return energy

# Preprocess signal
def preprocess_signal(filename: str, short_term_length:float=0.020, short_term_overlap:float=0,\
                      medium_term_length:float=1, medium_term_overlap:float=0.020) -> np.array:
    """
    Preprocessing of the audiofile to get 28 coeficients after three steps:
    - Short term analysis: segmentation of audio to get energy and 13 MFCCs per segment. 
    - Medium term analysis: segmentation of audio to get mean and standard deviation per segment.
    - Long term analysis: mean of the medium term values per segment.
    """
    
    # Import audio signal
    sr, signal = read(filename)
    
    # Convert to 8kHz
    sr_objective = 8000
    sr_ratio = int(sr/sr_objective)
    try:
        signal = signal[::sr_ratio,0]
    except IndexError:
        signal = signal[::sr_ratio]
    sr = sr_objective    

    # Normalise
    signal = signal.astype(np.float32)
    signal = signal / np.abs(signal).max() / 2
    
    # Calculate length and define segments
    length = len(signal)
    length_s = length/sr # length of segment in seconds
    short_term_length = 0.020 # s 
    short_term_overlap = 0 # s
    medium_term_length = 1 # s 
    medium_term_overlap = 0.020 # s

    # Convert to samples per segment
    n_fft_st = int(length_s // (short_term_length - short_term_overlap))
    hop_length_st = n_fft_st # no overlap
    segment_length = n_fft_st
    energy = np.array(energy_calc(signal, n_fft_st))
    
    # SHORT TERM ANALYSIS
    # Calculate MFCCs for short term
    mfcc_st = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft_st, n_mfcc=13, hop_length=hop_length_st)
    mfcc_st = mfcc_st[:,:len(energy)]
    coefficients_st = np.vstack((mfcc_st, energy))

    
    # MEDIUM TERM ANALYSIS
    # Calculation of segments length for medium term analysis
    n_segments_mt = int(length_s // (medium_term_length - medium_term_overlap))
    n_fft_mt = int(coefficients_st.shape[1] * medium_term_length / length_s)
    hop_length_mt = int(coefficients_st.shape[1] * (medium_term_length - medium_term_overlap) / length_s)     

    # Calculation of parameters for medium term analysis
    for i in range(n_segments_mt):
        coefficient_i = coefficients_st[:, i*hop_length_mt:i*hop_length_mt+n_fft_mt]
        mean_i = np.mean(coefficient_i, axis=1)
        std_i = np.std(coefficient_i, axis=1)
        if i == 0:
            parameters_mt = np.hstack((mean_i, std_i))
        else:
            parameters_mt = np.row_stack((parameters_mt, np.hstack((mean_i, std_i))))

    # LONG TERM ANALYSIS 
    # Calculation of parameters for long term analysis
    if n_segments_mt > 1:
        parameters_lt = np.mean(parameters_mt, axis=0)
    else: 
        parameters_lt = parameters_mt

    return parameters_lt

# Get labels from directories
def get_label(filename:str) -> str:
    """
    Assign label from directory name.
    """
    label = filename.split("/")[-2]
    return label

# Merge characteristics and labels
def add_label(filename:str) -> np.array:
    """
    Add label to numpy array with 28 characteristics.
    """
    coefficients = preprocess_signal(filename)
    label = np.array(get_label(filename))
    return np.hstack((coefficients, label))

# Merge characteristics and labels from numpy arrays
def add_label_arrays(x:np.array, y:np.array) -> np.array:
    return np.hstack((x, y))
