# Goal is to preprocess daphnet data 
from scipy.signal import butter, filtfilt
import numpy as np
# Overlapping of 0.5s (0-4, 0.5-4.5, 1-5)
# Defaults are cutoff after 20 hz, have a sampling rate of 64, window size of 4, and window overlap of 0.5
class DataPreprocessor:
    def __init__(self,cutoff=20, sampling_rate=64, window_size=4.0, window_overlap=0.5):
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.window_overlap = window_overlap
        
        
    def apply_butter(self, data):
        b, a = butter(N=4, Wn=self.cutoff, fs=self.sampling_rate, btype='low')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    
    def create_windows(self,data):
        # window iteration of 265 samples (4s * 64 hz) and overlap iter of 32 samples (0.5s * 64hz )
        windows = []
        window_size = int(self.window_size * self.sampling_rate)
        overlap_size = int(self.window_overlap * self.sampling_rate)
        for start in range(0, len(data) - window_size + 1, overlap_size):
            end = start + window_size
            windows.append(data[start:end])
        return np.array(windows)
    
    
    def normalize_windows(self, windows):
        means = windows.mean(axis=1, keepdims=True) 
        stds = windows.std(axis=1, keepdims=True)    
        return (windows - means) / stds
    