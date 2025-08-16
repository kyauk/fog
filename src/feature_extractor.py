# Goal is to extract key features from processed data
"""Key Features:

    1. Freeze Index which identifies if a patient is undergoing a freeze episode or not, but does not distinguish activity vs non-activity
    2. Energy Levels which identiifes if a patient is exerting energy or not, to recognize if a patient truly is in a freeze episode vs just sitting
    3. Variance in signals to see if a patient has an irregular cadence/shuffling to realize a freeze episode may be occuring
    4. Skewness to determine if there is a gait irregularity
    5. Spectral centroid to identify if there is trembling, vs if they're actually walking"""
from scipy import signal, stats
import numpy as np
import pandas as pd

class FeatureExtractor:
    def __init__(self, sampling_rate=64, expected_window_size = 4.0):
        self.sampling_rate = sampling_rate
        self.expected_window_size = expected_window_size * self.sampling_rate
    def calculate_freq_features(self, window):
        # if inputted window less than 4 seconds
        if len(window) != self.expected_window_size:
            raise ValueError(f"Expected {self.expected_window_size} samples, got {len(window)}")
        f,psd = signal.welch(window, fs=self.sampling_rate)
        # Freeze power is TOTAL energy Between [3,8]
        freeze_power = psd[(f >=3) & (f <= 8)].sum()
        # Locomotion power is TOTAL energy Between [0.5,3]
        loco_power = psd[(f >= 0.5) & (f <= 3)].sum()
        freeze_index =  freeze_power/loco_power if loco_power > 0 else 0
        spectral_centroid = np.sum(f * psd) / np.sum(psd)
        return freeze_index, spectral_centroid
    
    def calculate_energy(self, window):
        return np.sum(window**2)
    
    def calculate_var(self, window):
        return np.var(window)
    
    def calculate_skew(self,window):
        return stats.skew(window)
    
    def extract_features(self, windows):
        features_matrix = []
        for window in windows:
            freeze_index, spectral_centroid = self.calculate_freq_features(window)
            energy = self.calculate_energy(window)
            var = self.calculate_var(window)
            skew = self.calculate_skew(window)
            features_matrix.append([freeze_index, energy, var, skew, spectral_centroid])
        return np.array(features_matrix)