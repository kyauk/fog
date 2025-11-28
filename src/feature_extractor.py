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
    def __init__(self, sampling_rate=64, expected_window_size=4.0):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Sensor sampling rate (Hz)
            expected_window_size: Expected window duration in seconds (4.0s optimal)
        """
        self.sampling_rate = sampling_rate
        self.expected_window_size = int(expected_window_size * self.sampling_rate)
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
        total_power = np.sum(psd)
        if total_power == 0 or np.isclose(total_power, 0):
            # no meaningful frequency content - assign default value
            spectral_centroid = 0.0  
        else:
            spectral_centroid = np.sum(f * psd) / total_power
        return freeze_index, spectral_centroid
    
    def calculate_energy(self, window):
        return np.sum(window**2)
    
    def calculate_var(self, window):
        return np.var(window)
    
    def calculate_skew(self,window):
        skew = stats.skew(window)
        if not np.isfinite(skew):
            skew = 0.0
        return skew
    
    def calculate_time_domain_features(self, window):
        """Calculate comprehensive time-domain statistical features"""
        # Basic statistics
        min_val = np.min(window)
        max_val = np.max(window)
        percentile_25 = np.percentile(window, 25)
        percentile_75 = np.percentile(window, 75)
        
        # Robust statistics
        mad = np.median(np.abs(window - np.median(window)))  # Median Absolute Deviation
        sma = np.sum(np.abs(window))  # Signal Magnitude Area
        
        # Variation measures
        std = np.std(window)
        mean = np.mean(window)
        cv = std / mean if mean != 0 else 0  # Coefficient of Variation
        
        # Peak detection
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(window)
        n_peaks = len(peaks)
        
        # Crest factor
        rms = np.sqrt(np.mean(window**2))
        crest_factor = max_val / rms if rms > 0 else 0
        
        return min_val, max_val, percentile_25, percentile_75, mad, sma, cv, n_peaks, crest_factor
    
    def calculate_advanced_freq_features(self, window):
        """Calculate advanced frequency-domain features"""
        f, psd = signal.welch(window, fs=self.sampling_rate)
        
        # Spectral entropy (measure of signal randomness)
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Power in specific frequency bands
        very_low_power = psd[(f >= 0) & (f < 0.5)].sum()  # Stillness
        low_power = psd[(f >= 0.5) & (f < 3)].sum()  # Normal gait
        mid_power = psd[(f >= 3) & (f < 8)].sum()  # Freeze/tremor
        high_power = psd[(f >= 8) & (f < 15)].sum()  # Artifacts
        
        # Spectral edge frequency (95% power threshold)
        cumsum_psd = np.cumsum(psd)
        total_power = cumsum_psd[-1]
        edge_freq = f[np.where(cumsum_psd >= 0.95 * total_power)[0][0]] if total_power > 0 and len(np.where(cumsum_psd >= 0.95 * total_power)[0]) > 0 else 0
        
        # Bandwidth (frequency range containing 90% of power)
        f_low = f[np.where(cumsum_psd >= 0.05 * total_power)[0][0]] if total_power > 0 and len(np.where(cumsum_psd >= 0.05 * total_power)[0]) > 0 else 0
        f_high = f[np.where(cumsum_psd >= 0.95 * total_power)[0][0]] if total_power > 0 and len(np.where(cumsum_psd >= 0.95 * total_power)[0]) > 0 else 0
        bandwidth = f_high - f_low
        
        # Dominant frequency amplitude
        dom_freq_amp = np.max(psd) if len(psd) > 0 else 0
        
        return spectral_entropy, very_low_power, low_power, mid_power, high_power, edge_freq, bandwidth, dom_freq_amp
    
    def calculate_wavelet_features(self, window):
        """Calculate wavelet decomposition features"""
        try:
            import pywt
            
            # Perform 3-level wavelet decomposition
            coeffs = pywt.wavedec(window, 'db4', level=3)
            
            # Energy in each level
            energies = [np.sum(c**2) for c in coeffs]
            wavelet_energy_1 = energies[0] if len(energies) > 0 else 0
            wavelet_energy_2 = energies[1] if len(energies) > 1 else 0
            wavelet_energy_3 = energies[2] if len(energies) > 2 else 0
            
            # Wavelet entropy
            total_energy = sum(energies)
            if total_energy > 0:
                energy_dist = [e / total_energy for e in energies]
                wavelet_entropy = -sum([e * np.log2(e + 1e-10) for e in energy_dist if e > 0])
            else:
                wavelet_entropy = 0
                
        except ImportError:
            # If pywt not available, return zeros
            wavelet_energy_1 = wavelet_energy_2 = wavelet_energy_3 = wavelet_entropy = 0
        
        return wavelet_energy_1, wavelet_energy_2, wavelet_energy_3, wavelet_entropy
    
    def calculate_temporal_features(self, window):
        """Calculate temporal correlation features"""
        # Autocorrelation at lag 1 and 2
        if len(window) > 2:
            autocorr_1 = np.corrcoef(window[:-1], window[1:])[0, 1]
            autocorr_1 = autocorr_1 if np.isfinite(autocorr_1) else 0
        else:
            autocorr_1 = 0
            
        if len(window) > 3:
            autocorr_2 = np.corrcoef(window[:-2], window[2:])[0, 1]
            autocorr_2 = autocorr_2 if np.isfinite(autocorr_2) else 0
        else:
            autocorr_2 = 0
        
        # Signal-to-noise ratio (approximation)
        signal_power = np.mean(window**2)
        noise_power = np.var(np.diff(window))  # High-frequency variation as noise
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        snr = snr if np.isfinite(snr) else 0
        
        return autocorr_1, autocorr_2, snr
    
    def calculate_additional_features(self, window):
        """Calculate additional time-domain and frequency-domain features"""
        # Time-domain features
        mean = np.mean(window)
        median = np.median(window)
        rms = np.sqrt(np.mean(window**2))  # Root Mean Square
        
        # Frequency-domain: peak frequency
        f, psd = signal.welch(window, fs=self.sampling_rate)
        peak_freq = f[np.argmax(psd)] if len(psd) > 0 else 0.0
        
        # Zero-crossing rate (cadence indicator)
        zcr = np.sum(np.diff(np.sign(window)) != 0) / len(window) if len(window) > 1 else 0.0
        
        # Statistical features
        kurtosis = stats.kurtosis(window)
        if not np.isfinite(kurtosis):
            kurtosis = 0.0
        
        # Amplitude variation features
        range_val = np.ptp(window)  # peak-to-peak (max - min)
        iqr = np.percentile(window, 75) - np.percentile(window, 25)  # Interquartile range
        
        return mean, median, rms, peak_freq, zcr, kurtosis, range_val, iqr
    
    def extract_features(self, windows_dict):
        """
        Extract features from multiple axes.
        
        Args:
            windows_dict: Dictionary where key is axis name (e.g., 'x', 'y', 'mag') 
                          and value is array of windows for that axis.
        
        Returns:
            Numpy array of shape (n_windows, n_features * n_axes)
        """
        all_axis_features = []
        
        # Iterate through each axis (e.g., 'x', 'y', 'z', 'mag')
        for axis_name, windows in windows_dict.items():
            axis_features_matrix = []
            for window in windows:
                # Original 5 features
                freeze_index, spectral_centroid = self.calculate_freq_features(window)
                energy = self.calculate_energy(window)
                var = self.calculate_var(window)
                skew = self.calculate_skew(window)
                
                # Additional 8 features
                mean, median, rms, peak_freq, zcr, kurtosis, range_val, iqr = self.calculate_additional_features(window)
                
                # New time-domain features (9 features)
                min_val, max_val, p25, p75, mad, sma, cv, n_peaks, crest_factor = self.calculate_time_domain_features(window)
                
                # New advanced frequency features (8 features)
                spec_entropy, vlow_pow, low_pow, mid_pow, high_pow, edge_freq, bandwidth, dom_freq_amp = self.calculate_advanced_freq_features(window)
                
                # Wavelet features (4 features)
                wav_e1, wav_e2, wav_e3, wav_entropy = self.calculate_wavelet_features(window)
                
                # Temporal features (3 features)
                autocorr1, autocorr2, snr = self.calculate_temporal_features(window)
                
                # Combine all features for this window
                axis_features_matrix.append([
                    freeze_index, energy, var, skew, spectral_centroid,
                    mean, median, rms, peak_freq, zcr, kurtosis, range_val, iqr,
                    min_val, max_val, p25, p75, mad, sma, cv, n_peaks, crest_factor,
                    spec_entropy, vlow_pow, low_pow, mid_pow, high_pow, edge_freq, bandwidth, dom_freq_amp,
                    wav_e1, wav_e2, wav_e3, wav_entropy,
                    autocorr1, autocorr2, snr
                ])
            all_axis_features.append(np.array(axis_features_matrix))
            
        # Concatenate features from all axes horizontally
        # Shape: (n_windows, n_features_per_axis * n_axes)
        return np.hstack(all_axis_features)
    
    def get_feature_names(self, axes=['x', 'y', 'z', 'mag']):
        """Return list of feature names for DataFrame columns, prefixed by axis"""
        base_features = [
            'freeze_index', 'energy', 'variance', 'skewness', 'spectral_centroid',
            'mean', 'median', 'rms', 'peak_frequency', 'zero_crossing_rate', 
            'kurtosis', 'range', 'iqr',
            'min', 'max', 'percentile_25', 'percentile_75', 'mad', 'sma', 
            'coeff_variation', 'n_peaks', 'crest_factor',
            'spectral_entropy', 'very_low_power', 'low_power', 'mid_power', 
            'high_power', 'edge_frequency', 'bandwidth', 'dom_freq_amplitude',
            'wavelet_energy_1', 'wavelet_energy_2', 'wavelet_energy_3', 'wavelet_entropy',
            'autocorr_lag1', 'autocorr_lag2', 'snr'
        ]
        
        full_feature_names = []
        for axis in axes:
            for feature in base_features:
                full_feature_names.append(f"{axis}_{feature}")
                
        return full_feature_names