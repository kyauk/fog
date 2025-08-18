from src.preprocess import DataPreprocessor
from src.feature_extractor import FeatureExtractor
import numpy as np

def test_feature_extractor():
    print("Testing FeatureExtractor...")
    
    # Create test data
    preprocessor = DataPreprocessor()
    feature_extractor = FeatureExtractor()
    
    # Generate synthetic signal (walking + some FOG-like patterns)
    np.random.seed(42)  # Reproducible results
    t = np.linspace(0, 15, 1000)  # 15 seconds of data
    
    # Simulate walking (2 Hz) + some higher frequency content (FOG-like at 5 Hz)
    walking_signal = np.sin(2 * np.pi * 2 * t)
    fog_signal = 0.3 * np.sin(2 * np.pi * 5 * t[500:])  # FOG in second half
    test_signal = walking_signal.copy()
    test_signal[500:] += fog_signal
    test_signal += 0.1 * np.random.randn(len(t))  # Add noise
    
    # Preprocess
    filtered_data = preprocessor.apply_butter(test_signal)
    windows = preprocessor.create_windows(filtered_data)
    normalized_windows = preprocessor.normalize_windows(windows)
    
    print(f"Created {len(normalized_windows)} windows for feature extraction")
    
    # Extract features
    features = feature_extractor.extract_features(normalized_windows)
    
    print(f"\nFeature Matrix Shape: {features.shape}")
    print(f"Features per window: {features.shape[1]}")
    
    # Check individual features
    print(f"\nFeature Statistics:")
    feature_names = ['freeze_index', 'energy', 'variance', 'skewness', 'spectral_centroid']
    
    for i, name in enumerate(feature_names):
        values = features[:, i]
        print(f"{name:17}: mean={values.mean():.3f}, std={values.std():.3f}, range=[{values.min():.3f}, {values.max():.3f}]")
    
    # Test individual window
    print(f"\nTesting single window...")
    single_window = normalized_windows[0]
    freeze_idx, spec_centroid = feature_extractor.calculate_freq_features(single_window)
    energy = feature_extractor.calculate_energy(single_window)
    variance = feature_extractor.calculate_var(single_window)
    skewness = feature_extractor.calculate_skew(single_window)
    
    print(f"Single window features:")
    print(f"  Freeze Index: {freeze_idx:.3f}")
    print(f"  Energy: {energy:.3f}")
    print(f"  Variance: {variance:.3f}")
    print(f"  Skewness: {skewness:.3f}")
    print(f"  Spectral Centroid: {spec_centroid:.3f}")
    
    # Verify no NaN values
    has_nan = np.isnan(features).any()
    print(f"\nContains NaN values: {has_nan}")
    
    if not has_nan:
        print("✅ Feature extraction working correctly!")
    else:
        print("❌ Found NaN values - check your calculations")

if __name__ == "__main__":
    test_feature_extractor()