from src.preprocess import DataPreprocessor
import numpy as np
import glob
"""
# Test with synthetic data
test_data = np.random.randn(1000)  # 1000 samples
preprocessor = DataPreprocessor()

filtered = preprocessor.apply_butter(test_data)
windows = preprocessor.create_windows(filtered)
normalized = preprocessor.normalize_windows(windows)

print(f"Windows shape: {windows.shape}")
print(f"Normalized mean: {normalized[0].mean():.6f}")  # Should be ~0
print(f"Normalized std: {normalized[0].std():.6f}")   # Should be ~1"""

# Test with all real data:
for patient_file in glob.glob(f'data/csv/*.csv'):
    print(patient_file)
    data = np.genfromtxt(patient_file, delimiter=',', skip_header=1)
    data = data[:,4]
    preprocessor = DataPreprocessor()

    filtered = preprocessor.apply_butter(data)
    windows = preprocessor.create_windows(filtered)
    normalized = preprocessor.normalize_windows(windows)

    print(f"Windows shape: {windows.shape}")
    print(f"Normalized mean: {normalized[0].mean():.6f}")  # Should be ~0
    print(f"Normalized std: {normalized[0].std():.6f}")   # Should be ~1"""
