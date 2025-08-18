from src.MLModel import Model
from src.feature_extractor import FeatureExtractor
from src.preprocess import DataPreprocessor
import numpy as np
import pandas as pd
import glob

def get_labels(windows):
    labels = []
    for window in windows:
        if 1 not in window and 0 not in window:
            labels.append(1)
        else:
            labels.append(0)
    return labels
def load_and_preprocess(patient_file_dir):
    preprocessor = DataPreprocessor()
    feature_extractor = FeatureExtractor()
    data = np.genfromtxt(patient_file_dir, delimiter=',', skip_header=1)
    # ankle is 2
    thigh_data = data[:,4]
    labels_data = data[:,-1]
    # preprocess data and extract features, labels
    filtered_data = preprocessor.apply_butter(thigh_data)
    windows = preprocessor.normalize_windows(preprocessor.create_windows(filtered_data))
    features = feature_extractor.extract_features(windows)
    labels_windows = preprocessor.create_windows(labels_data)
    labels = get_labels(labels_windows)
    
    # reformat to pandas
    df = pd.DataFrame(features, columns=[
        'freeze_index',
        'energy_threshold', 
        'variance',
        'skewness',
        'spectral_centroid'
    ])
    
    # add labels and patient ID
    df['label'] = labels
    return df

def main():
    print("Starting main pipeline:")
    data_dir = 'data/csv'
    patients_data = {}
    for i, patient_file in enumerate(glob.glob(f'{data_dir}/*.csv')):
        processed_data = load_and_preprocess(patient_file)
        patients_data[i+1] = processed_data
    print("Preprocessing and dictionary loading done")
    print("Starting Model LOSO training and testing")
    model = Model(model_type='random_forest')
    y_true, y_pred = model.loso(patients_data)
    results = model._calculate_metrics(y_true, y_pred)
    print(f"\nRESULTS:")
    print(f"   Sensitivity: {results['sensitivity']:.1%}")
    print(f"   Specificity: {results['specificity']:.1%}")
    print(f"   Target: 73% sens, 81% spec")
    
    return patients_data, results

if __name__ == "__main__":
    main()