from src.MLModel import Model
from src.feature_extractor import FeatureExtractor
from src.preprocess import DataPreprocessor
from src.visuals import DataVisualizer
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

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
    patient_files = list(glob.glob(f'{data_dir}/*.csv'))
    patients_data = {}
    for i, patient_file in enumerate(glob.glob(f'{data_dir}/*.csv')):
        processed_data = load_and_preprocess(patient_file)
        patients_data[i+1] = processed_data
    print("Preprocessing and dictionary loading done")
    
    visualizer = DataVisualizer()
    
    print("Creating raw vs processed signal visualization...")
    if patient_files:
        preprocessor = DataPreprocessor()
        try:
            raw_fig = visualizer.plot_raw_vs_processed_data(
                patient_files[2], 
                preprocessor, 
                start_idx=51343,  # ~3 minutes
                duration=120      # 120 seconds
            )
            plt.show()
        except Exception as e:
            print(f"Note: Raw signal plot had an issue: {e}")
            
    print("Starting Model LOSO training and testing")
    model = Model(model_type='random_forest')
    y_true, y_pred, y_proba, patient_results = model.loso(patients_data)
    results = model._calculate_metrics(y_true, y_pred)
    
    print(f"\nRESULTS:")
    print(f"   Sensitivity: {results['sensitivity']:.1%}")
    print(f"   Specificity: {results['specificity']:.1%}")
    print(f"   Target: 73% sens, 81% spec")
    
    # data visualization:
    print("Creating performance plots...")
    performance_fig = visualizer.plot_confusion_matrix_roc_pr(y_true, y_pred, y_proba)
    plt.show()
    
    print("\nPatient-wise LOSO Performance:")
    print("-" * 60)
    print(f"{'Patient':<8} {'Sensitivity':<12} {'Specificity':<12} {'Windows':<8} {'FOG':<6}")
    print("-" * 60)
    for p in patient_results:
        print(f"{p['patient_id']:<8} {p['sensitivity']:<12.1%} {p['specificity']:<12.1%} "
            f"{p['n_windows']:<8} {p['n_fog_windows']:<6}")
    print("-" * 60)

    # Calculate patient variability stats
    sensitivities = [p['sensitivity'] for p in patient_results]
    specificities = [p['specificity'] for p in patient_results]
    print(f"\nPatient Variability:")
    print(f"Sensitivity: {np.mean(sensitivities):.1%} ± {np.std(sensitivities):.1%}")
    print(f"Specificity: {np.mean(specificities):.1%} ± {np.std(specificities):.1%}")
    print(f"Patients meeting sensitivity target (>73%): {sum(1 for s in sensitivities if s >= 0.73)}/{len(sensitivities)}")
    print(f"Patients meeting specificity target (>81%): {sum(1 for s in specificities if s >= 0.81)}/{len(specificities)}")
    
    
    return patients_data, results

if __name__ == "__main__":
    main()