from src.MLModel import Model
from src.feature_extractor import FeatureExtractor
from src.preprocess import DataPreprocessor
from src.visuals import DataVisualizer
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

def get_labels(windows, threshold=0.3):
    """
    Convert window labels to binary using majority voting.
    
    Args:
        windows: Array of label windows
        threshold: Minimum ratio of FoG samples to classify as FoG (default 0.3)
                  Lowered from 0.5 to catch FoG episodes at window boundaries.
                  Clinical priority: better to alert early than miss FoG.
    
    Returns:
        List of binary labels (0=no FoG, 1=FoG)
    """
    labels = []
    for window in windows:
        # Count FoG samples (label 2) in window
        fog_ratio = np.sum(window == 2) / len(window) if len(window) > 0 else 0
        # Use 30% threshold: if >= 30% of window is FoG, label as FoG
        labels.append(1 if fog_ratio >= threshold else 0)
    return labels
def load_and_preprocess(patient_file_dir):
    preprocessor = DataPreprocessor()
    feature_extractor = FeatureExtractor()
    data = np.genfromtxt(patient_file_dir, delimiter=',', skip_header=1)
    
    # Use thigh sensor as proxy for wrist (columns 4,5,6 = forward, vertical, lateral)
    # Mapping to standard X, Y, Z (assuming forward=X, vertical=Y, lateral=Z)
    thigh_x = data[:, 3]
    thigh_y = data[:, 4]
    thigh_z = data[:, 5]
    
    # Compute 3D magnitude: sqrt(x^2 + y^2 + z^2)
    thigh_magnitude = np.sqrt(thigh_x**2 + thigh_y**2 + thigh_z**2)
    
    labels_data = data[:, -1]
    
    # Preprocess all signals with low-pass filter
    filtered_x = preprocessor.apply_butter(thigh_x)
    filtered_y = preprocessor.apply_butter(thigh_y)
    filtered_z = preprocessor.apply_butter(thigh_z)
    filtered_mag = preprocessor.apply_butter(thigh_magnitude)
    
    # Create windows for all axes WITHOUT normalization
    windows_x = preprocessor.create_windows(filtered_x)
    windows_y = preprocessor.create_windows(filtered_y)
    windows_z = preprocessor.create_windows(filtered_z)
    windows_mag = preprocessor.create_windows(filtered_mag)
    
    # Prepare dictionary for feature extractor
    windows_dict = {
        'x': windows_x,
        'y': windows_y,
        'z': windows_z,
        'mag': windows_mag
    }
    
    # Extract features from all axes
    # This will generate 37 features * 4 axes = 148 features
    features = feature_extractor.extract_features(windows_dict)
    
    # Get labels with 30% threshold
    labels_windows = preprocessor.create_windows(labels_data)
    labels = get_labels(labels_windows, threshold=0.3)
    
    # Create DataFrame with expanded feature names
    feature_names = feature_extractor.get_feature_names(axes=['x', 'y', 'z', 'mag'])
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add labels
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
    
    print("Creating raw vs processed signal visualization...", flush=True)
    if patient_files:
        preprocessor = DataPreprocessor()
        try:
            # Fix: Don't pass save_path to the function, save the returned figure instead
            raw_fig = visualizer.plot_raw_vs_processed_data(
                patient_files[2], 
                preprocessor
            )
            raw_fig.savefig('raw_vs_processed.png')
            print("Visualization created successfully", flush=True)
        except Exception as e:
            print(f"Visualization failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
    print("Starting Model LOSO training and testing", flush=True)
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