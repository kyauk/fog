import numpy as np
from scipy import signal
import pandas as pd

def calc_freeze_index(data, sampling_rate=64):
    
    data = (pd.to_numeric(data, errors='coerce').dropna()).values
    # Minimum of 4 seconds @ 64 Hz (256)
    if len(data) < 256:
        return 0
    f,psd = signal.welch(data, fs=sampling_rate)
    # Freeze power is TOTAL energy Between [3,8]
    freeze_power = psd[(f >=3) & (f <= 8)].sum()
    # Locomotion power is TOTAL energy Between [0.5,3]
    loco_power = psd[(f >= 0.5) & (f <= 3)].sum()
    
    return freeze_power/loco_power if loco_power > 0 else 0

def calculate_energy(data):
    sampling_rate = 64
    f, psd = signal.welch(data, fs=sampling_rate)
    return np.sum(psd)
    
def determine_power_threshold(data):
    standing_data = data[data['label'] == 0]['thigh_vert']
    
    standing_energies = [calculate_energy(row) for row in standing_data]
    standing_mean = np.mean(standing_energies)
    standing_std = np.std(standing_energies)
    # threshold starts at the upper most percentile for standing
    return standing_mean + 2 * standing_std 

# paper used a window size of 4seconds, with a step size of 0.5 sec. translating that to hz is 256 and 32 respectively.
def sliding_window(data, PowerTH):
    window_size = 256
    step_size = 32
    fog_threshold = 2.0
    results = []
    # start from 0, last frame will be len(data) - window_size, and move by step size
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[start:start + window_size].values
        energy = calculate_energy(window)
        if energy > PowerTH:
            fi = calc_freeze_index(window)
            fog_detected = fi > fog_threshold
        else:
            fog_detected = False
        results.append(fog_detected)
    return results
    

def test_baseline():
    data = pd.read_csv('data/raw/daphnet/daphnet_data.csv')
    normal_thigh_sensor_data = data[data['label']==1]['thigh_vert']
    fog_thigh_sensor_data = data[data['label']==2]['thigh_vert']
    
    PowerTH = determine_power_threshold(data)
    
    normal_results = sliding_window(normal_thigh_sensor_data, PowerTH)
    fog_results = sliding_window(fog_thigh_sensor_data, PowerTH)
    
    
    normal_detections = sum(normal_results)
    fog_detections = sum(fog_results)
    
    print(f"Normal walking FOG detections: {normal_detections}/{len(normal_results)}")
    print(f"FOG episode detections: {fog_detections}/{len(fog_results)}")
    print(f"Normal detection rate: {normal_detections/len(normal_results):.3f}")
    print(f"FOG detection rate: {fog_detections/len(fog_results):.3f}")
    

if __name__ == "__main__":
    test_baseline()