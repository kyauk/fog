import pandas as pd
import numpy as np
import glob
import os
# load .txt files
files = glob.glob("data/csv/daphnet/*.txt")
# create columns to a csv
'''
1. Time of sample in millisecond
2. Ankle (shank) acceleration - horizontal forward acceleration (mg)
3. Ankle (shank) acceleration - vertical (mg)
4. Ankle (shank) acceleration - horizontal lateral (mg)
5. Upper leg (thigh) acceleration - horizontal forward acceleration (mg)
6. Upper leg (thigh) acceleration - vertical (mg)
7. Upper leg (thigh) acceleration - horizontal lateral (mg)
8. Trunk acceleration - horizontal forward acceleration (mg)
9. Trunk acceleration - vertical (mg)
10. Trunk acceleration - horizontal lateral (mg)
11. Annotations (see Annotations section) 0 = Not part of experiment 1 = No Freeze 2 = Freeze

'''
columns = [
    'time', 
    'ankle_forward', 'ankle_vert', 'ankle_lat', 
    'thigh_forward', 'thigh_vert', 'thigh_lat',
    'trunk_forward', 'trunk_vert', 'trunk_lat',
    'label'
]

for i, file in enumerate(files):
    df = pd.read_csv(file, sep=' ', header=None, names=columns)
    
    # Save individual patient file
    patient_filename = f'data/csv/patient_{i:02d}.csv'
    os.makedirs('data/csv', exist_ok=True)
    df.to_csv(patient_filename, index=False)
    
    print(f"Patient {i}: {df.shape} - Labels: {df['label'].value_counts().to_dict()}")

print("Saved individual patient files to data/csv/")