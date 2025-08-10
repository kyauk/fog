import pandas as pd
import numpy as np
import glob

# load .txt files
files = glob.glob("data/raw/daphnet/*.txt")
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

all_data = []

for file in files:
    df = pd.read_csv(file, sep=' ', header=None, names=columns)
    all_data.append(df)
    
# here we will combine data then turn it to a csv and return it back to this path: data/raw/daphnet/
combined_data = pd.concat(all_data, ignore_index=True)
# for future debugging purposes and seeing if we formatted correctly:

print(f'Shape: {combined_data.shape}')
print(f"Label counts: {combined_data['label'].value_counts()}")
print("first few rows:\n", combined_data.head())
# Now we route to csv to path
path = 'data/raw/daphnet/daphnet_data.csv'
combined_data.to_csv(path, index=False)
print(f"Saved CSV to {path}!")