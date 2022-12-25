import os
import json
import numpy as np
import pandas as pd

DATASET_DIR='./dataset'

# Prepare
os.makedirs(DATASET_DIR, exist_ok=True)

# Load the JSON documents into a list
documents = []
for filename in os.listdir('./input/users'):
    with open(os.path.join('./input/users', filename)) as f:
        documents.append(json.load(f))

# Define dataframe
df = pd.DataFrame(documents, columns=[
    'age',
    'sex',
    'ethnicity',
    'num_diseases',
    'num_vaccines',
    'num_chemo',
    'num_hormonal_therapies',
    'num_cancers',
    'hrt',
    'hiv_aids_positive',
    'remission',
    'red_blood_cell_count',
    'white_blood_cell_count',
    'platelet_count',
    'sodium_level',
    'potassium_level',
    'creatinine_level',
    'ast_level',
    'alt_level',
    'bilirubin_level',
    'thyroid_stimulating_hormone_level',
    'prolactin_level',
    'clotting_time'
])

df = pd.get_dummies(df, columns=['ethnicity'])

# Preprocess the data
for doc in documents:
    # Convert age to integer
    doc['age'] = int(doc['age'])
    # Convert sex to integer (0 for female, 1 for male)
    doc['sex'] = int(doc['sex'] == 'male')
    # Convert ethnicity to integer
    #doc['ethnicity'] = int(doc['ethnicity'])
    # Convert hrt to integer (0 for false, 1 for true)
    doc['hrt'] = int(doc['hrt'])
    # Convert hiv_aids_positive to integer (0 for false, 1 for true)
    doc['hiv_aids_positive'] = int(doc['hiv_aids_positive'])
    # Convert remission to integer (0 for false, 1 for true)
    doc['remission'] = int(doc['remission'])

# Create a list of data points
data = []
for doc in documents:
    data.append([
        int(doc['age']),
        doc['sex'],
        doc['ethnicity'],
        len(doc['diseases']),
        len(doc['vaccines']),
        len(doc['chemo']),
        len(doc['hormonal_therapies']),
        len(doc['cancers']),
        doc['hrt'],
        doc['hiv_aids_positive'],
        doc['remission'],
        doc['red_blood_cell_count'],
        doc['white_blood_cell_count'],
        doc['platelet_count'],
        doc['sodium_level'],
        doc['potassium_level'],
        doc['creatinine_level'],
        doc['ast_level'],
        doc['alt_level'],
        doc['bilirubin_level'],
        doc['thyroid_stimulating_hormone_level'],
        doc['prolactin_level'],
        doc['clotting_time']
    ])
data = np.array(data)
np.save(f"./{DATASET_DIR}/medical_data.npy", data)

