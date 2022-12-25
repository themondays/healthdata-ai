import numpy as np
import joblib
import tensorflow as tf

# Load the trained model from a file
model = joblib.load('model.pkl')

# Define the user object
user = {
    "age": 35,
    "sex": "male",
    "ethnicity": "Caucasian",
    "location": "New York, NY",
    "country": "US",
    "vaccines": ["influenza", "tetanus"],
    "hrt": True,
    "hormonal_therapies": ["testosterone replacement"],
    "hiv_aids_positive": True,
    "cancers": ["lung", "colon"],
    "chemo": ["carboplatin", "paclitaxel"],
    "remission": False,
    "red_blood_cell_count": 4.7,
    "white_blood_cell_count": 8.1,
    "platelet_count": 250,
    "sodium_level": 142,
    "potassium_level": 4.3,
    "creatinine_level": 0.9,
    "ast_level": 22,
    "alt_level": 32,
    "bilirubin_level": 0.5,
    "thyroid_stimulating_hormone_level": 1.4,
    "prolactin_level": 10,
    "clotting_time": 8,
    "infectious_agent": "none"
}

# Convert the user object to a NumPy array
X = np.array([
    user['age'],
    1 if user['sex'] == 'male' else 0,
    1 if user['ethnicity'] == 'Caucasian' else 0,
    len(user['vaccines']),
    int(user['hrt']),
    len(user['hormonal_therapies']),
    int(user['hiv_aids_positive']),
    len(user['cancers']),
    len(user['chemo']),
    int(user['remission']),
    user['red_blood_cell_count'],
    user['white_blood_cell_count'],
    user['platelet_count'],
    user['sodium_level'],
    user['potassium_level'],
    user['creatinine_level'],
    user['ast_level'],
    user['alt_level'],
    user['bilirubin_level'],
    user['thyroid_stimulating_hormone_level'],
    user['prolactin_level'],
    user['clotting_time'],
])

prediction = model.predict(X)
y_pred = model.predict_proba(X)

# diseases = model.diseases_
# diseases_with_probs = sorted(zip(diseases, prediction[0]), key=lambda x: x[1], reverse=True)

diseases = []
for i, disease in enumerate(model.classes_):
    diseases.append({
        'name': disease,
        'probability': y_pred[0, i]
    })

print(diseases)
print(prediction[0] * 100)



