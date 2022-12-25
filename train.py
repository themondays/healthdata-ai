import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATASET_DIR="dataset"

if not os.path.exists(DATASET_DIR):
  os.makedirs(DATASET_DIR)

# Load the data from the .npy file
data = np.load(f"{DATASET_DIR}/medical_data.npy")

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2)

# Create a random forest classifier
model = RandomForestClassifier(n_estimators=100)

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)

print('Test accuracy:', accuracy)

joblib.dump(model, 'model.pkl')

