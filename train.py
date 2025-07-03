import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Define disease categories
CATEGORIES = {'normal_noisy': 0, 'artifact': 1, 'murmur': 2, 'extrahls': 3, 'Aunlabelledtest': 4, 'normal': 5}

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def load_data(data_folder):
    X, y = [], []
    for subset in ["set_a", "set_b"]:
        subset_path = os.path.join(data_folder, subset)
        if os.path.exists(subset_path):
            for file_name in os.listdir(subset_path):
                if file_name.endswith(".wav"):
                    label_name = file_name.split("_")[0].lower()  # Extract label from filename
                    if label_name in CATEGORIES:
                        label = CATEGORIES[label_name]
                        file_path = os.path.join(subset_path, file_name)
                        features = extract_features(file_path)
                        X.append(features)
                        y.append(label)
    return np.array(X), np.array(y)

data_folder = "heartsounds_dataset"
X, y = load_data(data_folder)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

with open("hb.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("✅ Model training complete. Saved as hb.pkl")
