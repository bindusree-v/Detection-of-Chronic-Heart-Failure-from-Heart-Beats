from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
with open("hb.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define disease categories
CATEGORIES = ['normal_noisy', 'artifact', 'murmur', 'extrahls', 'Aunlabelledtest', 'normal']

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Extract features and predict
    features = extract_features(file_path)
    prediction_index = model.predict(features)[0]
    prediction = CATEGORIES[int(prediction_index)]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
