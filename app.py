import joblib
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
model = joblib.load("hb.pkl")

# Function to extract exactly 131 features
def extract_features(file):
    y, sr = librosa.load(file, sr=22050)

    # Extract MFCCs (124 features)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=124)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Extract additional features (7 more features)
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    
    # Additional features to reach 131
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  # 1 feature
    rms_energy = np.mean(librosa.feature.rms(y=y))  # 1 feature
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # 1 feature (beat tracking)

    # Combine all features (124 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 131)
    features = np.hstack([
        mfccs_mean, zero_crossing, spectral_centroid, spectral_rolloff, chroma_stft, 
        spectral_bandwidth, rms_energy, tempo
    ])
    
    return features.reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Extract features
        features = extract_features(file)
        print("Extracted Features Shape:", features.shape)  # Debugging

        # Ensure the feature shape matches the model
        if features.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Feature shape mismatch: Model expects {model.n_features_in_}, got {features.shape[1]}"}), 500

        # Predict class
        prediction = model.predict(features)[0]

        # Convert prediction to label
        label_map = {4: "normal", 3: "murmur", 0: "artifact", 2: "extrastole", 1: "extrahls"}
        predicted_label = label_map.get(prediction, "Unknown")

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
