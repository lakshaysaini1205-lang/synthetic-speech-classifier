import os
import time
import io
import base64
import datetime
import traceback
import numpy as np
import librosa
import matplotlib
# Use 'Agg' for headless server-side image generation
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify

# Keras 3 / TensorFlow Integration
import keras
from keras.models import load_model

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_deepfake_model.h5')

print("Initializing Neural Brain...")
model = load_model(MODEL_PATH)

# Model Constraints
N_MFCC = 40
MAX_TIME_STEPS = 63

def process_audio_features(file_path):
    """Pre-processes audio into 16kHz MFCCs for the CNN."""
    audio, sr = librosa.load(file_path, sr=16000, duration=2.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    
    if mfcc.shape[1] < MAX_TIME_STEPS:
        pad_width = MAX_TIME_STEPS - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_TIME_STEPS]
        
    return mfcc.reshape(1, N_MFCC, MAX_TIME_STEPS, 1), audio, sr

def generate_spectrogram(audio, sr):
    """Creates the Mel-Spectrogram visual report."""
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#09090b') 
    ax.set_facecolor('#09090b')
    
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.axis('off') 
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return f"data:image/png;base64,{base64_img}"

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    start_time = time.time()
    
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "No audio data received"})
    
    file = request.files['audio_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        # 1. Neural Analysis
        features, audio, sr = process_audio_features(filepath)
        prediction = model.predict(features)
        spectrogram_url = generate_spectrogram(audio, sr)
        
        confidence = float(np.max(prediction) * 100)
        predicted_class = int(np.argmax(prediction)) 
        
        # 2. Case Identification
        report_id = datetime.datetime.now().strftime("SCAN_%Y%m%d_%H%M%S")
        
        # 3. Performance Metrics
        actual_time = time.time() - start_time
        latency = round(actual_time * 1000, 2)
        storage_kb = round(os.path.getsize(filepath) / 1024, 2)
        
        # Artificial delay for UI presentation flow
        if actual_time < 3.0:
            time.sleep(3.0 - actual_time)
        
        result_text = "Real Human Speech" if predicted_class == 1 else "AI-Generated/Synthetic"
        interp = "Natural spectral signatures detected." if predicted_class == 1 else "Digital artifacts and phase errors found."

        return jsonify({
            "status": "success", "result": result_text, "report_id": report_id,
            "confidence": f"{confidence:.2f}%", "latency": f"{latency} ms",
            "storage": f"{storage_kb} KB", "interpretation": interp,
            "spectrogram": spectrogram_url
        })

    except Exception as e:
        # Logs the specific error to your local terminal for troubleshooting
        print(f"\n[SCAN ERROR]: {repr(e)}")
        traceback.print_exc() 
        return jsonify({"status": "error", "message": str(e) or "Format incompatible. Check FFmpeg."})
    
    finally:
        if os.path.exists(filepath): os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)