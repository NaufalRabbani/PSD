import streamlit as st
import numpy as np
import librosa
from pydub import AudioSegment
import tempfile
import joblib

MODEL_PATH = "voice_svm_model.pkl"
model = joblib.load(MODEL_PATH)

SAMPLE_RATE = 22050
DURATION = 1.5
N_MFCC = 40
SAMPLES = int(SAMPLE_RATE * DURATION)

st.title("🎤 Voice Recognition: Buka / Tutup (SVM Version)")

uploaded_file = st.file_uploader("Upload file suara", type=["wav", "mp3", "m4a"])

def preprocess_audio(uploaded):
    # Simpan ke file sementara
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        file_path = tmp.name

    # Convert jika bukan WAV
    if uploaded.type != "audio/wav":
        audio = AudioSegment.from_file(file_path)
        wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(wav_temp.name, format="wav")
        file_path = wav_temp.name

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.flatten()

    return mfcc.reshape(1, -1)

if uploaded_file:
    st.audio(uploaded_file)

    features = preprocess_audio(uploaded_file)
    pred = model.predict(features)[0]

    st.subheader("🔊 Prediksi:")
    st.success("BUKA" if pred == 1 else "TUTUP")
