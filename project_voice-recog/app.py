import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model
MODEL_PATH = "voice_cnn_model.h5"
model = load_model(MODEL_PATH)

SAMPLE_RATE = 22050
DURATION = 1.5
N_MFCC = 40
SAMPLES = int(SAMPLE_RATE * DURATION)

st.title("🎤 Voice Recognition: Buka / Tutup")

uploaded_file = st.file_uploader("Upload file suara", type=["wav", "mp3", "m4a"])

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)

    if len(y) < SAMPLES:
        pad = SAMPLES - len(y)
        y = np.concatenate([y, np.zeros(pad)])
    else:
        y = y[:SAMPLES]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc

if uploaded_file:
    st.audio(uploaded_file)

    features = preprocess_audio(uploaded_file)
    pred = model.predict(features)
    label = "BUKA" if pred[0][0] > 0.5 else "TUTUP"

    st.subheader("🔊 Prediksi:")
    st.success(label)
