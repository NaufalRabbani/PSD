# train_svm.py
import os
import librosa
import numpy as np
from sklearn.svm import SVC
import joblib

DATASET_PATH = "dataset/augmented_audio"  # Folder dataset kamu
SAMPLE_RATE = 22050
DURATION = 1.5
N_MFCC = 40
SAMPLES = int(SAMPLE_RATE * DURATION)

X = []
y = []

labels = {"buka": 1, "tutup": 0}  # label klasifikasi

print("\n🔄 Loading dataset...")

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    print(f"\n📁 Processing folder: {folder}")

    for file_name in os.listdir(folder):
        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(folder, file_name)
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Padding/truncate ke durasi tertentu
        if len(y_audio) < SAMPLES:
            y_audio = np.pad(y_audio, (0, SAMPLES - len(y_audio)))
        else:
            y_audio = y_audio[:SAMPLES]

        # Ekstraksi fitur MFCC
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
        mfcc = mfcc.flatten()  # SVM butuh input 1D

        X.append(mfcc)
        y.append(labels[label])

print("\n✅ Ekstraksi fitur selesai")
print(f"Total data: {len(X)}")

# SVM Training
print("\n🧠 Training SVM model...")
clf = SVC(kernel="linear", probability=True)
clf.fit(X, y)

# Save model
joblib.dump(clf, "voice_svm_model.pkl")
print("\n✅ Model saved: voice_svm_model.pkl")
