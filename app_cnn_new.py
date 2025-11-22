import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. Konfigurasi ---
DATASET_PATH = "dataset_amino/" # GANTI DENGAN PATH DATASET ANDA
N_MELS = 128
FMAX = 8000
MAX_LEN = 216 # Durasi sekitar 5 detik dengan sr=22050
AUGMENT_FACTOR = 5 # Berapa kali augmentasi per file asli

# --- 2. Fungsi Augmentasi ---
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def shift_time(data, sr, shift_max_sec=0.2):
    shift_range = int(sr * shift_max_sec)
    shift = np.random.randint(-shift_range, shift_range)
    return np.roll(data, shift)

def change_pitch(data, sr, pitch_factor_range=(-2, 2)):
    pitch_factor = np.random.randint(pitch_factor_range[0], pitch_factor_range[1])
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

# --- 3. Fungsi Ekstraksi Fitur ---
def extract_mel_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_LEN]
    return mel_spec_db

# --- 4. Proses Pemuatan dan Augmentasi Data ---
def load_and_process_data(dataset_path):
    X, y = [], []
    classes = os.listdir(dataset_path)
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                file_path = os.path.join(class_path, filename)
                try:
                    audio_data, sr = librosa.load(file_path, sr=None)
                    
                    # 1. Data asli
                    X.append(extract_mel_spectrogram(audio_data, sr))
                    y.append(class_name)
                    
                    # 2. Data augmentasi
                    for _ in range(AUGMENT_FACTOR):
                        aug_data = audio_data
                        if np.random.rand() > 0.5: aug_data = add_noise(aug_data)
                        if np.random.rand() > 0.5: aug_data = shift_time(aug_data, sr)
                        if np.random.rand() > 0.5: aug_data = change_pitch(aug_data, sr)
                        
                        X.append(extract_mel_spectrogram(aug_data, sr))
                        y.append(class_name)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return np.array(X), np.array(y)

# --- 5. Membangun Model CNN ---
def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'), MaxPooling2D((2, 2)), Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'), MaxPooling2D((2, 2)), Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'), Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main Execution ---
if __name__ == "__main__":
    # Load data
    X, y = load_and_process_data(DATASET_PATH)
    print(f"\nTotal samples after augmentation: {len(X)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Tambah channel dimension
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Build and train model
    model = build_cnn_model(X_train.shape[1:])
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint('audio_classifier_best.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train, epochs=100, batch_size=32,
        validation_data=(X_test, y_test), callbacks=callbacks
    )

    # Evaluasi
    best_model = tf.keras.models.load_model('audio_classifier_best.h5')
    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
