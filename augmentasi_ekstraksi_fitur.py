import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- 1. Konfigurasi ---
DATASET_PATH = "dataset_amino/" # GANTI DENGAN PATH DATASET ANDA
OUTPUT_FILE = "processed_data_v2.npz"
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
    pitch_factor = np.random.uniform(pitch_factor_range[0], pitch_factor_range[1])
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
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            file_path = os.path.join(class_path, filename)
            try:
                audio_data, sr = librosa.load(file_path, sr=None)
                
                # 1. Data asli
                X.append(extract_mel_spectrogram(audio_data, sr))
                y.append(class_name)
                
                # 2. Data augmentasi
                for _ in range(AUGMENT_FACTOR):
                    aug_data = audio_data.copy()
                    if np.random.rand() > 0.5: aug_data = add_noise(aug_data)
                    if np.random.rand() > 0.5: aug_data = shift_time(aug_data, sr)
                    if np.random.rand() > 0.5: aug_data = change_pitch(aug_data, sr)
                    
                    X.append(extract_mel_spectrogram(aug_data, sr))
                    y.append(class_name)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return np.array(X), y_encoded, label_encoder

# --- Main Execution ---
if __name__ == "__main__":
    X, y_encoded, le = load_and_process_data(DATASET_PATH)
    
    print(f"\nBentuk fitur (X): {X.shape}")
    print(f"Bentuk label (y): {y_encoded.shape}")
    print(f"Kelas: {le.classes_}")
    
    # Simpan data yang telah diproses
    np.savez(OUTPUT_FILE, X=X, y=y_encoded, classes=le.classes_)
    print(f"\nData berhasil diproses dan disimpan di {OUTPUT_FILE}")
