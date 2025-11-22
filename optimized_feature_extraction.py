"""
Script Optimized untuk Ekstraksi Fitur Multi-Modal Audio
- Support untuk 2 kelas (Normal dan Skizofrenia)
- Augmentasi yang lebih beragam
- Multi-feature extraction (Mel + MFCC + Chroma + Spectral)
"""

import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import joblib

# --- 1. KONFIGURASI ---
DATASET_PATH = "dataset_amino/"
OUTPUT_FILE = "processed_data_optimized.npz"

# Parameter Audio
SAMPLE_RATE = 22050
N_MELS = 128
N_MFCC = 40
FMAX = 8000
MAX_LEN = 216  # ~5 detik audio
HOP_LENGTH = 512

# Augmentasi
AUGMENT_FACTOR = 7  # Tingkatkan jumlah augmentasi untuk dataset kecil

# --- 2. FUNGSI AUGMENTASI YANG DITINGKATKAN ---
def add_noise(data, noise_factor_range=(0.002, 0.01)):
    """Tambah noise dengan variasi intensitas"""
    noise_factor = np.random.uniform(low=noise_factor_range[0], high=noise_factor_range[1])
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def shift_time(data, sr, shift_max_sec=0.3):
    """Time shifting dengan range yang lebih besar"""
    shift_range = int(sr * shift_max_sec)
    shift = np.random.randint(-shift_range, shift_range)
    return np.roll(data, shift)

def change_pitch(data, sr, pitch_factor_range=(-3, 3)):
    """Pitch shifting dengan range yang lebih lebar"""
    pitch_factor = np.random.uniform(low=pitch_factor_range[0], high=pitch_factor_range[1])
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

def change_speed(data, speed_factor_range=(0.85, 1.15)):
    """Time stretching untuk variasi kecepatan"""
    speed_factor = np.random.uniform(low=speed_factor_range[0], high=speed_factor_range[1])
    return librosa.effects.time_stretch(data, rate=speed_factor)

def add_reverb(data, sr):
    """Simple reverb effect menggunakan convolution"""
    # Buat impulse response sederhana
    impulse = np.zeros(int(sr * 0.1))
    impulse[0] = 1
    impulse[int(sr * 0.05)] = 0.3
    impulse[int(sr * 0.08)] = 0.1
    return np.convolve(data, impulse, mode='same')[:len(data)]

# --- 3. EKSTRAKSI FITUR MULTI-MODAL ---
def extract_features(y, sr):
    """
    Ekstrak multiple features untuk representasi audio yang lebih kaya
    Returns: Combined feature matrix
    """
    # 1. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 2. MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    
    # 3. Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    # 4. Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    # Combine all features
    # Stack semua fitur secara vertikal
    combined = np.vstack([mel_spec_db, mfcc, chroma, spectral_contrast])
    
    # Padding atau truncate untuk dimensi waktu yang konsisten
    if combined.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - combined.shape[1]
        combined = np.pad(combined, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :MAX_LEN]
    
    return combined

def extract_mel_only(y, sr):
    """
    Ekstrak hanya Mel Spectrogram (untuk kompatibilitas dengan model lama)
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_LEN]
    
    return mel_spec_db

# --- 4. PROSES PEMUATAN DAN AUGMENTASI DATA ---
def load_and_process_data(dataset_path, use_multi_features=False):
    """
    Load audio files dengan augmentasi yang agresif
    
    Args:
        dataset_path: Path ke dataset
        use_multi_features: True untuk multi-modal features, False untuk Mel saja
    """
    X, y = [], []
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"\n🎯 Kelas yang ditemukan: {classes}")
    print(f"📊 Mode ekstraksi: {'Multi-Features' if use_multi_features else 'Mel Spectrogram Only'}")
    print(f"🔄 Augmentasi per file: {AUGMENT_FACTOR}x\n")
    
    augmentation_techniques = [
        ("noise", add_noise),
        ("shift", lambda data, sr: shift_time(data, sr)),
        ("pitch", lambda data, sr: change_pitch(data, sr)),
        ("speed", lambda data, sr: change_speed(data)),
        ("reverb", lambda data, sr: add_reverb(data, sr))
    ]
    
    extract_fn = extract_features if use_multi_features else extract_mel_only
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(('.wav', '.mp3', '.ogg'))]
        
        for filename in tqdm(files, desc=f"Processing {class_name}"):
            file_path = os.path.join(class_path, filename)
            try:
                # Load audio dengan sample rate yang konsisten
                audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # 1. Data asli
                X.append(extract_fn(audio_data, sr))
                y.append(class_name)
                
                # 2. Data augmentasi dengan teknik yang beragam
                for i in range(AUGMENT_FACTOR):
                    aug_data = audio_data.copy()
                    
                    # Randomly apply 2-3 augmentation techniques
                    num_augs = np.random.randint(2, 4)
                    selected_augs = np.random.choice(len(augmentation_techniques), 
                                                    num_augs, replace=False)
                    
                    for aug_idx in selected_augs:
                        aug_name, aug_fn = augmentation_techniques[aug_idx]
                        aug_data = aug_fn(aug_data, sr)
                    
                    X.append(extract_fn(aug_data, sr))
                    y.append(class_name)
                    
            except Exception as e:
                print(f"\n❌ Error processing {file_path}: {e}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert to numpy array
    X_array = np.array(X)
    
    print(f"\n✅ Total samples: {len(X_array)}")
    print(f"📈 Distribution:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    for cls, count in zip(label_encoder.classes_, counts):
        print(f"   - {cls}: {count} samples")
    
    return X_array, y_encoded, label_encoder

# --- 5. COMPUTE CLASS WEIGHTS ---
def get_class_weights(y):
    """Hitung class weights untuk menangani imbalance"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return {i: weight for i, weight in enumerate(class_weights)}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED FEATURE EXTRACTION UNTUK KLASIFIKASI AUDIO")
    print("=" * 60)
    
    # Pilih mode ekstraksi fitur
    USE_MULTI_FEATURES = False  # Set True untuk multi-features, False untuk Mel only
    
    X, y_encoded, le = load_and_process_data(DATASET_PATH, USE_MULTI_FEATURES)
    
    print(f"\n📊 Bentuk fitur (X): {X.shape}")
    print(f"📊 Bentuk label (y): {y_encoded.shape}")
    print(f"🏷️  Kelas: {le.classes_}")
    
    # Hitung class weights
    class_weights = get_class_weights(y_encoded)
    print(f"\n⚖️  Class Weights: {class_weights}")
    
    # Simpan data yang telah diproses
    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y_encoded,
        classes=le.classes_,
        class_weights=class_weights
    )
    
    # Simpan label encoder
    joblib.dump(le, 'label_encoder_optimized.joblib')
    
    print(f"\n✅ Data berhasil diproses dan disimpan di {OUTPUT_FILE}")
    print(f"✅ Label encoder disimpan di label_encoder_optimized.joblib")
    print("=" * 60)
