import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import io
import pandas as pd


# --- Konfigurasi ---
MODEL_PATH = "audio_classifier_best.h5"
PROCESSED_DATA_PATH = "processed_data.npz"
N_MELS = 128
FMAX = 8000
MAX_LEN = 216

# --- Fungsi Helper ---
# Cache model dan label untuk performa lebih baik
@st.cache_resource
def load_model_and_labels():
    model = load_model(MODEL_PATH)
    with np.load(PROCESSED_DATA_PATH) as data:
        class_names = data['classes']
    return model, class_names, None

def extract_mel_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_LEN]
    return mel_spec_db

# --- Tampilan Aplikasi Streamlit ---
st.title("Klasifikasi Gangguan Jiwa Berdasarkan Audio")
st.header("RSJD dr.Amino Gondohutomo")
st.write("---")
st.write("Unggah file audio (.wav) untuk memprediksi kategori (Normal, Bipolar, atau Skizofrenia).")


# Muat model dan label
# try:
with st.spinner("Mempersiapkan model AI..."):
    model, class_names, load_error = load_model_and_labels()
    
    
if load_error :
    st.error("Gagal memuat model. Berikut adalah detail error teknisnya:")
    st.exception(load_error)
else:
    st.success("Model berhasil dimuat! Aplikasi siap digunakan.")
    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Analisis Audio"):
            with st.spinner("Mengekstrak fitur dan melakukan prediksi..."):

        # Proses file audio yang diunggah
                audio_bytes = uploaded_file.read()
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # Ekstraksi fitur
                spectrogram = extract_mel_spectrogram(audio_data, sr)
        
        # Ubah bentuk untuk prediksi
                spectrogram_expanded = np.expand_dims(spectrogram, axis=0) # Tambah batch dimension
                spectrogram_expanded = np.expand_dims(spectrogram_expanded, axis=-1) # Tambah channel dimension
                features_for_model = spectrogram[np.newaxis, ..., np.newaxis]

        # Lakukan prediksi
                prediction = model.predict(spectrogram_expanded)
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_class = class_names[predicted_index]
                confidence = prediction[0][predicted_index] * 100

                prediction_proba = model.predict(features_for_model)[0]
                prediction_index = np.argmax(prediction_proba)
                prediction_text = class_names.inverse_transform([prediction_index])[0]
        
        # Tampilkan hasil
                #st.subheader("Hasil Prediksi:")
                #st.success(f"**Kelas Terdeteksi:** {predicted_class}")
                
                st.subheader("Hasil Prediksi:")
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 10px;border: 2px solid #73AD21;">
                        <h3 style="text-align: center; color: #00e04b">{predicted_class.upper()}</h3>
                    </div>
                    <br>
                    """,
                    unsafe_allow_html=True
                )
                st.success(f"**Tingkat Keyakinan:** {confidence:.2f}%")

                st.divider()
                 # st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                   
                