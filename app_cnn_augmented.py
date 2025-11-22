# Nama file: 3_app_streamlit.py
# Tugas: Membuat antarmuka web interaktif untuk menguji model yang sudah dilatih
#        pada file audio baru.

import streamlit as st
import numpy as np
import librosa
import librosa.display # <-- Import tambahan untuk visualisasi spectrogram
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Konfigurasi & Fungsi ---
MODEL_PATH = "audio_classifier_best_v2.h5"
ENCODER_PATH = "label_encoder.joblib"

# Konfigurasi ini harus SAMA PERSIS dengan saat pre-processing
N_MELS = 128
FMAX = 8000
MAX_LEN = 216

def extract_mel_spectrogram(y, sr):
    """Fungsi untuk mengekstrak fitur dari satu file audio."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_LEN]
    return mel_spec_db

@st.cache_resource
def load_app_models():
    """Memuat model dan encoder ke cache agar tidak di-load ulang."""
    try:
        model = load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder, None
    except Exception as e:
        return None, None, e

# --- 2. Antarmuka Streamlit ---
st.set_page_config(page_title="Klasifikasi Gangguan Jiwa", layout="wide")

st.subheader("RSJD. dr. Amino Gondohutomo")
st.title("Klasifikasi Gangguan Jiwa dengan CNN")
st.write("Unggah file audio (.wav) untuk diprediksi.")

# Memuat model saat aplikasi dimulai
with st.spinner("Mempersiapkan model AI..."):
    model, encoder, load_error = load_app_models()

if load_error:
    st.error("Gagal memuat model. Berikut adalah detail error teknisnya:")
    st.exception(load_error)
else:
    st.success("Model berhasil dimuat! Aplikasi siap digunakan.")
    
    uploaded_file = st.file_uploader("Pilih file audio (.wav)...", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analisis Audio"):
            with st.spinner("Mengekstrak fitur dan melakukan prediksi..."):
                audio_data, sr = librosa.load(uploaded_file, sr=None)
                
                spectrogram = extract_mel_spectrogram(audio_data, sr)
                features_for_model = spectrogram[np.newaxis, ..., np.newaxis]
                
                prediction_proba = model.predict(features_for_model)[0]
                prediction_index = np.argmax(prediction_proba)
                prediction_text = encoder.inverse_transform([prediction_index])[0]
                
                st.subheader("Hasil Prediksi:")
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-radius: 10px;border: 2px solid #73AD21;">
                        <h1 style="text-align: center; color: #00e04b">{prediction_text.upper()}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Visualisasi Spectrogram")
                    fig_spec, ax_spec = plt.subplots(figsize=(8, 5))
                    img = librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=FMAX, ax=ax_spec)
                    fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
                    ax_spec.set_title('Mel-frequency Spectrogram')
                    st.pyplot(fig_spec)

                with col2:
                    st.subheader("Detail Probabilitas:")
                    prob_df = pd.DataFrame(prediction_proba, index=encoder.classes_, columns=['Probabilitas'])
                    
                    # --- PERUBAHAN: Chart diubah menjadi doughnut chart ---
                    fig_doughnut, ax_doughnut = plt.subplots(figsize=(4, 4))
                    
                    labels = prob_df.index
                    sizes = prob_df['Probabilitas']
                    colors = ['#8da0cb', '#fc8d62', '#66c2a5'] # Skema warna
                    
                    # Buat pie chart utama
                    wedges, texts, autotexts = ax_doughnut.pie(
                        sizes, 
                        autopct='%1.1f%%', 
                        startangle=50, 
                        colors=colors,
                        pctdistance=0.80, # Jarak persentase dari pusat
                        textprops={'color':"w", 'weight':'bold'}
                    )
                    
                    # Gambar lingkaran putih di tengah untuk membuat efek doughnut
                    centre_circle = plt.Circle((0,0),0.60,fc='white')
                    fig_doughnut.gca().add_artist(centre_circle)
                    
                    # Pastikan lingkaran sempurna dan hilangkan label di dalam chart
                    ax_doughnut.axis('equal')  
                    plt.setp(texts, visible=False) # Sembunyikan label kelas di dalam

                    # Buat legenda di samping chart
                    legend_labels = [f'{l}: {s:.1%}' for l, s in zip(labels, sizes)]
                    ax_doughnut.legend(wedges, legend_labels,
                                      title="Probabilitas",
                                      loc="center left",
                                      bbox_to_anchor=(1, 0, 0.5, 1))

                    ax_doughnut.set_title("Proporsi Probabilitas Prediksi", size=12)
                    
                    st.pyplot(fig_doughnut)

