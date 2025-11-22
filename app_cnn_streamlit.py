import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import io
import pandas as pd
import matplotlib.pyplot as plt # <-- Import matplotlib
# import joblib # Kita tidak pakai joblib lagi untuk encoder

# --- Konfigurasi ---
# Pastikan nama file ini cocok dengan output skrip Anda
MODEL_PATH = "audio_classifier_best.h5"
PROCESSED_DATA_PATH = "processed_data_v2.npz"
# ENCODER_PATH = "label_encoder.joblib" # Tidak digunakan

N_MELS = 128
FMAX = 8000
MAX_LEN = 216

# --- Fungsi Helper ---
@st.cache_resource
def load_model_and_classnames(): # Nama fungsi disesuaikan
    """Memuat model dan nama kelas dari file .npz."""
    try:
        model = load_model(MODEL_PATH)
        # PERUBAHAN: Muat nama kelas dari file .npz
        with np.load(PROCESSED_DATA_PATH) as data:
            # Pastikan key 'classes' ada di file .npz Anda
            if 'classes' not in data:
                 raise KeyError("Kunci 'classes' tidak ditemukan di file processed_data_v2.npz. Pastikan skrip augmentasi menyimpannya.")
            class_names = data['classes']
        return model, class_names, None # Tidak mengembalikan encoder
    except FileNotFoundError:
         # Beri pesan error yang lebih spesifik
         st.error(f"Error: Pastikan '{MODEL_PATH}' dan '{PROCESSED_DATA_PATH}' ada.")
         return None, None, FileNotFoundError(f"File '{MODEL_PATH}' atau '{PROCESSED_DATA_PATH}' tidak ditemukan.")
    except KeyError as e:
         st.error(f"Error saat memuat nama kelas: {e}")
         return None, None, e
    except Exception as e:
         return None, None, e

def extract_mel_spectrogram(y, sr):
    """Fungsi ekstraksi fitur (tetap sama)."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_LEN]
    return mel_spec_db

# --- Tampilan Aplikasi Streamlit ---
st.set_page_config(page_title="Klasifikasi Gangguan Jiwa", layout="wide")

st.subheader("RSJD dr.Amino Gondohutomo")
st.title("Klasifikasi Gangguan Jiwa Berdasarkan Audio")
st.write("---")


# Muat model dan nama kelas
with st.spinner("Mempersiapkan model AI..."):
    # PERUBAHAN: Nama variabel disesuaikan
    model, class_names, load_error = load_model_and_classnames()


if load_error :
    st.error("Gagal memuat model/data kelas. Berikut adalah detail error teknisnya:")
    st.exception(load_error)
else:
    st.success("Model berhasil dimuat! Aplikasi siap digunakan.")

    st.write("Unggah file audio (.wav) untuk memprediksi kategori (Normal, Bipolar, atau Skizofrenia).")
    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Analisis Audio"):
            with st.spinner("Mengekstrak fitur dan melakukan prediksi..."):

                audio_bytes = uploaded_file.read()
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
                spectrogram = extract_mel_spectrogram(audio_data, sr)
                features_for_model = spectrogram[np.newaxis, ..., np.newaxis]

                # Lakukan prediksi
                prediction_proba = model.predict(features_for_model)[0]
                predicted_index = np.argmax(prediction_proba)
                # PERUBAHAN: Gunakan list class_names langsung
                predicted_class = class_names[predicted_index]
                confidence = prediction_proba[predicted_index] * 100

                st.subheader("Hasil Prediksi:")
                st.markdown(
                    f"""
                    <div style="background-color: #ccc; padding: 10px; border-radius: 5px;">
                        <h1 style="text-align: center; color: #004085;">{predicted_class.upper()}</h1>
                    </div>
                    <br>
                    """,
                    unsafe_allow_html=True
                )
                st.success(f"**Tingkat Keyakinan:** {confidence:.2f}%")

                st.divider()

                # --- PENAMBAHAN: Menampilkan detail probabilitas dengan Bar Chart Horizontal ---
                st.subheader("Detail Probabilitas:")
                # PERUBAHAN: Gunakan class_names sebagai index DataFrame
                prob_df = pd.DataFrame(prediction_proba, index=class_names, columns=['Probabilitas'])

                fig_bar, ax_bar = plt.subplots(figsize=(8, 4)) # Ukuran proporsional
                # Buat bar chart horizontal
                bars = ax_bar.barh(prob_df.index, prob_df['Probabilitas'], color=['#8da0cb', '#fc8d62', '#66c2a5'])
                ax_bar.set_xlabel("Probabilitas")
                ax_bar.set_title("Probabilitas Prediksi")
                ax_bar.set_xlim(0, 1) # Set limit sumbu x dari 0% hingga 100%
                # Urutkan dari probabilitas tertinggi ke terendah
                ax_bar.invert_yaxis()

                # Tambahkan label persentase pada setiap bar
                for bar in bars:
                    width = bar.get_width()
                    # Tempatkan teks persentase di ujung bar
                    ax_bar.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', va='center')

                # Tampilkan chart di Streamlit
                st.pyplot(fig_bar)
                # --------------------------------------------------------------------------

