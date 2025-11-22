"""
Optimized Streamlit App untuk Klasifikasi Audio (Normal vs Skizofrenia)

Features:
- Modern, clean UI with Gauge Meter and Horizontal Bar Chart
- Real-time audio visualization
- Detailed prediction with confidence scores
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import io
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- KONFIGURASI ---
MODEL_PATH = "models/best_model.h5"
PROCESSED_DATA_PATH = "processed_data_optimized.npz"

N_MELS = 128
FMAX = 8000
MAX_LEN = 216
SAMPLE_RATE = 22050

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model_and_info():
    """Load model dan informasi kelas"""
    try:
        model = load_model(MODEL_PATH)
        
        with np.load(PROCESSED_DATA_PATH, allow_pickle=True) as data:
            if 'classes' not in data:
                raise KeyError("Key 'classes' tidak ditemukan dalam processed data")
            class_names = [str(c) for c in data['classes']]
        
        return model, class_names, None
    except FileNotFoundError as e:
        return None, None, f"File tidak ditemukan: {e}"
    except Exception as e:
        return None, None, f"Error: {e}"

def extract_mel_spectrogram(y, sr):
    """Ekstrak mel spectrogram dari audio"""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_LEN]
    
    return mel_spec_db

def plot_waveform(y, sr):
    """Plot waveform audio"""
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_spectrogram(mel_spec_db, sr):
    """Plot mel spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        mel_spec_db, x_axis='time', y_axis='mel',
        sr=sr, fmax=FMAX, ax=ax, cmap='viridis'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    return fig

def create_gauge_meter(confidence, predicted_class):
    """Buat gauge meter untuk menampilkan confidence score"""
    # Determine color based on class
    if predicted_class.lower() == 'normal':
        color = "#28a745"  # Green
    else:
        color = "#dc3545"  # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>Confidence Score</b><br><span style='font-size:0.9em;color:{color}'>{predicted_class.upper ()}</span>", 
                 'font': {'size': 18}},
        number = {'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffcccc'},
                {'range': [60, 80], 'color': '#ffffcc'},
                {'range': [80, 100], 'color': '#ccffcc' if predicted_class.lower() == 'normal' else '#ffcccc'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=80, b=10),
        font={'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_horizontal_bar_chart(prediction_proba, class_names):
    """Buat horizontal bar chart untuk probabilitas kedua kelas"""
    colors = ['#28a745', '#dc3545']  # Green for normal, Red for schizophrenia
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=class_names,
        x=prediction_proba,
        orientation='h',
        text=[f'{p:.1%}' for p in prediction_proba],
        textposition='outside',
        textfont=dict(size=16, color='black', family='Arial Black'),
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.4)', width=2)
        ),
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Class Probabilities</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        xaxis_title='<b>Probability</b>',
        yaxis_title='',
        xaxis=dict(range=[0, 1.1], tickformat='.0%', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(tickfont=dict(size=14, family='Arial Black')),
        height=350,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=20, r=80, t=80, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white"
    )
    
    return fig

def get_interpretation(predicted_class, confidence):
    """Generate interpretation berdasarkan prediksi"""
    interpretations = {
        'normal': {
            'high': (
                "✅ **Hasil menunjukkan pola audio NORMAL dengan confidence tinggi.** "
                "Audio menunjukkan karakteristik bicara yang normal tanpa indikasi gangguan jiwa yang signifikan."
            ),
            'medium': (
                "✅ **Hasil menunjukkan pola audio NORMAL dengan confidence moderat.** "
                "Sebagian besar karakteristik audio menunjukkan pola normal, namun terdapat beberapa variasi yang perlu diperhatikan."
            ),
            'low': (
                "⚠️ **Hasil menunjukkan kecenderungan NORMAL namun dengan confidence rendah.** "
                "Model kurang yakin dengan prediksi ini. Disarankan untuk melakukan evaluasi lebih lanjut atau menggunakan audio yang lebih jelas."
            )
        },
        'skizofrenia': {
            'high': (
                "⚠️ **Hasil menunjukkan indikasi SKIZOFRENIA dengan confidence tinggi.** "
                "Audio menunjukkan karakteristik pola bicara yang konsisten dengan gejala skizofrenia. "
                "Sangat disarankan untuk melakukan evaluasi klinis lebih lanjut oleh profesional kesehatan mental."
            ),
            'medium': (
                "⚠️ **Hasil menunjukkan indikasi SKIZOFRENIA dengan confidence moderat.** "
                "Terdapat beberapa karakteristik audio yang menunjukkan kemungkinan adanya gejala skizofrenia. "
                "Disarankan untuk melakukan evaluasi lebih lanjut."
            ),
            'low': (
                "⚠️ **Hasil menunjukkan kecenderungan SKIZOFRENIA namun dengan confidence rendah.** "
                "Model kurang yakin dengan prediksi ini. Diperlukan evaluasi tambahan atau audio dengan kualitas yang lebih baik."
            )
        }
    }
    
    if confidence >= 0.80:
        level = 'high'
    elif confidence >= 0.60:
        level = 'medium'
    else:
        level = 'low'
    
    return interpretations[predicted_class][level]

# --- STREAMLIT APP ---
st.set_page_config(
    page_title="Klasifikasi Gangguan Jiwa - Audio Analysis",
    page_icon="🎙️",
    layout="wide"
)

# Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🎙️ Klasifikasi Gangguan Jiwa Berdasarkan Audio
        </h1>
        <p style="color: white; text-align: center; margin-top: 0.5rem; font-size: 1.1rem;">
            RSJD dr. Amino Gondohutomo
        </p>
    </div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("⏳ Memuat model AI..."):
    model, class_names, error = load_model_and_info()

if error:
    st.error(f"❌ Gagal memuat model: {error}")
    st.stop()

st.success("✅ Model berhasil dimuat dan siap digunakan!")

# Sidebar Info
with st.sidebar:
    st.markdown("### ℹ️ Informasi Aplikasi")
    st.markdown(f"""
    - **Model**: CNN Deep Learning
    - **Kelas**: {len(class_names)} ({', '.join(class_names)})
    - **Input**: File audio (.wav, .mp3, .ogg)
    - **Sample Rate**: {SAMPLE_RATE} Hz
    - **Accuracy**: 93.48%
    - **Recall**: 100%
    """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Cara Penggunaan")
    st.markdown("""
    1. Upload file audio
    2. Klik tombol **Analisis Audio**
    3. Tunggu hasil prediksi
    4. Lihat interpretasi hasil
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "Hasil prediksi ini **HANYA** sebagai alat bantu screening awal. "
        "Diagnosis pasti harus dilakukan oleh profesional kesehatan mental yang berkualifikasi."
    )

# Main content
st.markdown("### 📤 Upload Audio")
st.markdown("Silakan upload file audio untuk dianalisis. Format yang didukung: WAV, MP3, OGG")

uploaded_file = st.file_uploader(
    "Pilih file audio...",
    type=["wav", "mp3", "ogg"],
    help="Upload file audio dengan durasi maksimal 30 detik untuk hasil optimal"
)

if uploaded_file is not None:
    # Audio player
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
    
    with col2:
        st.markdown(f"""
        **File Info:**
        - Nama: `{uploaded_file.name}`
        - Ukuran: `{uploaded_file.size / 1024:.2f} KB`
        """)
    
    st.markdown("---")
    
    if st.button("🔍 Analisis Audio", type="primary", use_container_width=True):
        with st.spinner("🔄 Memproses audio..."):
            try:
                # Load audio
                audio_bytes = uploaded_file.read()
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
                
                # Extract features
                spectrogram = extract_mel_spectrogram(audio_data, sr)
                features_for_model = spectrogram[np.newaxis, ..., np.newaxis]
                
                # Predict
                prediction_proba = model.predict(features_for_model, verbose=0)[0]
                
                # Handle binary classification
                if len(prediction_proba.shape) == 0 or prediction_proba.shape[0] == 1:
                    # Binary with sigmoid output
                    prob_positive = float(prediction_proba) if prediction_proba.shape == () else float(prediction_proba[0])
                    prediction_proba = np.array([1 - prob_positive, prob_positive])
                
                predicted_index = np.argmax(prediction_proba)
                predicted_class = class_names[predicted_index]
                confidence = prediction_proba[predicted_index]
                
                # Results section
                st.markdown("---")
                st.markdown("## 📊 Hasil Analisis")
                
                # **NEW: 2 COLUMNS LAYOUT - Gauge Meter (Left) + Horizontal Bar Chart (Right)**
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.plotly_chart(
                        create_gauge_meter(confidence, predicted_class),
                        use_container_width=True,
                        key="gauge"
                    )
                
                with viz_col2:
                    st.plotly_chart(
                        create_horizontal_bar_chart(prediction_proba, class_names),
                        use_container_width=True,
                        key="bar"
                    )
                
                # Interpretation
                st.markdown("### 💡 Interpretasi Hasil")
                interpretation = get_interpretation(predicted_class.lower(), confidence)
                st.info(interpretation)
                
                # Audio Visualizations
                st.markdown("---")
                st.markdown("### 🎵 Visualisasi Audio")
                
                tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
                
                with tab1:
                    fig_wave = plot_waveform(audio_data, sr)
                    st.pyplot(fig_wave)
                    
                    # Audio statistics
                    st.markdown("**Audio Statistics:**")
                    cols = st.columns(4)
                    cols[0].metric("Duration", f"{len(audio_data)/sr:.2f} s")
                    cols[1].metric("Sample Rate", f"{sr} Hz")
                    cols[2].metric("Max Amplitude", f"{np.max(np.abs(audio_data)):.3f}")
                    cols[3].metric("RMS Energy", f"{np.sqrt(np.mean(audio_data**2)):.3f}")
                
                with tab2:
                    fig_spec = plot_spectrogram(spectrogram, sr)
                    st.pyplot(fig_spec)
                
            except Exception as e:
                st.error(f"❌ Error saat memproses audio: {e}")
                st.exception(e)

else:
    # Placeholder
    st.info("👆 Silakan upload file audio untuk memulai analisis")
    
    st.markdown("---")
    st.markdown("### 🎯 Tentang Sistem")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        **🧠 Teknologi**
        - Deep Learning (CNN)
        - TensorFlow/Keras
        - Mel Spectrogram Features
        """)
    
    with cols[1]:
        st.markdown("""
        **📊 Performance**
        - Validation Accuracy: 93.48%
        - Precision: 87.50%
        - Recall: 100%
        - ROC-AUC: 99.90%
        """)
    
    with cols[2]:
        st.markdown("""
        **🎯 Aplikasi**
        - Screening awal
        - Monitoring pasien
        - Riset klinis
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>© 2025 RSJD dr. Amino Gondohutomo | Developed with ❤️ using Streamlit & Deep Learning</p>
    </div>
""", unsafe_allow_html=True)
