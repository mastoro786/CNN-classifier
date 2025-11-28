"""
Optimized Streamlit App untuk Klasifikasi Audio (Normal vs Skizofrenia)

Features:
- User Authentication & Role-based Access Control
- Modern, clean UI with Gauge Meter
- Real-time audio visualization
- Detailed prediction with confidence scores
- Audit logging
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

# Authentication
from auth.authenticator import get_authenticator

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RSJD Audio Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def create_simple_gauge(percentage, class_name):
    """Buat semi-circular gauge meter untuk satu kelas"""
    # Color based on class
    if 'skizo' in class_name.lower():
        gauge_color = "#FF6B6B"  # Red
    else:
        gauge_color = "#90EE90"  # Light green
    
    fig = go.Figure(go.Indicator(
        mode="gauge",  # Only gauge, no default number
        value=percentage * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 0,
                'tickcolor': "white",
                'showticklabels': False
            },
            'bar': {'color': gauge_color, 'thickness': 0.7},
            'bgcolor': "#E8E8D0",
            'borderwidth': 0,
            'shape': "angular",
            'threshold': {
                'line': {'color': "#2c3e50", 'width': 6},
                'thickness': 0.75,
                'value': percentage * 100
            }
        }
    ))
    
    # Add percentage text in center
    fig.add_annotation(
        text=f"<b>{percentage*100:.1f}%</b>",
        x=0.5, y=0.3,  # Position in center-bottom of gauge
        font=dict(size=36, color='white', family='Arial Black'),
        showarrow=False,
        xref="paper",
        yref="paper"
    )
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial Black"}
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

# ===================================================================
# AUTHENTICATION CHECK (FIRST!)
# ===================================================================

# Get authenticator instance
auth = get_authenticator()

# Check if authenticated
if not auth.is_authenticated():
    # Show CLEAN login page (NO sidebar, NO header)
    auth.show_login_page()
    st.stop()  # Stop execution - don't load anything else

# ===================================================================
# USER IS AUTHENTICATED - LOAD MAIN APPLICATION
# ===================================================================

# Header (ONLY shown after login)
st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🎙️ Klasifikasi Gangguan Jiwa Berdasarkan Audio
        </h1>
        <p style="color: white; text-align: center; margin-top: 0.5rem; font-size: 2.5rem;">
            RSJD dr. Amino Gondohutomo
        </p>
    </div>
""", unsafe_allow_html=True)

# Load model (ONLY after login)
with st.spinner("⏳ Memuat model AI..."):
    model, class_names, error = load_model_and_info()

if error:
    st.error(f"❌ Gagal memuat model: {error}")
    st.stop()

st.success("✅ Model berhasil dimuat dan siap digunakan!")

# Sidebar Info (ONLY after login)
with st.sidebar:
    # User info at top
    auth.show_user_info_sidebar()
    
    st.markdown("---")
    
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

# ===================================================================
# MAIN APPLICATION CONTENT
# ===================================================================

# Main content
st.markdown("### 📤 Upload Audio")
st.markdown("Silakan upload file audio untuk dianalisis. Format yang didukung: WAV, MP3, OGG")

uploaded_file = st.file_uploader(
    "Pilih file audio...",
    type=["wav", "mp3", "ogg"],
    help="Upload file audio dengan durasi maksimal 30 detik untuk hasil optimal"
)

if uploaded_file is not None:
    # Show loading animation while processing file
    with st.spinner("⏳ Memuat audio file..."):
        import time
        time.sleep(0.3)  # Small delay to show loading animation
    
    # Success indicator - Audio ready
    st.success("✅ Audio berhasil dimuat dan siap untuk dianalisis!")
    
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
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            import time
            
            # Step 1: Loading audio
            status_text.info("📂 Step 1/4: Loading audio file...")
            audio_bytes = uploaded_file.read()
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
            progress_bar.progress(25)
            time.sleep(0.2)
            
            # Step 2: Extracting features
            status_text.info("🎵 Step 2/4: Extracting Mel Spectrogram features...")
            spectrogram = extract_mel_spectrogram(audio_data, sr)
            features_for_model = spectrogram[np.newaxis, ..., np.newaxis]
            progress_bar.progress(50)
            time.sleep(0.2)
            
            # Step 3: Running AI model
            status_text.info("🤖 Step 3/4: Running CNN classification model...")
            prediction_proba = model.predict(features_for_model, verbose=0)[0]
            progress_bar.progress(75)
            time.sleep(0.2)
            
            # Step 4: Processing results
            status_text.info("📊 Step 4/4: Processing results...")
            
            # Handle binary classification
            if len(prediction_proba.shape) == 0 or prediction_proba.shape[0] == 1:
                # Binary with sigmoid output
                prob_positive = float(prediction_proba) if prediction_proba.shape == () else float(prediction_proba[0])
                prediction_proba = np.array([1 - prob_positive, prob_positive])
            
            predicted_index = np.argmax(prediction_proba)
            predicted_class = class_names[predicted_index]
            confidence = prediction_proba[predicted_index]
            
            progress_bar.progress(100)
            time.sleep(0.3)
            
            # Completion message
            status_text.success("✅ Analisis selesai! Menampilkan hasil...")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Results section
            st.markdown("---")
            
            # Header dengan hasil prediksi (Purple)
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #5B3A8C 0%, #3D2564 100%); 
                            padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
                    <h2 style="color: white; text-align: center; margin: 0; font-size: 2rem;">
                        HASIL ANALISA : {predicted_class.upper()} {confidence*100:.1f}%
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Calculate probabilities
            skizo_idx = [i for i, name in enumerate(class_names) if 'skizo' in name.lower()]
            normal_idx = [i for i, name in enumerate(class_names) if 'normal' in name.lower()]
            skizo_prob = prediction_proba[skizo_idx[0]] if skizo_idx else 0
            normal_prob = prediction_proba[normal_idx[0]] if normal_idx else 0
            
            # Two columns for gauges (NO CONTAINER - direct to dark background)
            col1, col2 = st.columns(2)
            
            with col1:
                # Title - WHITE TEXT
                st.markdown("""
                    <h3 style="text-align: center; color: white; margin-bottom: 10px; font-family: Arial Black;">
                        SKIZOFRENIA
                    </h3>
                """, unsafe_allow_html=True)
                
                # Gauge
                st.plotly_chart(
                    create_simple_gauge(skizo_prob, "skizofrenia"),
                    use_container_width=True,
                    key="gauge_skizo"
                )
            
            with col2:
                # Title - WHITE TEXT
                st.markdown("""
                    <h3 style="text-align: center; color: white; margin-bottom: 10px; font-family: Arial Black;">
                        NORMAL
                    </h3>
                """, unsafe_allow_html=True)
                
                # Gauge
                st.plotly_chart(
                    create_simple_gauge(normal_prob, "normal"),
                    use_container_width=True,
                    key="gauge_normal"
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
