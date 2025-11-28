# 🎙️ CNN Audio Classification - Schizophrenia Detection

**Deep Learning-based Audio Classification System for Mental Health Screening**

Developed for RSJD dr. Amino Gondohutomo

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Validation Accuracy** | 93.48% | ✅ Excellent |
| **Precision** | 87.50% | ✅ Very Good |
| **Recall** | 100.00% | 🌟 Perfect |
| **ROC-AUC** | 99.90% | 🏆 Outstanding |

**No false negatives** - Critical for medical screening applications!

---

## 🎯 Features

### 🔐 Security & Authentication (NEW!)
- ✅ **User Authentication**: Secure login system with PBKDF2-SHA256 password hashing
- ✅ **Role-Based Access Control**: Admin, Doctor, and Staff roles
- ✅ **Session Management**: 30-minute auto-logout for security
- ✅ **Audit Logging**: Complete login history tracking
- ✅ **Admin Panel**: User management dashboard

### 🤖 Machine Learning
- ✅ **Binary Classification**: Normal vs Skizofrenia
- ✅ **Deep CNN Architecture**: 4 convolutional blocks with batch normalization
- ✅ **Advanced Data Augmentation**: 5 techniques (noise, pitch shift, time stretch, reverb, time shift)
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC, ROC
- ✅ **Production Ready**: Early stopping, model checkpointing, adaptive learning rate

### 🎨 User Interface
- ✅ **Modern Web Interface**: Streamlit app with gauge meters and interactive charts
- ✅ **Clean Login Page**: Professional authentication UI
- ✅ **Real-time Visualization**: Audio waveform and spectrogram
- ✅ **Responsive Design**: Works on desktop and tablets

### 📱 Mobile Deployment
- ✅ **TensorFlow Lite**: Optimized model (1.24MB quantized)
- ✅ **Flutter Guide**: Complete offline mobile app documentation
- ✅ **Cross-platform**: Android & iOS support

---

## 🏗️ Architecture

### Model Overview
- **Type**: Deep Convolutional Neural Network (CNN)
- **Input**: Mel Spectrogram (128 x 216)
- **Parameters**: ~3M
- **Layers**: 
  - 4 Conv2D blocks (32 → 64 → 128 → 256 filters)
  - Batch Normalization after each layer
  - Dropout for regularization
  - Global Average Pooling
  - Dense layers with sigmoid output

### Data Pipeline
```
Raw Audio (.wav/.mp3/.ogg)
    ↓
Librosa Feature Extraction (Mel Spectrogram)
    ↓
Data Augmentation (7x multiplication)
    ├─ Add Noise
    ├─ Time Shift
    ├─ Pitch Shift
    ├─ Time Stretch
    └─ Reverb
    ↓
CNN Model (Binary Classification)
    ↓
Prediction (Normal / Skizofrenia)
```

---

## 📁 Project Structure

```
CNN_amino/
│
├── 📜 SCRIPTS (Optimized v2.0)
│   ├── optimized_feature_extraction.py  # Feature extraction + augmentation
│   ├── optimized_cnn_model.py           # 3 CNN architectures (Simple/Deep/Attention)
│   ├── optimized_train.py               # Training script with K-Fold CV
│   ├── app_optimized.py                 # Streamlit web application (with auth)
│   └── setup_auth.py                    # Authentication setup script
│
├── 🔐 AUTHENTICATION
│   ├── auth/
│   │   ├── __init__.py                  # Auth package
│   │   ├── authenticator.py             # Login/logout logic
│   │   ├── password_utils.py            # Password hashing
│   │   ├── database.py                  # SQLite operations
│   │   └── users.db                     # User credentials (NOT in git)
│   └── AUTH_README.md                   # Authentication documentation
│
├── 📚 DOCUMENTATION
│   ├── README.md                        # This file
│   ├── AUTH_README.md                   # Authentication guide
│   ├── FLUTTER_MOBILE_GUIDE.md          # Mobile deployment guide
│   ├── PROJECT_CHECKPOINT.md            # Project continuity doc
│   └── ANALYSIS_REPORT.md               # Technical analysis
│
├── 🛠️ UTILITIES
│   ├── compare_models.py                # Model benchmarking tool
│   ├── show_results.py                  # Training results viewer
│   ├── check_data.py                    # Data statistics
│   ├── convert_to_tflite.py             # TFLite model converter
│   └── QUICK_START.py                   # Interactive guide
│
├── ⚙️ CONFIGURATION
│   ├── requirements_optimized.txt       # Python dependencies
│   └── .gitignore                       # Git ignore rules
│
└── 📂 DATA (Not included in repo)
    ├── dataset_amino/                   # Raw audio files
    ├── models/                          # Trained models (.h5, .tflite)
    ├── processed_data_optimized.npz     # Processed features
    ├── visualizations/                  # Training graphs
    ├── logs/                            # Training & login logs
    └── backups/                         # Database backups

```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/mastoro786/CNN_amino.git
cd CNN_amino
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements_optimized.txt
```

### 3. Prepare Dataset

Place your audio files in the following structure:

```
dataset_amino/
├── normal/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── skizofrenia/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

### 4. Extract Features

```bash
python optimized_feature_extraction.py
```

**Output**: `processed_data_optimized.npz` (~145 MB)

### 5. Setup Authentication (NEW!)

```bash
# Initialize database and create default users
python setup_auth.py
```

**Default Credentials** (⚠️ Change immediately after first login!):
| Username | Password | Role |
|----------|----------|------|
| `admin` | `Admin123` | Administrator |
| `dr_amino` | `Doctor123` | Doctor |
| `staff1` | `Staff123` | Staff |

📖 **Full Authentication Guide**: See [AUTH_README.md](AUTH_README.md)

### 6. Train Model

```bash
python optimized_train.py
```

**Configuration** (edit in script):
```python
MODEL_TYPE = 'deep'      # 'simple', 'deep', or 'attention'
EPOCHS = 150
BATCH_SIZE = 16
```

**Output**: 
- `models/best_model.h5` - Trained model
- `visualizations/` - Training graphs
- `logs/` - Training logs

### 7. Run Web Application

```bash
streamlit run app_optimized.py
```

**Access**: http://localhost:8501

**Login** with the credentials above (change password after first login!)

### 8. Convert to TFLite (Optional - For Mobile)

---

## 📊 Model Architectures

### Simple CNN (800K params)
- Fast training (~10-15 min)
- Good for small datasets
- Expected accuracy: 88-92%

### Deep CNN (3M params) ⭐ **Recommended**
- Best performance
- Production-ready quality
- Expected accuracy: 92-96%
- **Achieved: 93.48%**

### Attention CNN (2M params)
- Experimental
- Attention mechanism
- Expected accuracy: 90-94%

---

## 🔧 Advanced Usage

### K-Fold Cross Validation

For robust evaluation:

```python
# Edit optimized_train.py
USE_KFOLD = True
K_FOLDS = 5
```

### Multi-Modal Features

Extract richer features:

```python
# Edit optimized_feature_extraction.py
USE_MULTI_FEATURES = True
```

Features extracted:
- Mel Spectrogram (128 bands)
- MFCC (40 coefficients)
- Chroma (12 bins)
- Spectral Contrast (7 bands)

---

## 📈 Training Results

### Convergence
- **Best Epoch**: 34/62
- **Early Stopping**: Worked perfectly
- **No Overfitting**: -2.37% gap (validation > training)

### Metrics Evolution
```
Initial  → Final
━━━━━━━━━━━━━━━
Accuracy:  53% → 93.48%
Loss:      2.5 → 1.42
Precision: 40% → 87.50%
Recall:    69% → 100%
```

### Visualizations

Training graphs saved in `visualizations/`:
- `training_history.png` - Loss & accuracy curves
- `confusion_matrix.png` - Error analysis
- `roc_curve.png` - ROC-AUC curve
- `pr_curve.png` - Precision-Recall curve

---

## 💻 Web Application Features

### Modern UI
- 🎨 Gradient header with purple theme
- 📊 **Gauge meter** showing confidence score
- 📈 **Horizontal bar chart** for class probabilities
- 🎵 Waveform and spectrogram visualization
- 📱 Responsive design

### Prediction Display
- Color-coded results (Green = Normal, Red = Skizofrenia)
- Confidence percentage
- Detailed interpretation based on confidence level
- Audio statistics (duration, RMS energy, etc.)

---

## 🔬 Technical Details

### Data Augmentation

5 techniques applied randomly (2-3 per sample):

1. **Add Noise**: Gaussian noise (0.2-1%)
2. **Time Shift**: ±0.3 seconds
3. **Pitch Shift**: ±3 semitones
4. **Time Stretch**: 0.85-1.15x speed
5. **Reverb**: Simple impulse response

**Result**: 109 files → 226 samples (after augmentation)

### Training Strategy

- **Optimizer**: Adam (LR=0.0001)
- **Loss**: Binary Crossentropy
- **Batch Size**: 16
- **Early Stopping**: Patience=25 epochs
- **ReduceLROnPlateau**: Factor=0.5, Patience=10
- **Class Weights**: Auto-computed for imbalance

### Regularization

- Dropout: 0.25 → 0.5 (progressive)
- Batch Normalization: After each conv layer
- L2 Regularization: 0.001
- Global Average Pooling

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{cnn_audio_schizophrenia,
  title={CNN-based Audio Classification for Schizophrenia Detection},
  author={RSJD dr. Amino Gondohutomo},
  year={2025},
  url={https://github.com/mastoro786/CNN_amino}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **RSJD dr. Amino Gondohutomo** for providing the dataset and domain expertise
- **TensorFlow/Keras** team for the deep learning framework
- **Librosa** team for audio processing capabilities
- **Streamlit** team for the amazing web framework

---

## 📞 Contact

For questions or collaborations:

- **Email**: [Your email]
- **Institution**: RSJD dr. Amino Gondohutomo
- **GitHub**: [@mastoro786](https://github.com/mastoro786)

---

## ⚠️ Disclaimer

**IMPORTANT**: This system is designed as a **screening tool only**. The predictions should **NOT** be used as a sole basis for clinical diagnosis. Always consult with qualified mental health professionals for accurate diagnosis and treatment.

---

<div align="center">

**Built with ❤️ using Python, TensorFlow, and Streamlit**

⭐ Star this repo if you find it helpful!

</div>
