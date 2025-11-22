"""
Quick Start Script - Panduan Cepat untuk Memulai

Script ini akan membantu Anda memilih workflow yang tepat
"""

import os

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_section(title):
    print(f"\n{'─'*80}")
    print(f"📌 {title}")
    print(f"{'─'*80}")

def main():
    print_header("🎙️ AUDIO CLASSIFICATION - QUICK START GUIDE")
    
    print("""
    Selamat datang di sistem klasifikasi audio untuk deteksi skizofrenia!
    
    Project ini memiliki 2 versi:
    
    📁 VERSI LAMA (Original - 3 kelas)
       - Script: augmentasi_ekstraksi_fitur.py, train_model.py, app_cnn_streamlit.py
       - Arsitektur: Simple CNN
       - Kelas: Normal, Bipolar, Skizofrenia
    
    ✨ VERSI BARU (Optimized - 2 kelas)
       - Script: optimized_feature_extraction.py, optimized_train.py, app_optimized.py
       - Arsitektur: 3 pilihan (Simple, Deep, Attention CNN)
       - Kelas: Normal, Skizofrenia
       - Features: Advanced augmentation, metrics lengkap, K-Fold CV
    """)
    
    print_section("🚀 WORKFLOW RECOMMENDED")
    
    print("""
    Untuk PROJECT BARU dengan 2 KELAS (Normal vs Skizofrenia):
    
    Step 1: Feature Extraction
    ──────────────────────────
    python optimized_feature_extraction.py
    
    Output: processed_data_optimized.npz
    Waktu: ~5-10 menit (tergantung jumlah file)
    
    
    Step 2: Model Training
    ──────────────────────
    python optimized_train.py
    
    Output: models/best_model.h5, visualizations/, logs/
    Waktu: ~15-30 menit (tergantung model yang dipilih)
    
    Konfigurasi yang bisa diubah di script:
    - MODEL_TYPE = 'simple' / 'deep' / 'attention'
    - EPOCHS = 150 (sesuaikan dengan kebutuhan)
    - BATCH_SIZE = 16 (reduce jika memory error)
    - USE_KFOLD = True/False (untuk cross validation)
    
    
    Step 3: Run Application
    ───────────────────────
    streamlit run app_optimized.py
    
    Aplikasi akan terbuka di browser (default: http://localhost:8501)
    
    """)
    
    print_section("📊 TIPS & RECOMMENDATIONS")
    
    print("""
    ✅ UNTUK DATASET KECIL (<500 samples):
       - Gunakan MODEL_TYPE = 'simple'
       - AUGMENT_FACTOR = 7-10
       - BATCH_SIZE = 8-16
       - USE_KFOLD = True (untuk evaluasi robust)
    
    ✅ UNTUK DATASET SEDANG (500-2000 samples):
       - Gunakan MODEL_TYPE = 'deep'
       - AUGMENT_FACTOR = 5-7
       - BATCH_SIZE = 16-32
       - USE_KFOLD = False (train-test split cukup)
    
    ✅ UNTUK DATASET BESAR (>2000 samples):
       - Gunakan MODEL_TYPE = 'deep' atau 'attention'
       - AUGMENT_FACTOR = 3-5
       - BATCH_SIZE = 32-64
       - USE_MULTI_FEATURES = True (optional)
    
    ⚠️ JIKA MODEL OVERFITTING (train acc >> val acc):
       - Tingkatkan dropout (0.5 → 0.6)
       - Tambah L2 regularization
       - Gunakan model 'simple'
       - Tambah augmentasi
    
    ⚠️ JIKA MODEL UNDERFITTING (train acc & val acc rendah):
       - Gunakan model 'deep' atau 'attention'
       - Tingkatkan epoch (150 → 200)
       - Reduce regularization
       - Check kualitas data
    """)
    
    print_section("🔧 COMMON COMMANDS")
    
    print("""
    # Activate virtual environment (jika menggunakan)
    .venv\\Scripts\\activate          # Windows
    source .venv/bin/activate       # Linux/Mac
    
    # Install dependencies
    pip install -r requirements_optimized.txt
    
    # Run feature extraction
    python optimized_feature_extraction.py
    
    # Run training
    python optimized_train.py
    
    # Run app
    streamlit run app_optimized.py
    
    # View TensorBoard (after training)
    tensorboard --logdir=logs/fit
    
    # Test model architectures
    python optimized_cnn_model.py
    """)
    
    print_section("📂 OUTPUT FILES")
    
    print("""
    After running the scripts, you'll have:
    
    processed_data_optimized.npz      → Extracted features
    label_encoder_optimized.joblib    → Label encoder
    models/best_model.h5              → Best trained model
    visualizations/
        ├── training_history.png      → Loss & accuracy curves
        ├── confusion_matrix.png      → Confusion matrix
        ├── roc_curve.png             → ROC curve
        └── pr_curve.png              → Precision-Recall curve
    logs/
        ├── training.csv              → Training metrics
        └── fit/                      → TensorBoard logs
    """)
    
    print_section("🎯 NEXT STEPS")
    
    print("""
    1. ✅ Pastikan dataset sudah ada di dataset_amino/
       - normal/ (folder berisi audio normal)
       - skizofrenia/ (folder berisi audio skizofrenia)
    
    2. ✅ Install dependencies
       pip install -r requirements_optimized.txt
    
    3. ✅ Run feature extraction
       python optimized_feature_extraction.py
    
    4. ✅ Run training
       python optimized_train.py
    
    5. ✅ Evaluate results
       - Check visualizations/ folder
       - Review training.csv
       - Run TensorBoard (optional)
    
    6. ✅ Test dengan aplikasi
       streamlit run app_optimized.py
    
    7. ✅ Deploy (optional)
       - Streamlit Cloud
       - Docker container
       - API endpoint
    """)
    
    print_section("❓ NEED HELP?")
    
    print("""
    📖 Baca dokumentasi lengkap: README_OPTIMIZED.md
    
    🐛 Common Issues:
       - Module not found → pip install -r requirements_optimized.txt
       - File not found → Check file paths dalam script
       - Memory error → Reduce BATCH_SIZE
       - Overfitting → Lihat tips di section RECOMMENDATIONS
    
    📧 Contact: [Your contact info here]
    """)
    
    print_header("🎉 You're Ready to Start!")
    
    print("\nRecommended first command:")
    print("  → python optimized_feature_extraction.py")
    print("\n")

if __name__ == "__main__":
    main()
