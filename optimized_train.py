"""
Optimized Training Script dengan K-Fold Cross Validation

Features:
- K-Fold Cross Validation untuk evaluasi yang robust
- ReduceLROnPlateau untuk adaptive learning rate
- Class weights untuk handling imbalance
- Extensive metrics dan visualisasi
- Model checkpointing yang lebih baik
- ROC-AUC curve visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.models import load_model
import datetime
import os

# Import model builders
from optimized_cnn_model import build_simple_cnn, build_deep_cnn, build_attention_cnn

# --- KONFIGURASI ---
PROCESSED_DATA_FILE = "processed_data_optimized.npz"
MODEL_SAVE_PATH = "models/best_model_optimized.h5"
LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Training parameters
EPOCHS = 150
BATCH_SIZE = 16  # Lebih kecil untuk dataset kecil
VALIDATION_SPLIT = 0.2
USE_KFOLD = False  # Set True untuk K-Fold CV
K_FOLDS = 5

# Model selection: 'simple', 'deep', or 'attention'
MODEL_TYPE = 'deep'

# --- HELPER FUNCTIONS ---
def plot_training_history(history, fold=None):
    """Plot training history dengan lebih detail"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    fold_str = f"_fold{fold}" if fold is not None else ""
    plt.savefig(f'visualizations/training_history{fold_str}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, fold=None):
    """Plot confusion matrix dengan percentage"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual Label')
    ax1.set_xlabel('Predicted Label')
    
    # Percentage matrix
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Actual Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    fold_str = f"_fold{fold}" if fold is not None else ""
    plt.savefig(f'visualizations/confusion_matrix{fold_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_roc_curve(y_true, y_pred_proba, fold=None):
    """Plot ROC curve untuk binary classification"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    fold_str = f"_fold{fold}" if fold is not None else ""
    plt.savefig(f'visualizations/roc_curve{fold_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba, fold=None):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    fold_str = f"_fold{fold}" if fold is not None else ""
    plt.savefig(f'visualizations/pr_curve{fold_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pr_auc

def evaluate_model(model, X_test, y_test, class_names, fold=None):
    """Comprehensive model evaluation"""
    # Predict
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Untuk binary classification
    if y_pred_proba.shape[1] == 1 or len(y_pred_proba.shape) == 1:
        y_pred_proba = y_pred_proba.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS (Binary Classification)")
        print("="*60)
        
        # Classification report
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
        
        # Confusion matrix
        print("\n--- Confusion Matrix ---")
        cm = plot_confusion_matrix(y_test, y_pred, class_names, fold)
        
        # ROC-AUC
        print("\n--- ROC-AUC Analysis ---")
        roc_auc_score = plot_roc_curve(y_test, y_pred_proba, fold)
        print(f"ROC-AUC Score: {roc_auc_score:.4f}")
        
        # Precision-Recall
        print("\n--- Precision-Recall Analysis ---")
        pr_auc_score = plot_precision_recall_curve(y_test, y_pred_proba, fold)
        print(f"PR-AUC Score: {pr_auc_score:.4f}")
        
        return {
            'confusion_matrix': cm,
            'roc_auc': roc_auc_score,
            'pr_auc': pr_auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    else:
        # Multi-class classification
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
        
        cm = plot_confusion_matrix(y_test, y_pred, class_names, fold)
        
        return {
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

def get_callbacks(fold=None):
    """Setup callbacks untuk training"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    fold_str = f"_fold{fold}" if fold is not None else ""
    model_path = f"models/best_model{fold_str}.h5"
    
    callbacks = [
        # Early stopping dengan patience yang cukup
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            verbose=1,
            restore_best_weights=True
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=LOG_DIR + fold_str,
            histogram_freq=1
        ),
        
        # CSV logger
        CSVLogger(f'logs/training{fold_str}.csv')
    ]
    
    return callbacks

# --- MAIN TRAINING ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPTIMIZED TRAINING SCRIPT - BINARY CLASSIFICATION")
    print("="*80)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load processed data (with allow_pickle for numpy 2.x compatibility)
    print("\n📂 Loading processed data...")
    with np.load(PROCESSED_DATA_FILE, allow_pickle=True) as data:
        X = data['X']
        y = data['y']
        class_names = [str(c) for c in data['classes']]  # Convert to regular strings
        class_weights = data['class_weights'] if 'class_weights' in data else None
    
    print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")
    print(f"🏷️  Classes: {class_names}")
    
    # Add channel dimension
    if len(X.shape) == 3:
        X = X[..., np.newaxis]
    
    # Compute class weights manually (avoid loading from npz due to numpy 2.x issues)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_array)}
    print(f"⚖️  Class weights: {class_weight_dict}")
    
    # Model selection
    print(f"\n🏗️  Building model: {MODEL_TYPE.upper()}")
    
    model_builders = {
        'simple': build_simple_cnn,
        'deep': build_deep_cnn,
        'attention': build_attention_cnn
    }
    
    build_fn = model_builders.get(MODEL_TYPE, build_deep_cnn)
    
    if USE_KFOLD:
        print(f"\n🔄 Using {K_FOLDS}-Fold Cross Validation")
        
        kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            print(f"\n{'='*80}")
            print(f"📊 FOLD {fold}/{K_FOLDS}")
            print(f"{'='*80}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build model
            model = build_fn(X_train.shape[1:], num_classes=2)
            
            # Train
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                class_weight=class_weight_dict,
                callbacks=get_callbacks(fold),
                verbose=1
            )
            
            # Plot history
            plot_training_history(history, fold)
            
            # Evaluate
            results = evaluate_model(model, X_val, y_val, class_names, fold)
            fold_results.append(results)
        
        print("\n" + "="*80)
        print("📊 CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        if 'roc_auc' in fold_results[0]:
            roc_aucs = [r['roc_auc'] for r in fold_results]
            print(f"Average ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
        
    else:
        print(f"\n📊 Using Train-Test Split ({int(VALIDATION_SPLIT*100)}% validation)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
        )
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Build model
        model = build_fn(X_train.shape[1:], num_classes=2)
        model.summary()
        
        # Train
        print("\n🚀 Starting training...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            class_weight=class_weight_dict,
            callbacks=get_callbacks(),
            verbose=1
        )
        
        # Plot history
        plot_training_history(history)
        
        # Load best model and evaluate
        print("\n📊 Evaluating best model...")
        best_model = load_model('models/best_model.h5')
        
        loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Test Accuracy: {accuracy:.4f}")
        print(f"✅ Test Loss: {loss:.4f}")
        
        # Comprehensive evaluation
        results = evaluate_model(best_model, X_test, y_test, class_names)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"\n📁 Model saved to: models/")
    print(f"📁 Visualizations saved to: visualizations/")
    print(f"📁 Logs saved to: logs/")
    print(f"\n💡 To view TensorBoard: tensorboard --logdir={LOG_DIR}")
    print("="*80)
