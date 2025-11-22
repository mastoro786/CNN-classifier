import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. Konfigurasi ---
PROCESSED_DATA_FILE = "processed_data_v2.npz"
MODEL_SAVE_PATH = "audio_classifier_best_v2.h5"

# --- 2. Membangun Model CNN ---
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'), MaxPooling2D((2, 2)), Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'), MaxPooling2D((2, 2)), Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main Execution ---
if __name__ == "__main__":
    # Muat data yang sudah diproses
    with np.load(PROCESSED_DATA_FILE) as data:
        X = data['X']
        y = data['y']
        class_names = data['classes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tambah channel dimension untuk input CNN
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Build and train model
    model = build_cnn_model(X_train.shape[1:], len(class_names))
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train, epochs=100, batch_size=32,
        validation_data=(X_test, y_test), callbacks=callbacks
    )

    # Evaluasi menggunakan model terbaik yang disimpan
    best_model = load_model(MODEL_SAVE_PATH)
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
