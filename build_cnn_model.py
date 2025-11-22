from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten dan Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax') # 3 kelas: normal, bipolar, schizophrenia
    ])
    
    model.summary()
    return model

# Contoh cara memanggil fungsi (input_shape akan ditentukan saat data diproses)
# INPUT_SHAPE = (128, 216, 1) # (n_mels, max_len, 1 channel)
# model = build_cnn_model(INPUT_SHAPE)
