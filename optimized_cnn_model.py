"""
Optimized CNN Architecture untuk Binary Classification (Normal vs Skizofrenia)

Improvements:
- Batch Normalization untuk stabilitas training
- Deeper architecture dengan residual-like connections
- Dropout yang lebih strategis
- Global Average Pooling untuk mengurangi overfitting
- L2 Regularization
- Binary classification dengan sigmoid activation
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, GlobalAveragePooling2D, Dense, Add,
    Activation, Concatenate
)
from tensorflow.keras.regularizers import l2

def build_simple_cnn(input_shape, num_classes=2):
    """
    Simple CNN untuk binary classification (backward compatible)
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    # Dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def build_deep_cnn(input_shape, num_classes=2):
    """
    Deep CNN dengan Batch Normalization dan Residual-like connections
    Lebih powerful untuk dataset yang cukup besar
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    # Block 4 - Additional depth
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    # Global pooling instead of Flatten untuk reduce overfitting
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def build_attention_cnn(input_shape, num_classes=2):
    """
    CNN dengan attention mechanism sederhana
    Fokus pada fitur yang paling penting
    """
    inputs = Input(shape=input_shape)
    
    # Main CNN pathway
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Simple attention mechanism
    attention = Conv2D(128, (1, 1), activation='sigmoid', padding='same')(x)
    x = tf.keras.layers.Multiply()([x, attention])
    
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    # Additional depth
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    
    # Output
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Main function untuk testing
if __name__ == "__main__":
    # Test dengan input shape dari mel spectrogram
    INPUT_SHAPE = (128, 216, 1)  # (n_mels, max_len, channels)
    
    print("\n" + "="*60)
    print("TESTING MODEL ARCHITECTURES")
    print("="*60)
    
    print("\n1️⃣ Simple CNN Model:")
    print("-" * 60)
    model_simple = build_simple_cnn(INPUT_SHAPE, num_classes=2)
    model_simple.summary()
    
    print("\n2️⃣ Deep CNN Model:")
    print("-" * 60)
    model_deep = build_deep_cnn(INPUT_SHAPE, num_classes=2)
    model_deep.summary()
    
    print("\n3️⃣ Attention CNN Model:")
    print("-" * 60)
    model_attention = build_attention_cnn(INPUT_SHAPE, num_classes=2)
    model_attention.summary()
    
    print("\n" + "="*60)
    print("Model testing completed!")
    print("="*60)
