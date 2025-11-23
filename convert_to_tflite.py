"""
Script untuk convert model Keras (.h5) ke TensorFlow Lite (.tflite)
Optimized untuk mobile deployment di Flutter

Usage:
    python convert_to_tflite.py

Output:
    - audio_classifier.tflite (full precision, ~15MB)
    - audio_classifier_quantized.tflite (quantized, ~5MB) <- Recommended for mobile
"""

import tensorflow as tf
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = "models/best_model.h5"
OUTPUT_DIR = "models/mobile"
TFLITE_MODEL_NAME = "audio_classifier.tflite"
QUANTIZED_MODEL_NAME = "audio_classifier_quantized.tflite"

def convert_to_tflite(model_path, output_path, quantize=False):
    """
    Convert Keras model to TensorFlow Lite
    
    Args:
        model_path: Path to .h5 model
        output_path: Path to save .tflite model
        quantize: Enable quantization for smaller model size
    
    Returns:
        Path to converted model
    """
    print(f"\n{'='*70}")
    print(f"CONVERTING MODEL TO TENSORFLOW LITE")
    print(f"{'='*70}\n")
    
    # Load Keras model
    print("📂 Loading Keras model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!\n")
        
        # Print model summary
        print("📊 Model Architecture:")
        model.summary()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create converter
    print(f"\n🔧 Creating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("⚙️  Enabling quantization...")
        print("   - Type: Dynamic range quantization")
        print("   - Benefit: 4x smaller model size")
        print("   - Trade-off: Minimal accuracy loss (<1%)")
        
        # Dynamic range quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Float16 quantization (even smaller)
        # converter.target_spec.supported_types = [tf.float16]
    else:
        print("⚙️  Converting without quantization (full precision)...")
    
    # Set optimization flags
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops (if needed)
    ]
    
    # Allow custom ops if needed
    converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
    converter._experimental_lower_tensor_list_ops = False
    
    # Convert model
    print("\n🔄 Converting model...")
    print("   This may take a few minutes...\n")
    
    try:
        tflite_model = converter.convert()
        print("✅ Conversion successful!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    print(f"\n💾 Saving model to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file size
    size_bytes = os.path.getsize(output_path)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"\n✅ Model saved successfully!")
    print(f"📊 File size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    
    return output_path

def test_tflite_model(tflite_path):
    """
    Test TFLite model with random input to verify it works
    
    Args:
        tflite_path: Path to .tflite model
    
    Returns:
        True if test successful, False otherwise
    """
    print(f"\n{'='*70}")
    print("TESTING TFLITE MODEL")
    print(f"{'='*70}\n")
    
    try:
        # Load TFLite model
        print("📂 Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        print("✅ Model loaded!\n")
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("📊 Model Details:")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input type: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output type: {output_details[0]['dtype']}")
        
        # Create test input (random Mel spectrogram)
        input_shape = input_details[0]['shape']
        print(f"\n🧪 Creating test input with shape: {input_shape}")
        
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        print("🔮 Running inference...")
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n✅ Inference successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output value: {output}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Interpret output for binary classification
        if output.shape[-1] == 1:
            prob_class_1 = float(output[0][0])
            prob_class_0 = 1.0 - prob_class_1
            print(f"\n📊 Predicted Probabilities:")
            print(f"   Normal: {prob_class_0:.4f} ({prob_class_0*100:.2f}%)")
            print(f"   Skizofrenia: {prob_class_1:.4f} ({prob_class_1*100:.2f}%)")
            predicted_class = "Skizofrenia" if prob_class_1 > 0.5 else "Normal"
            print(f"   Prediction: {predicted_class}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_label_map():
    """Create label map file for Flutter app"""
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.txt")
    
    with open(label_map_path, 'w') as f:
        f.write("normal\n")
        f.write("skizofrenia\n")
    
    print(f"\n📋 Label map created: {label_map_path}")
    return label_map_path

if __name__ == "__main__":
    print("\n" + "🚀 " * 35)
    print("   TENSORFLOW LITE MODEL CONVERTER FOR FLUTTER")
    print("🚀 " * 35)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print("   Please train the model first using: python optimized_train.py")
        exit(1)
    
    print(f"📦 Input model: {MODEL_PATH}")
    print(f"   Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB\n")
    
    # =====================================================================
    # STEP 1: Convert to TFLite (Full Precision)
    # =====================================================================
    print("\n" + "="*70)
    print("STEP 1: Converting to TFLite (Full Precision)")
    print("="*70)
    
    tflite_path = os.path.join(OUTPUT_DIR, TFLITE_MODEL_NAME)
    result1 = convert_to_tflite(MODEL_PATH, tflite_path, quantize=False)
    
    if result1:
        test_tflite_model(result1)
    else:
        print("\n❌ Full precision conversion failed!")
    
    # =====================================================================
    # STEP 2: Convert to TFLite (Quantized) - RECOMMENDED FOR MOBILE
    # =====================================================================
    print("\n" + "="*70)
    print("STEP 2: Converting to TFLite (Quantized) - RECOMMENDED")
    print("="*70)
    
    quantized_path = os.path.join(OUTPUT_DIR, QUANTIZED_MODEL_NAME)
    result2 = convert_to_tflite(MODEL_PATH, quantized_path, quantize=True)
    
    if result2:
        test_tflite_model(result2)
    else:
        print("\n❌ Quantized conversion failed!")
    
    # =====================================================================
    # STEP 3: Create Label Map
    # =====================================================================
    print("\n" + "="*70)
    print("STEP 3: Creating Label Map")
    print("="*70)
    create_label_map()
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    
    print(f"\n📦 Generated Files:")
    
    if result1:
        size1 = os.path.getsize(result1) / (1024 * 1024)
        print(f"\n1️⃣ Full Precision Model:")
        print(f"   Path: {result1}")
        print(f"   Size: {size1:.2f} MB")
        print(f"   Use: For accuracy testing")
    
    if result2:
        size2 = os.path.getsize(result2) / (1024 * 1024)
        print(f"\n2️⃣ Quantized Model: ⭐ RECOMMENDED FOR MOBILE")
        print(f"   Path: {result2}")
        print(f"   Size: {size2:.2f} MB")
        print(f"   Size reduction: {((1 - size2/size1) * 100):.1f}%")
        print(f"   Use: For Flutter mobile app")
    
    print(f"\n3️⃣ Label Map:")
    print(f"   Path: {os.path.join(OUTPUT_DIR, 'label_map.txt')}")
    
    # =====================================================================
    # NEXT STEPS
    # =====================================================================
    print("\n" + "="*70)
    print("📱 NEXT STEPS FOR FLUTTER INTEGRATION")
    print("="*70)
    
    print(f"""
1. Copy files to Flutter project:
   
   cp {result2} <your_flutter_app>/assets/models/
   cp {os.path.join(OUTPUT_DIR, 'label_map.txt')} <your_flutter_app>/assets/models/

2. Add to pubspec.yaml:
   
   flutter:
     assets:
       - assets/models/audio_classifier_quantized.tflite
       - assets/models/label_map.txt

3. Follow the integration guide:
   
   See: FLUTTER_MOBILE_GUIDE.md

4. Test the model:
   
   Run the Flutter app and record a sample audio
    """)
    
    print("="*70)
    print("🎉 Happy coding!")
    print("="*70 + "\n")
