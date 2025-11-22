import numpy as np

# Load processed data
data = np.load('processed_data_optimized.npz')

print("\n" + "="*60)
print("FEATURE EXTRACTION RESULTS")
print("="*60)

print(f"\n📊 Shape X (features): {data['X'].shape}")
print(f"📊 Shape y (labels): {data['y'].shape}")
print(f"🏷️  Classes: {list(data['classes'])}")

# Distribution
unique, counts = np.unique(data['y'], return_counts=True)
print(f"\n📈 Sample Distribution:")
for cls, count in zip(data['classes'], counts):
    percentage = (count / len(data['y'])) * 100
    print(f"   - {cls}: {count} samples ({percentage:.1f}%)")

print(f"\n⚖️  Class Weights:")
class_weights = data['class_weights']
for cls_name, weight in zip(data['classes'], class_weights):
    print(f"   - {cls_name}: {weight:.4f}")

print(f"\n✅ Total samples: {len(data['y'])}")
print(f"✅ Original files: ~109 (estimated)")
print(f"✅ Augmentation factor: ~{len(data['y']) // 109}x")

print("="*60)
print("✅ Data ready for training!")
print("="*60)
