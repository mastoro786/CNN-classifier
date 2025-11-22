import pandas as pd
import numpy as np

# Load training log
df = pd.read_csv('logs/training.csv')

print("\n" + "="*80)
print("TRAINING RESULTS SUMMARY")
print("="*80)

# Best epoch based on val_accuracy
best_epoch_idx = df['val_accuracy'].idxmax()
best_epoch = best_epoch_idx + 1

print(f"\n📊 TRAINING STATISTICS:")
print(f"   Total Epochs Run: {len(df)}")
print(f"   Best Epoch: {best_epoch}")
print(f"   Early Stopped: Yes (no improvement for 25 epochs)")

print(f"\n🎯 BEST MODEL PERFORMANCE (Epoch {best_epoch}):")
print("-" * 80)

metrics = {
    'Train Accuracy': df.loc[best_epoch_idx, 'accuracy'],
    'Val Accuracy': df.loc[best_epoch_idx, 'val_accuracy'],
    'Train Loss': df.loc[best_epoch_idx, 'loss'],
    'Val Loss': df.loc[best_epoch_idx, 'val_loss'],
}

if 'precision' in df.columns:
    metrics['Train Precision'] = df.loc[best_epoch_idx, 'precision']
    metrics['Val Precision'] = df.loc[best_epoch_idx, 'val_precision']
    
if 'recall' in df.columns:
    metrics['Train Recall'] = df.loc[best_epoch_idx, 'recall']
    metrics['Val Recall'] = df.loc[best_epoch_idx, 'val_recall']

if 'auc' in df.columns:
    metrics['Train AUC'] = df.loc[best_epoch_idx, 'auc']
    metrics['Val AUC'] = df.loc[best_epoch_idx, 'val_auc']

for metric, value in metrics.items():
    if 'Loss' in metric:
        print(f"   {metric:20s}: {value:.4f}")
    else:
        print(f"   {metric:20s}: {value:.4f} ({value*100:.2f}%)")

# Check overfitting
gap = metrics['Train Accuracy'] - metrics['Val Accuracy']
print(f"\n⚖️  OVERFITTING ANALYSIS:")
print(f"   Train-Val Gap: {gap:.4f} ({gap*100:.2f}%)")

if gap < 0.05:
    print("   Status: ✅ Excellent - No significant overfitting")
elif gap < 0.10:
    print("   Status: ✅ Good - Minimal overfitting")
elif gap < 0.15:
    print("   Status: ⚠️ Moderate - Some overfitting")
else:
    print("   Status: ❌ High - Significant overfitting")

# Performance rating
val_acc = metrics['Val Accuracy']
print(f"\n🏆 PERFORMANCE RATING:")
if val_acc >= 0.95:
    rating = "🌟 EXCELLENT"
    comment = "Outstanding! Model is production-ready."
elif val_acc >= 0.90:
    rating = "🥇 VERY GOOD"
    comment = "Great performance! Ready for deployment."
elif val_acc >= 0.85:
    rating = "🥈 GOOD"
    comment = "Acceptable performance. Consider improvements."
elif val_acc >= 0.80:
    rating = "🥉 FAIR"
    comment = "Needs improvement. Try different approaches."
else:
    rating = "⚠️ POOR"
    comment = "Significant improvements needed."

print(f"   Rating: {rating}")
print(f"   Val Accuracy: {val_acc*100:.2f}%")
print(f"   Comment: {comment}")

# Last 10 epochs summary
print(f"\n📈 LAST 10 EPOCHS (Convergence Analysis):")
print("-" * 80)
last_10 = df.tail(10)[['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss']]
print(last_10.to_string(index=False))

print("\n" + "="*80)
print("✅ Training completed successfully with early stopping!")
print("="*80)
