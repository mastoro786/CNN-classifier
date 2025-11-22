import pandas as pd

df = pd.read_csv('logs/training.csv')

# Get best epoch
best_idx = df['val_accuracy'].idxmax()

print("="*70)
print("TRAINING RESULTS")
print("="*70)

print(f"\nTotal Epochs: {len(df)}")
print(f"Best Epoch: {best_idx + 1}")

print(f"\n🎯 BEST METRICS (Epoch {best_idx + 1}):")
print(f"  Train Accuracy: {df.loc[best_idx, 'accuracy']:.4f} ({df.loc[best_idx, 'accuracy']*100:.2f}%)")
print(f"  Val Accuracy:   {df.loc[best_idx, 'val_accuracy']:.4f} ({df.loc[best_idx, 'val_accuracy']*100:.2f}%)")
print(f"  Train Loss:     {df.loc[best_idx, 'loss']:.4f}")
print(f"  Val Loss:       {df.loc[best_idx, 'val_loss']:.4f}")

if 'precision' in df.columns:
    print(f"  Val Precision:  {df.loc[best_idx, 'val_precision']:.4f}")
if 'recall' in df.columns:
    print(f"  Val Recall:     {df.loc[best_idx, 'val_recall']:.4f}")
if 'auc' in df.columns:
    print(f"  Val AUC:        {df.loc[best_idx, 'val_auc']:.4f}")

gap = df.loc[best_idx, 'accuracy'] - df.loc[best_idx, 'val_accuracy']
print(f"\n  Overfitting Gap: {gap:.4f} ({gap*100:.2f}%)")

val_acc = df.loc[best_idx, 'val_accuracy']
if val_acc >= 0.90:
    print(f"\n✅ STATUS: EXCELLENT - Ready for production!")
elif val_acc >= 0.85:
    print(f"\n✅ STATUS: GOOD - Acceptable performance")
else:
    print(f"\n⚠️ STATUS: Needs improvement")

print("="*70)
