"""
Model Comparison & Benchmark Script

Script ini untuk membandingkan performa berbagai model
dan membuat benchmark report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def load_training_logs(log_file):
    """Load training logs dari CSV"""
    try:
        df = pd.read_csv(log_file)
        return df
    except FileNotFoundError:
        return None

def plot_model_comparison(results_dict):
    """
    Plot perbandingan multiple model results
    
    Args:
        results_dict: {
            'model_name': {
                'accuracy': 0.95,
                'precision': 0.94,
                'recall': 0.96,
                'auc': 0.97
            }
        }
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        models = list(results_dict.keys())
        values = [results_dict[m].get(metric, 0) for m in models]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.90)')
        ax.legend()
        
        # Rotate x labels
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_benchmark_report(results_dict, save_path='benchmark_report.json'):
    """Create detailed benchmark report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'models': results_dict,
        'best_model': {},
        'recommendations': []
    }
    
    # Find best model for each metric
    metrics = ['accuracy', 'precision', 'recall', 'auc']
    
    for metric in metrics:
        best_model = max(results_dict.items(), 
                        key=lambda x: x[1].get(metric, 0))
        report['best_model'][metric] = {
            'model': best_model[0],
            'value': best_model[1].get(metric, 0)
        }
    
    # Generate recommendations
    overall_best = max(results_dict.items(),
                      key=lambda x: x[1].get('accuracy', 0))
    
    report['recommendations'].append({
        'type': 'best_overall',
        'model': overall_best[0],
        'reason': f"Highest accuracy: {overall_best[1].get('accuracy', 0):.3f}"
    })
    
    # Check for overfitting
    for model, metrics_dict in results_dict.items():
        if 'train_acc' in metrics_dict and 'val_acc' in metrics_dict:
            diff = metrics_dict['train_acc'] - metrics_dict['val_acc']
            if diff > 0.1:
                report['recommendations'].append({
                    'type': 'warning_overfitting',
                    'model': model,
                    'reason': f"Gap train-val: {diff:.3f}"
                })
    
    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def print_comparison_table(results_dict):
    """Print formatted comparison table"""
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create DataFrame
    df = pd.DataFrame(results_dict).T
    
    # Sort by accuracy
    df = df.sort_values('accuracy', ascending=False)
    
    # Format as percentage
    for col in df.columns:
        if col in ['accuracy', 'precision', 'recall', 'auc']:
            df[col] = df[col].apply(lambda x: f"{x:.2%}")
    
    print(df.to_string())
    print("="*80)
    
    # Find best
    print("\n🏆 BEST MODELS:")
    print("-"*80)
    
    metrics_full = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'auc': 'ROC-AUC'
    }
    
    for metric, full_name in metrics_full.items():
        if metric in df.columns:
            best_idx = df[metric].idxmax() if df[metric].dtype != object else \
                       max(results_dict.items(), key=lambda x: x[1].get(metric, 0))[0]
            print(f"  {full_name:12} : {best_idx}")
    
    print("="*80 + "\n")

def analyze_single_model(log_csv_path, model_name="Model"):
    """Analyze single model training log"""
    
    df = load_training_logs(log_csv_path)
    
    if df is None:
        print(f"❌ Log file not found: {log_csv_path}")
        return None
    
    print(f"\n📊 Analysis for {model_name}")
    print("-"*60)
    
    # Best epoch
    best_epoch = df['val_accuracy'].idxmax()
    
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Val Accuracy: {df.loc[best_epoch, 'val_accuracy']:.4f}")
    print(f"Train Accuracy: {df.loc[best_epoch, 'accuracy']:.4f}")
    print(f"Gap (Train-Val): {df.loc[best_epoch, 'accuracy'] - df.loc[best_epoch, 'val_accuracy']:.4f}")
    
    if 'val_precision' in df.columns:
        print(f"Val Precision: {df.loc[best_epoch, 'val_precision']:.4f}")
        print(f"Val Recall: {df.loc[best_epoch, 'val_recall']:.4f}")
    
    # Total epochs
    print(f"Total Epochs: {len(df)}")
    
    # Convergence speed
    target_acc = 0.85
    convergence_epoch = df[df['val_accuracy'] >= target_acc].index
    if len(convergence_epoch) > 0:
        print(f"Convergence to {target_acc:.0%}: Epoch {convergence_epoch[0] + 1}")
    
    print("-"*60)
    
    return {
        'accuracy': df.loc[best_epoch, 'val_accuracy'],
        'precision': df.loc[best_epoch, 'val_precision'] if 'val_precision' in df.columns else 0,
        'recall': df.loc[best_epoch, 'val_recall'] if 'val_recall' in df.columns else 0,
        'train_acc': df.loc[best_epoch, 'accuracy'],
        'val_acc': df.loc[best_epoch, 'val_accuracy'],
        'total_epochs': len(df),
        'best_epoch': best_epoch + 1
    }

def main():
    """Main comparison workflow"""
    
    print("="*80)
    print("MODEL COMPARISON & BENCHMARK TOOL")
    print("="*80)
    
    # Example: Compare different models
    # Modify paths based on your actual log files
    
    models_to_compare = {
        'Simple CNN': 'logs/training.csv',  # Update path
        'Deep CNN': 'logs/training_deep.csv',  # If exists
        # 'Attention CNN': 'logs/training_attention.csv',
    }
    
    results = {}
    
    for model_name, log_path in models_to_compare.items():
        result = analyze_single_model(log_path, model_name)
        if result:
            results[model_name] = result
    
    if len(results) > 1:
        # Print comparison table
        print_comparison_table(results)
        
        # Plot comparison
        plot_model_comparison(results)
        
        # Create benchmark report
        report = create_benchmark_report(results, 'visualizations/benchmark_report.json')
        
        print("\n✅ Comparison complete!")
        print("📊 Visualizations saved to: visualizations/model_comparison.png")
        print("📄 Report saved to: visualizations/benchmark_report.json")
    
    elif len(results) == 1:
        print("\n✅ Single model analysis complete!")
        print("💡 Train more models to enable comparison.")
    
    else:
        print("\n❌ No valid training logs found.")
        print("💡 Make sure to train models first:")
        print("   → python optimized_train.py")

if __name__ == "__main__":
    # Create visualizations directory if not exists
    Path('visualizations').mkdir(exist_ok=True)
    
    main()
