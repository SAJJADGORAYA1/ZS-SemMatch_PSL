import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import json

def create_confusion_matrix(actual_words, predicted_words, method_name, out_dir="results"):
    """Create and save confusion matrix"""
    if not actual_words or not predicted_words:
        print(f"[WARNING] No data available for confusion matrix for {method_name}")
        return
    
    # Create confusion matrix
    cm = confusion_matrix(actual_words, predicted_words, labels=sorted(set(actual_words + predicted_words)))
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(actual_words + predicted_words)),
                yticklabels=sorted(set(actual_words + predicted_words)))
    plt.title(f'Confusion Matrix - {method_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plots_dir = "plots"
    confusion_dir = os.path.join(plots_dir, "confusion_matrix")
    os.makedirs(confusion_dir, exist_ok=True)
    plot_path = os.path.join(confusion_dir, f"{method_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {plot_path}")
    
    # Also save confusion matrix data as JSON
    data_path = os.path.join(confusion_dir, f"{method_name}_confusion_matrix.json")
    cm_data = {
        "method": method_name,
        "actual_words": actual_words,
        "predicted_words": predicted_words,
        "confusion_matrix": cm.tolist(),
        "labels": sorted(set(actual_words + predicted_words))
    }
    with open(data_path, 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    print(f"Confusion matrix data saved to: {data_path}")

def create_loss_graph(epoch_losses, method_name, out_dir="results"):
    """Create and save loss graph"""
    if not epoch_losses:
        print(f"[WARNING] No loss data available for {method_name}")
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(epoch_losses) + 1)
    plt.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o')
    plt.title(f'Training Loss - {method_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plots_dir = "plots"
    loss_dir = os.path.join(plots_dir, "loss_graphs")
    os.makedirs(loss_dir, exist_ok=True)
    plot_path = os.path.join(loss_dir, f"{method_name}_loss_graph.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss graph saved to: {plot_path}")
    
    # Also save loss data as JSON
    data_path = os.path.join(loss_dir, f"{method_name}_loss_data.json")
    loss_data = {
        "method": method_name,
        "epochs": list(epochs),
        "losses": epoch_losses
    }
    with open(data_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    print(f"Loss data saved to: {data_path}")
