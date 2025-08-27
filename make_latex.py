import os
import json

RESULTS_DIR = "results"
NUM_WORDS = 30
SEED = 42
OUTPUT_DIR = "outputs"

methods = [
    ("cnn_lstm", "CNN+LSTM (Scratch)"),
    ("c3d", "3D CNN (R3D-18, Scratch)"),
    ("mediapipe_transformer", "MediaPipe Keypoints + Transformer"),
    ("mediapipe_lstm", "MediaPipe Keypoints + LSTM"),
    ("mhi_fusion", "MHI Fusion"),
    ("finetuned_gemini", "Gemini-LoRA (Finetuned)"),
    ("zero_shot", "Ours (Zero-Shot Matching)"),
    ("semantic_shot", "Ours (Semantic Shot)"),
]

def get_result(method):
    """Return (train_acc, test_acc) or ('X.X', 'X.X') if missing."""
    # All methods now use the same naming pattern: accuracy_seed1_n30.json
    possible_paths = [
        os.path.join(RESULTS_DIR, method, f"accuracy_seed1_n30.json"),
        os.path.join(RESULTS_DIR, method, f"accuracy_seed{SEED}_n30.json"),
        # Fallback patterns if needed
        os.path.join(RESULTS_DIR, method, f"accuracy_n30.json"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                train_acc = f"{data.get('train_accuracy', 0.0):.2f}"
                test_acc = f"{data.get('test_accuracy', 0.0):.2f}"
                return train_acc, test_acc
    
    return "X.X", "X.X"

def make_table():
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Signer-independent Top-1 accuracy (\\%) on PSL test set with 30 words. Best test result in \\textbf{bold}.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Train Acc.} & \\textbf{Test Acc.} \\\\")
    lines.append("\\midrule")

    # Get all results to find the best test accuracy
    all_results = []
    for method, label in methods:
        train_acc, test_acc = get_result(method)
        all_results.append((label, train_acc, test_acc))
    
    # Find the best test accuracy (excluding X.X values)
    best_test_acc = 0.0
    for _, _, test_acc in all_results:
        if test_acc != "X.X":
            try:
                acc_val = float(test_acc)
                if acc_val > best_test_acc:
                    best_test_acc = acc_val
            except ValueError:
                continue
    
    # Generate table rows with best result in bold
    for label, train_acc, test_acc in all_results:
        if test_acc != "X.X" and float(test_acc) == best_test_acc:
            # Bold the best test result
            lines.append(f"{label} & {train_acc} & \\textbf{{{test_acc}}} \\\\")
        else:
            lines.append(f"{label} & {train_acc} & {test_acc} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

if __name__ == "__main__":
    table = make_table()
    print(table)
    
    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "latex_table.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(table)
    print(f"\nLaTeX table saved to: {output_path}")
