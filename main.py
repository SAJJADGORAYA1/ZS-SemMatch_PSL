"""
Unified entry point that dispatches to baseline modules. No training code here.

Usage:
  python main.py --method METHOD --num_words N

Methods:
  - c3d
  - cnn_lstm
  - zero_shot
  - semantic_zero_shot
  - mhi_baseline
  - mhi_fusion
  - mhi_attention
  - finetuned_gemini
"""

import argparse
import os
import random
import json
from methods.c3d_baseline import run_c3d
from methods.cnn_lstm_baseline import run_cnn_lstm
from methods.Mediapipe_baseline import run_mediapipe
from methods.MHI_baseline import run_mhi
from methods.finetuned_gemini_baseline import run_finetuned_gemini
from methods.zero_shot_semantic_matching import run_zero_shot_matching, run_semantic_matching

def main():
    parser = argparse.ArgumentParser(description='Run PSL baseline methods')
    parser.add_argument('--method', type=str, required=True,
                       choices=['c3d', 'c3d_pretrained', 'cnn_lstm', 'cnn_lstm_pretrained', 'mediapipe_transformer', 'mediapipe_lstm', 'mhi_fusion', 'mhi_attention', 'finetuned_gemini', 'zero_shot', 'semantic_shot'],
                       help='Method to run')
    parser.add_argument('--num_words', type=int, default=1, help='Number of words to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--out_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained weights for C3D')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set random seed
    random.seed(args.seed)
    
    # Run the selected method
    if args.method == 'c3d':
        results = run_c3d(num_words=args.num_words, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, use_pretrained=False)
    elif args.method == 'c3d_pretrained':
        results = run_c3d(num_words=args.num_words, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, use_pretrained=True)
    elif args.method == 'cnn_lstm':
        results = run_cnn_lstm(num_words=args.num_words, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, use_pretrained=False)
    elif args.method == 'cnn_lstm_pretrained':
        results = run_cnn_lstm(num_words=args.num_words, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, use_pretrained=True)
    # Removed plain mediapipe option - only transformer and lstm backends available
    elif args.method == 'mediapipe_transformer':
        results = run_mediapipe(num_words=args.num_words, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, backend='transformer')
    elif args.method == 'mediapipe_lstm':
        results = run_mediapipe(num_words=args.num_words, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, backend='lstm')
    # Removed plain mhi_baseline option - only fusion and attention variants available
    elif args.method == 'mhi_fusion':
        results = run_mhi(num_words=args.num_words, mode='fusion', seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, use_pretrained=not args.no_pretrained)
    elif args.method == 'mhi_attention':
        results = run_mhi(num_words=args.num_words, mode='attention', seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, use_pretrained=not args.no_pretrained)
    elif args.method == 'finetuned_gemini':
        results = run_finetuned_gemini(num_words=args.num_words, seed=args.seed, out_dir=args.out_dir)
    elif args.method == 'zero_shot':
        results = run_zero_shot_matching(num_words=args.num_words, seed=args.seed)
    elif args.method == 'semantic_shot':
        results = run_semantic_matching(num_words=args.num_words, seed=args.seed)
    else:
        print(f"Unknown method: {args.method}")
        return
    
    # Save results consistently
    if results:
        method_name = results.get('method', args.method)
        method_dir = os.path.join(args.out_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        out_path = os.path.join(method_dir, f"accuracy_seed{args.seed}_n{args.num_words}.json")
        
        # Ensure consistent output format
        output_data = {
            "method": method_name,
            "num_words": args.num_words,
            "train_accuracy": results.get('train_accuracy', 0.0),
            "test_accuracy": results.get('test_accuracy', 0.0)
        }
        
        # Add epochs if available
        if 'epochs' in results:
            output_data["epochs"] = results['epochs']
        
        # Add any additional fields from results
        for key, value in results.items():
            if key not in output_data:
                output_data[key] = value
        
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved results -> {out_path}")
    else:
        print("No results returned from method")

if __name__ == "__main__":
    main()

