# Zero-Shot Semantic Matching for Pakistan Sign Language (PSL)

## Project Overview

This project tackles **isolated sign language recognition in low-data regimes** for Pakistan Sign Language (PSL). Traditional deep learning methods struggle with limited training data due to overfitting. This repository implements and compares multiple approaches designed to handle data scarcity:

- **CNN-LSTM**: Temporal modeling with frozen CNN features
- **C3D**: 3D convolutional networks with pretrained weights
- **AttentionLite-MHI**: Motion History Image guided spatial attention
- **MediaPipe**: Keypoint-based recognition with Transformer/LSTM backends
- **Zero-Shot Matching**: Structured semantic matching without training

## Quick Start

### Easiest Way to Run Experiments

The simplest way to run any method is using the **batch files** (`run_code.bat` for Windows or `run_code.sh` for macOS/Linux). This allows you to:
- Test multiple methods sequentially
- Run on different data splits (train/test)
- Use multiple seeds for statistical validation

### How to Use the Batch Files

#### Windows (run_code.bat)
1. **Open `run_code.bat`** in any text editor
2. **Configure your experiment** by modifying these lists:

```batch
set seed_list=1 2 3                   # Multiple seeds for robust results
set method_list=c3d c3d_pretrained cnn_lstm cnn_lstm_pretrained mhi_baseline mhi_fusion mhi_attention mediapipe mediapipe_transformer mediapipe_lstm
set num_words_list=1 5 10                 # Number of videos to process
```

3. **Run the batch file**:
```cmd
.\run_code.bat
```

#### macOS/Linux (run_code.sh)
1. **Open `run_code.sh`** in any text editor
2. **Configure your experiment** by modifying these lists:

```bash
seed_list=(1 42 123)                      # Multiple seeds for robust results
method_list=(c3d c3d_pretrained cnn_lstm cnn_lstm_pretrained mhi_baseline mhi_fusion mhi_attention mediapipe mediapipe_transformer mediapipe_lstm)
num_words_list=(1 5 10)                   # Number of videos to process
```

3. **Make executable and run**:
```bash
chmod +x run_code.sh
./run_code.sh
```

### Example Configurations

**Test all methods on 1 video:**
```batch
set method_list=c3d c3d_pretrained cnn_lstm cnn_lstm_pretrained mhi_baseline mhi_fusion mhi_attention mediapipe mediapipe_transformer mediapipe_lstm
set num_words_list=1
set seed_list=1
```

**Focus on specific methods with multiple scales:**
```batch
set method_list=c3d c3d_pretrained cnn_lstm cnn_lstm_pretrained
set num_words_list=1 5 10 20
set seed_list=1 42 123
```

**Test pretrained vs non-pretrained variants:**
```batch
set method_list=c3d c3d_pretrained cnn_lstm cnn_lstm_pretrained
set num_words_list=5
set seed_list=1 42
```

### Manual Execution

You can also run individual methods:
```bash
python main.py --method=zero_shot --split=test --num_words=5 --seed=42
```

## Data Structure

- `Words_train/`: Training videos (flat structure: all .mp4 files in one directory)
- `Words_test/`: Testing videos (flat structure: all .mp4 files in one directory)
- `results/`: Output accuracy files organized by method

## Results

Results are saved as JSON files in `results/{method}/accuracy_seed{seed}_n{num_words}_{split}.json` with format:
```json
{
  "method": "zero_shot",
  "split": "test", 
  "seed": 42,
  "num_words": 10,
  "total": 10,
  "correct": 7,
  "accuracy": 0.7
}
```

## Low-Data Challenges

In low-data regimes for sign language recognition:
- **Overfitting** is the primary challenge
- **Spatio temporal Deep learning methods** (C3D, CNN-LSTM) help with feature extraction
- **Zero-shot methods** avoid training entirely
- **Attention mechanisms** (AttentionLite-MHI) focus on motion-relevant regions
- **Keypoint-based approaches** (MediaPipe) reduce dimensionality