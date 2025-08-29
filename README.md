# Zero-Shot Semantic Matching for Pakistan Sign Language (PSL)

## Project Overview

This project implements **zero-shot semantic matching** for Pakistan Sign Language (PSL) recognition, comparing our novel approach against traditional deep learning baselines. Our method enables sign recognition without requiring training data by leveraging semantic understanding capabilities of large language models.

## Our Main Method: Zero-Shot Semantic Matching

**Zero-Shot Semantic Matching** is our primary contribution that:
- Uses Gemini to generate structured descriptions of PSL videos
- Performs semantic similarity matching against reference descriptions
- Enables recognition without training data or fine-tuning
- Leverages the semantic understanding of large language models

## Baseline Methods for Comparison

The following baseline methods are implemented for comparison:

- **CNN+LSTM**: Temporal modeling with InceptionV3 features and LSTM
- **3D CNN (R3D-18)**: 3D convolutional networks for spatiotemporal learning
- **MediaPipe Keypoints + Transformer**: Pose/hand/face keypoints with transformer
- **MediaPipe Keypoints + LSTM**: Pose/hand/face keypoints with LSTM
- **MHI Fusion**: Motion History Images with 3D CNN fusion
- **Gemini-LoRA (Finetuned)**: Finetuned Gemini model for comparison

## Quick Start

### Running Our Main Method (Zero-Shot Semantic Matching)

#### Prerequisites
1. **Google Cloud Project** with Vertex AI enabled
2. **Gemini API access** (gemini-2.5-pro model)
3. **Environment variables** configured:
   ```bash
   export PROJECT_ID="your-project-id"
   export LOCATION="us-central1"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
   ```

#### Method 1: Direct Python Execution
```bash
# Zero-shot mode (our main method)
python main.py --method=zero_shot --num_words=30 --seed=42

# Semantic shot mode (alternative approach)
python main.py --method=semantic_shot --num_words=30 --seed=42
```

#### Method 2: Batch Processing with Scripts
Use the provided batch scripts to run multiple methods and configurations:

**Windows (run_code.bat):**
```batch
# Edit the script to configure your experiment
set method_list=zero_shot semantic_shot cnn_lstm c3d mediapipe_transformer
set num_words_list=1 5 10 30
set seed_list=1 42

# Run the script
.\run_code.bat
```

**macOS/Linux (run_code.sh):**
```bash
# Edit the script to configure your experiment
method_list=(zero_shot semantic_shot cnn_lstm c3d mediapipe_transformer)
num_words_list=(1 5 10 30)
seed_list=(1 42)

# Make executable and run
chmod +x run_code.sh
./run_code.sh
```

### Running Baseline Methods

```bash
# Traditional deep learning baselines
python main.py --method=cnn_lstm --num_words=30 --epochs=20 --seed=42
python main.py --method=c3d --num_words=30 --epochs=20 --seed=42
python main.py --method=mediapipe_transformer --num_words=30 --epochs=20 --seed=42
python main.py --method=mediapipe_lstm --num_words=30 --epochs=20 --seed=42
python main.py --method=mhi_fusion --num_words=30 --epochs=20 --seed=42

# Finetuned Gemini baseline
python main.py --method=finetuned_gemini --num_words=30 --seed=42
```

## Data Structure

- `data/Words_train/`: Training videos (.MOV files)
- `data/Words_test/`: Testing videos (.MOV files)
- `outputs/`: Generated descriptions and predictions
- `results/`: Accuracy results organized by method

## Results Format

Results are saved as JSON files in `results/{method}/accuracy_seed{seed}_n{num_words}.json`:

```json
{
  "method": "zero_shot",
  "num_words": 30,
  "train_accuracy": 0.85,
  "test_accuracy": 0.80,
  "epochs": null
}
```

## Key Advantages of Our Method

1. **No Training Required**: Works immediately without model training
2. **Semantic Understanding**: Leverages LLM capabilities for sign interpretation
3. **Scalable**: Easy to add new signs without retraining
4. **Interpretable**: Provides human-readable descriptions and reasoning
5. **Low-Data Friendly**: Designed specifically for limited data scenarios

## Technical Details

### Zero-Shot Mode
- Generates descriptions using Gemini 2.5 Pro
- Compares against reference descriptions from training data
- Direct classification without semantic embeddings

### Semantic Shot Mode
- Generates descriptions using Gemini 2.5 Pro
- Creates semantic embeddings using SentenceTransformers
- Performs similarity-based classification

### Prompt Engineering
Our prompts are carefully designed to:
- Generate consistent, structured descriptions
- Focus on hand shapes, movements, and body locations
- Avoid hallucination and maintain accuracy
- Enable reliable semantic matching

## Citation

If you use this work, please cite:
