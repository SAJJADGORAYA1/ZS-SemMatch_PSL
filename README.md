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

#### Method 2: Using Vertex AI Studio
1. **Open Vertex AI Studio** in your Google Cloud Console
2. **Navigate to Model Garden** and select Gemini 2.5 Pro
3. **Upload PSL videos** to Cloud Storage
4. **Use our prompt templates** from the code for consistent descriptions
5. **Run semantic matching** using the generated descriptions

#### Method 3: Batch Processing
```bash
# Run multiple configurations
python main.py --method=zero_shot --num_words=1 --seed=1
python main.py --method=zero_shot --num_words=5 --seed=1
python main.py --method=zero_shot --num_words=10 --seed=1
python main.py --method=zero_shot --num_words=30 --seed=1
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
