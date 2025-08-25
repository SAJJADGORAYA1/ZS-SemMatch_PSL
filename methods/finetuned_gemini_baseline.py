# finetuned_gemini_baseline.py
# Baseline using finetuned Gemini model on PSL video classification
from google import genai
from google.genai import types
import os
import re
import json
import time
import random
from typing import Dict, List, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# Configuration
# -------------------------
# Update these paths to your finetuned model
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
FINETUNED_MODEL_NAME = os.getenv("FINETUNED_MODEL_NAME")

GCS_PREFIX_TRAIN = os.getenv("GCS_PREFIX_TRAIN")
GCS_PREFIX_TEST = os.getenv("GCS_PREFIX_TEST")

# Output files
RESULTS_DIR = "results"
OUTPUT_JSON = "finetuned_gemini_results.json"
PREDICTIONS_TSV = "finetuned_gemini_results.tsv"
OUTPUTS_DIR = "outputs"

# Method for reproducibility
METHOD = "finetuned_gemini_baseline"

# -------------------------
# Helper Functions
# -------------------------
def get_video_files(directory: str) -> List[str]:
    """Get all .mov video files from directory"""
    video_dir = Path(directory)
    if not video_dir.exists():
        print(f"[WARNING] Directory {directory} not found")
        return []
    
    video_files = [f.stem for f in video_dir.glob("*.MOV")]
    return sorted(video_files)

def load_train_descriptions(path: str) -> Dict[str, str]:
    """Load training descriptions if available"""
    d = {}
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Will proceed without reference descriptions.")
        return d
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            word, desc = line.split(":", 1)
            word = word.strip()
            desc = desc.strip()
            if word:
                d[word] = desc
    return d

def parse_model_response(text: str) -> Tuple[str, str, str]:
    """Parse model response to extract prediction, confidence, and description"""
    text = text.strip()
    
    # Extract prediction (word)
    prediction = ""
    m_pred = re.search(r"(?im)^Best guess:\s*([^\n\r]+)", text)
    if m_pred:
        prediction = m_pred.group(1).strip()
    
    # Extract confidence
    confidence = ""
    m_conf = re.search(r"(?im)^Confidence:\s*([^\n\r]+)", text)
    if m_conf:
        confidence = m_conf.group(1).strip()
    
    # Extract description
    description = ""
    m_desc = re.search(r"(?is)Test description:\s*(.+?)(?:\n\s*Best guess:|\Z)", text)
    if m_desc:
        description = m_desc.group(1).strip()
    
    return prediction, confidence, description

def calculate_accuracy(predictions: List[Tuple[str, str]]) -> float:
    """Calculate accuracy from predictions"""
    if not predictions:
        return 0.0
    
    correct = sum(1 for actual, predicted in predictions if actual.lower() == predicted.lower())
    return correct / len(predictions)

def save_results(train_accuracy: float, test_accuracy: float, 
                train_predictions: List[Tuple[str, str]], 
                test_predictions: List[Tuple[str, str]],
                method: str, seed: int):
    """Save results to JSON file"""
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {
        "method": method,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": FINETUNED_MODEL_NAME,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_predictions": [
            {"actual": actual, "predicted": predicted} 
            for actual, predicted in train_predictions
        ],
        "test_predictions": [
            {"actual": actual, "predicted": predicted} 
            for actual, predicted in test_predictions
        ]
    }
    
    output_path = os.path.join(RESULTS_DIR, OUTPUT_JSON)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    
    # Also save detailed predictions to TSV in outputs folder
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    tsv_path = os.path.join(OUTPUTS_DIR, PREDICTIONS_TSV)
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("Dataset\tActualWord\tPredictedWord\n")
        
        for actual, predicted in train_predictions:
            f.write(f"train\t{actual}\t{predicted}\n")
        
        for actual, predicted in test_predictions:
            f.write(f"test\t{actual}\t{predicted}\n")
    
    print(f"Detailed predictions saved to: {tsv_path}")

# -------------------------
# Prompt Template
# -------------------------
PROMPT_TEMPLATE = """You are an expert PSL (Pakistani Sign Language) video analyst.

Watch the video clip carefully and identify the sign being performed.

Available vocabulary words: {vocabulary}

Instructions:
1. Watch the clip end-to-end, then replay once silently
2. Describe exactly ONE complete iteration of the sign
3. Ignore tiny differences (minor finger curvature, small wrist angle/path jitters)
4. If any detail is unclear, omit it rather than guessing
5. The complete clip shows only the sign being performed

OUTPUT FORMAT (exact headings):
Test description: <two sentences describing the sign>
Best guess: <ONE word from the vocabulary list>
Confidence: <high/medium/low>
"""

# -------------------------
# Main Processing Functions
# -------------------------
def process_videos(video_dir: str, vocabulary: List[str], client, model_name: str, seed: int, is_test: bool = False) -> List[Tuple[str, str]]:
    """Process videos in directory and return predictions"""
    predictions = []
    video_files = get_video_files(video_dir)
    
    if not video_files:
        print(f"[WARNING] No video files found in {video_dir}")
        return predictions
    
    print(f"Processing {len(video_files)} videos from {video_dir}")
    
    for i, word in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {word}")
        
        # Create prompt with vocabulary
        prompt = PROMPT_TEMPLATE.format(vocabulary=", ".join(f'"{w}"' for w in vocabulary))
        
        try:
            # Use environment variables for GCS URIs
            if is_test:
                # For test videos, use the test GCS prefix from env
                gcs_uri = f"{GCS_PREFIX_TEST}{word}.MOV"
            else:
                # For train videos, use the train GCS prefix from env
                gcs_uri = f"{GCS_PREFIX_TRAIN}{word}.MOV"
            
            print(f"  Using GCS URI: {gcs_uri}")
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=gcs_uri,
                                mime_type="video/MOV",
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                )
            ]
            
            config = types.GenerateContentConfig(
                temperature=0.0,  # deterministic
                top_p=1,
                seed=seed,  # Use the seed argument passed to the function
                max_output_tokens=1000,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ]
            )
            
            print(f"  Sending request to model...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            
            output_text = response.text or ""
            if not output_text:
                print(f"  [ERROR] Empty response from model")
                continue
            
            # Parse response
            prediction, confidence, description = parse_model_response(output_text)
            
            if prediction:
                predictions.append((word, prediction))
                print(f"  Actual: {word} | Predicted: {prediction} | Confidence: {confidence}")
            else:
                print(f"  [WARNING] Could not parse prediction from response")
                print(f"  Response: {output_text[:200]}...")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {word}: {str(e)}")
            continue
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    return predictions

def run_finetuned_gemini(num_words=1, seed: int = 42, out_dir: str = "results"):
    """Run finetuned Gemini baseline on PSL video classification."""
    print("=" * 60)
    print("Finetuned Gemini Baseline for PSL Video Classification")
    print("=" * 60)
    
    # Initialize Gemini client
    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        print(f"Initialized Gemini client for project: {PROJECT_ID}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini client: {str(e)}")
        return
    
    # Get vocabulary from local directories (for file listing)
    local_train_dir = "data/Words_train"
    local_test_dir = "data/Words_test"
    
    train_vocabulary = get_video_files(local_train_dir)
    test_vocabulary = get_video_files(local_test_dir)
    
    if not train_vocabulary:
        print(f"[ERROR] No training videos found. Please check the {local_train_dir} path.")
        return
    
    if not test_vocabulary:
        print(f"[ERROR] No test videos found. Please check the {local_test_dir} path.")
        return
    
    # Select subset by seed
    rng = random.Random(seed)
    rng.shuffle(train_vocabulary)
    subset_vocabulary = train_vocabulary[:max(1, num_words)]
    
    print(f"Selected {len(subset_vocabulary)} words: {subset_vocabulary}")
    
    # Process training videos
    print(f"\n{'='*20} Processing Training Videos {'='*20}")
    train_predictions = process_videos(local_train_dir, subset_vocabulary, client, FINETUNED_MODEL_NAME, seed, is_test=False)
    
    # Process test videos
    print(f"\n{'='*20} Processing Test Videos {'='*20}")
    test_predictions = process_videos(local_test_dir, subset_vocabulary, client, FINETUNED_MODEL_NAME, seed, is_test=True)
    
    # Calculate accuracies
    train_accuracy = calculate_accuracy(train_predictions)
    test_accuracy = calculate_accuracy(test_predictions)
    
    # Display results
    print(f"\n{'='*20} Results Summary {'='*20}")
    print(f"Method: finetuned_gemini")
    print(f"Seed: {seed}")
    print(f"Training Accuracy: {train_accuracy:.4f} ({len(train_predictions)} samples)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({len(test_predictions)} samples)")
    
    # Return results for main.py to handle (consistent with other baselines)
    return {
        "method": "finetuned_gemini",
        "num_words": num_words,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "model_name": FINETUNED_MODEL_NAME
    }

def main():
    """Main function to run the baseline evaluation"""
    run_finetuned_gemini(num_words=20, seed=SEED, out_dir=RESULTS_DIR)

if __name__ == "__main__":
    main()
