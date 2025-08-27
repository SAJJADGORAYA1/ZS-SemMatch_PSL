# test_infer_from_descriptions.py
from google import genai
from google.genai import types
import os
import re
import threading
import keyboard  # pip install keyboard
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import json
import argparse
from utils.plot_utils import create_confusion_matrix

# Load environment variables
load_dotenv()

# -------------------------
# Paths / config
# -------------------------
# Force use of local data paths, ignore environment variables
video_dir_train = "data/Words_train"
video_dir_test  = "data/Words_test"
gcs_prefix_train = os.getenv("GCS_PREFIX_TRAIN", "gs://psl-train-clipped/train/")
gcs_prefix_test  = os.getenv("GCS_PREFIX_TEST", "gs://psl-train-clipped/train/")
train_desc_file  = "outputs/descriptions_train-clipped-32.txt"

# Output file paths - separate files for each mode
OUTPUTS_DIR = "outputs"
ZERO_SHOT_TEST_DESC_FILE = "outputs/zero_shot_descriptions.txt"
ZERO_SHOT_PRED_FILE = "outputs/zero_shot_predictions.txt"
SEMANTIC_TEST_DESC_FILE = "outputs/semantic_descriptions.txt"
SEMANTIC_PRED_FILE = "outputs/semantic_predictions.txt"

# -------------------------
# Helpers
# -------------------------
def load_train_descriptions(path: str) -> Dict[str, str]:
    """
    Expected format per line:
      word: description...
    """
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

def get_vocab_from_train_videos(train_dir: str) -> List[str]:
    print(f"[DEBUG] Scanning train directory: {train_dir}")
    
    if not os.path.exists(train_dir):
        print(f"[ERROR] Directory does not exist: {train_dir}")
        print(f"[INFO] Current working directory: {os.getcwd()}")
        print(f"[INFO] Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
        return []
    
    try:
        files = os.listdir(train_dir)
        print(f"[DEBUG] Found {len(files)} files in {train_dir}")
        mov_files = [f for f in files if f.lower().endswith(".mov")]
        print(f"[DEBUG] Found {len(mov_files)} .mov files")
        
        vocab = sorted([
            os.path.splitext(f)[0]
            for f in mov_files
        ])
        print(f"[DEBUG] Extracted vocabulary: {vocab[:5]}... (total: {len(vocab)})")
        return vocab
    except Exception as e:
        print(f"[ERROR] Failed to scan directory {train_dir}: {str(e)}")
        return []

def get_test_filename(word: str) -> str:
    # Capitalize first letter for test set
    # return word[0].upper() + word[1:] + ".MOV"
    return word[0] + word[1:] + ".MOV"

def build_reference_block(train_descs: Dict[str, str], allowed_words: List[str]) -> str:
    """
    Render concise reference descriptions for the model to compare against.
    Only include words in allowed_words (order is alphabetical).
    """
    lines = []
    for w in allowed_words:
        ref = train_descs.get(w, "").strip()
        if ref:
            lines.append(f"- {w}: {ref}")
    return "\n".join(lines)

def parse_model_guess(text: str):
    """
    Extract:
      - test description (after 'Test description:' line)
      - best guess (after 'Best guess:' line)
      - optional confidence (after 'Confidence:' line)
    """
    # Normalize line endings
    t = text.strip()

    # Test description block
    test_desc = ""
    m_desc = re.search(r"(?is)Test description:\s*(.+?)(?:\n\s*Best guess:|\Z)", t)
    if m_desc:
        test_desc = m_desc.group(1).strip()

    # Best guess
    best_guess = ""
    m_guess = re.search(r"(?im)^Best guess:\s*([^\n\r]+)", t)
    if m_guess:
        best_guess = m_guess.group(1).strip()

    # Confidence (optional)
    confidence = ""
    m_conf = re.search(r"(?im)^Confidence:\s*([^\n\r]+)", t)
    if m_conf:
        confidence = m_conf.group(1).strip()

    return test_desc, best_guess, confidence

def append_line(path: str, line: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


# -------------------------
# Semantic Search Helpers
# -------------------------
def create_embeddings(descriptions: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Create embeddings for all descriptions"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = {}
    for word, desc in descriptions.items():
        embeddings[word] = model.encode(desc, convert_to_tensor=False)
    return embeddings

def get_top_n_similar(query_desc: str, 
                     word_embeddings: Dict[str, np.ndarray],
                     train_descs: Dict[str, str],
                     n: int = 5) -> Dict[str, str]:
    """Get top N most similar descriptions using cosine similarity"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query_desc, convert_to_tensor=False)
    
    similarities = {}
    for word, embedding in word_embeddings.items():
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities[word] = similarity
    
    top_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
    return {word: train_descs[word] for word, _ in top_words}

# -------------------------
# Prompts for different modes
# -------------------------

# Prompt for testing_2.py (zero_shot mode)
PROMPT_HEADER_ZERO_SHOT = (
    "You are an expert PSL video analyst.\n\n"
    "INPUTS:\n"
    "• TEST VIDEO: The clip contains one full iteration of the sign.\n"
    "• REFERENCE DESCRIPTIONS: One per vocabulary word; each is written as exactly two sentences in a fixed format.\n\n"
    """INSTRUCTIONS (strict):
    1) Watch the clip end-to-end, then replay once silently.
    2) Describe exactly ONE complete iteration of the sign.
    3) Ignore tiny differences (minor finger curvature, small wrist angle/path jitters).
    4) If any detail is unclear, omit it rather than guessing.
    5) The complete clip is only of the sign being performed, not a conversation or other context. So do not miss any part of the sign.

    WRITE EXACTLY TWO SENTENCES in this fixed order:
    • Sentence 1: number of hands + general handshape family (fist/open/flat/pinch/thumbs-up/point) + broad body area (head/face/chest/abdomen/neutral space).
    • Sentence 2: ordered motion and contact (tap/rub/grasp/link/none) described step-by-step."""
    "Using only the literal content of your two-sentence TEST DESCRIPTION, choose the SINGLE best-matching word strictly from the provided list of REFERENCE DESCRIPTIONS.\n"
    "OUTPUT (exact headings, nothing else):\n"
    "Test description: Description: <two sentences>\n"
    "Best guess: <ONE word from the provided list>\n"
)

# Prompt for semantic matching mode
PROMPT_HEADER_SEMANTIC = (
    "You will receive:\n"
    "1) A TEST DESCRIPTION (two sentences, fixed format described earlier).\n"
    "2) REFERENCE DESCRIPTIONS (same format), each labeled with ONE allowed vocabulary word.\n\n"
    "TASK (strict):\n"
    "- Choose the SINGLE best-matching word ONLY from the provided list by comparing the literal content (hand count/shape/area, ordered motion, contact).\n"
    "- Do NOT paraphrase, explain, or output confidence.\n\n"
    "Output EXACTLY one line:\n"
    "Best guess: <ONE word from the provided list>\n"
)

DESCRIPTION_PROMPT = """
You are an expert PSL video analyst.

INSTRUCTIONS (strict):
1) Watch the clip end-to-end, then replay once silently.
2) Describe exactly ONE complete iteration of the sign.
3) Ignore tiny differences (minor finger curvature, small wrist angle/path jitters).
4) If any detail is unclear, omit it rather than guessing.
5) The complete clip is only of the sign being performed, not a conversation or other context. So do not miss any part of the sign.

WRITE EXACTLY TWO SENTENCES in this fixed order:
• Sentence 1: number of hands + general handshape family (fist/open/flat/pinch/thumbs-up/point) + broad body area (head/face/chest/abdomen/neutral space).
• Sentence 2: ordered motion and contact (tap/rub/grasp/link/none) described step-by-step.

Rules:
- Do NOT guess the word.
- Do NOT use alphabet letters or technical sign-language labels.
- Output ONLY:

Description: <two sentences>

Examples:
- Description: Two hands form open palms at chest level. The right palm taps the left palm in a forward motion with brief contact.
- Description: Both hands form circles with index and thumb near the chest. They link briefly then pull apart in a single link-and-separate motion.

Counterexample to avoid (hallucination): do not mention a circular motion unless it is clearly seen in the clip you watched.
"""

# -------------------------
# Stop support
# -------------------------
stop_flag = False
def listen_for_stop():
    global stop_flag
    print("Press 'q' to stop the script.")
    keyboard.wait('q')
    stop_flag = True

# -------------------------
# Main functions for different modes
# -------------------------

def run_zero_shot_mode(allowed_words, train_descs, client, model):
    """Run the testing_2.py logic (zero shot mode)"""
    print("Running in ZERO_SHOT mode")
    
    reference_block = build_reference_block(train_descs, allowed_words)
    allowed_words_str = ", ".join(f'"{w}"' for w in allowed_words)

    # Collect results for standardized output
    actual_words = []
    predicted_words = []

    for word in allowed_words:
        if stop_flag:
            print("Stopping as requested by user.")
            break

        # Use training videos (this function is for training data)
        video_filename = f"{word}.MOV"
        gcs_uri = f"{gcs_prefix_train}{video_filename}"
        local_path = os.path.join(video_dir_train, video_filename)
        
        # Check if local video file exists (for debugging naming issues)
        if not os.path.exists(local_path):
            print(f"[WARNING] Local training video not found: {local_path}")

        # Compose the full instruction (references + allowed words)
        instruction = (
            PROMPT_HEADER_ZERO_SHOT
            + "REFERENCE DESCRIPTIONS (one per word):\n"
            + reference_block
        )

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
                    types.Part(text=instruction),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0.0,      # deterministic
            top_p=1,
            seed=0,
            max_output_tokens=12000,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            media_resolution="MEDIA_RESOLUTION_LOW",
        )

        try:
            print(f"\n[DEBUG] Sending request to model for word: {word}")
            print(f"[DEBUG] Request details:")
            print(f"  - Model: {model}")
            print(f"  - Video URI: {gcs_uri}")
            print(f"  - Content length: {len(instruction)} chars")
            
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            # Add detailed response debugging
            print(f"[DEBUG] Response details:")
            print(f"  - Response object type: {type(resp)}")
            print(f"  - Has candidates: {bool(resp.candidates)}")
            if hasattr(resp, 'candidates') and resp.candidates:
                print(f"  - Number of candidates: {len(resp.candidates)}")
                print(f"  - First candidate finish reason: {resp.candidates[0].finish_reason}")
            
            output_text = resp.text or ""
            
            if not output_text:
                print(f"[ERROR] Empty response from model for word: {word}")
            elif len(output_text) < 10:
                print(f"[WARNING] Suspiciously short response for word: {word}")
                
        except Exception as e:
            print(f"[ERROR] API error for word {word}:")
            print(f"  - Error type: {type(e).__name__}")
            print(f"  - Error message: {str(e)}")
            print(f"  - Error details: {getattr(e, 'details', 'No details available')}")
            continue

        test_desc, best_guess, confidence = parse_model_guess(output_text)


        if not test_desc:
            print(f"[WARNING] No test description parsed for word: {word}")
        if not best_guess:
            print(f"[WARNING] No best guess parsed for word: {word}")

        # Console summary
        print(f"\n=== {word} [train] ===")
        print(f"Best guess: {best_guess}")
        print("=====================\n")

        # Collect results for standardized output
        actual_words.append(word)
        predicted_words.append(best_guess)

    return actual_words, predicted_words

def run_zero_shot_mode_test(allowed_words, train_descs, client, model):
    """Run the testing_2.py logic for test videos"""
    print("Running in ZERO_SHOT mode for TEST videos")
    
    reference_block = build_reference_block(train_descs, allowed_words)
    allowed_words_str = ", ".join(f'"{w}"' for w in allowed_words)

    # Collect results for standardized output
    actual_words = []
    predicted_words = []

    for word in allowed_words:
        if stop_flag:
            print("Stopping as requested by user.")
            break

        # Use test videos
        video_filename = get_test_filename(word)
        gcs_uri = f"{gcs_prefix_test}{video_filename}"
        local_path = os.path.join(video_dir_test, video_filename)
        
        # Check if local video file exists (for debugging naming issues)
        if not os.path.exists(local_path):
            print(f"[WARNING] Local test video not found: {local_path}")

        # Compose the full instruction (references + allowed words)
        instruction = (
            PROMPT_HEADER_ZERO_SHOT
            + "REFERENCE DESCRIPTIONS (one per word):\n"
            + reference_block
        )

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
                    types.Part(text=instruction),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0.0,      # deterministic
            top_p=1,
            seed=0,
            max_output_tokens=12000,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            media_resolution="MEDIA_RESOLUTION_LOW",
        )

        try:
            print(f"\n[DEBUG] Sending request to model for word: {word}")
            print(f"[DEBUG] Request details:")
            print(f"  - Model: {model}")
            print(f"  - Video URI: {gcs_uri}")
            print(f"  - Content length: {len(instruction)} chars")
            
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            # Add detailed response debugging
            print(f"[DEBUG] Response details:")
            print(f"  - Response object type: {type(resp)}")
            print(f"  - Has candidates: {bool(resp.candidates)}")
            if hasattr(resp, 'candidates') and resp.candidates:
                print(f"  - Number of candidates: {len(resp.candidates)}")
                print(f"  - First candidate finish reason: {resp.candidates[0].finish_reason}")
            
            output_text = resp.text or ""
            
            if not output_text:
                print(f"[ERROR] Empty response from model for word: {word}")
            elif len(output_text) < 10:
                print(f"[WARNING] Suspiciously short response for word: {word}")
                
        except Exception as e:
            print(f"[ERROR] API error for word {word}:")
            print(f"  - Error type: {type(e).__name__}")
            print(f"  - Error message: {str(e)}")
            print(f"  - Error details: {getattr(e, 'details', 'No details available')}")
            continue

        test_desc, best_guess, confidence = parse_model_guess(output_text)

        if not test_desc:
            print(f"[WARNING] No test description parsed for word: {word}")
        if not best_guess:
            print(f"[WARNING] No best guess parsed for word: {word}")

        # Console summary
        print(f"\n=== {word} [test] ===")
        print(f"Best guess: {best_guess}")
        print("=====================\n")

        # Collect results for standardized output
        actual_words.append(word)
        predicted_words.append(best_guess)

    return actual_words, predicted_words

def run_semantic_shot_mode(allowed_words, train_descs, client, model, is_test=False):
    """Run the semantic matching logic (original zero_shot_semantic_matching.py logic)"""
    mode = "TEST" if is_test else "TRAIN"
    print(f"Running in SEMANTIC_SHOT mode for {mode} videos")
    
    # Create embeddings once at startup
    print("[DEBUG] Creating embeddings for train descriptions...")
    word_embeddings = create_embeddings(train_descs)

    # Clear/initialize output files

    # Ensure outputs directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    open(SEMANTIC_TEST_DESC_FILE, "w", encoding="utf-8").close()
    open(SEMANTIC_PRED_FILE, "w", encoding="utf-8").close()
    append_line(SEMANTIC_PRED_FILE, "ActualWord\tPredictedWord\tConfidence\tTestDescription")

    # Collect results for standardized output
    actual_words = []
    predicted_words = []

    for word in allowed_words:
        if stop_flag:
            print("Stopping as requested by user.")
            break

        # Use training videos (this function is for training data)
        video_filename = f"{word}.MOV"
        gcs_uri = f"{gcs_prefix_train}{video_filename}"
        print(f"[DEBUG] Processing word: '{word}' | Training filename: '{video_filename}' | GCS URI: '{gcs_uri}'")

        # Check if local training file exists (for debugging naming issues)
        local_path = os.path.join(video_dir_train, video_filename)
        if not os.path.exists(local_path):
            print(f"[WARNING] Local training video not found: {local_path}")

        generate_content_config = types.GenerateContentConfig(
            temperature=0.0,      # deterministic
            top_p=1,
            seed=0,
            max_output_tokens=12000,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            media_resolution="MEDIA_RESOLUTION_LOW",
        )
        
        # Step 1: Get initial description
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                file_data=types.FileData(
                                    file_uri=gcs_uri,
                                    mime_type="video/MOV",
                                )
                            ),
                            types.Part(text=DESCRIPTION_PROMPT),
                        ],
                    )
                ],
                config=generate_content_config,
            )
            
            initial_desc = resp.text.split("Description:", 1)[-1].strip()

            # Step 2: Get similar descriptions
            print("\n[DEBUG] Top similar descriptions retrieved:")
            top_matches = get_top_n_similar(initial_desc, word_embeddings, train_descs, n=20)
            for matched_word in top_matches.keys():
                print(f"  - {matched_word}")
            
            reference_block = build_reference_block(top_matches, allowed_words)
            # Compose the full instruction (references + allowed words)
            instruction = (
                PROMPT_HEADER_SEMANTIC
                + "\nREFERENCE DESCRIPTIONS (one per word):\n"
                + reference_block
                + f"\nTest description: {initial_desc}"
            )

            # Step 3: Make final prediction
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=instruction),
                        ],
                    )
                ],
                config=generate_content_config,
            )
            
            output_text = resp.text or ""
            
            if not output_text:
                print(f"[ERROR] Empty response from model for word: {word}")
                continue
                
        except Exception as e:
            print(f"[ERROR] API error for word {word}: {str(e)}")
            continue

        test_desc, best_guess, confidence = parse_model_guess(output_text)
        
        # Keep only the final result summary
        print(f"\n=== {word} [train] ===")
        print(f"Best guess: {best_guess}")
        print("=====================\n")

        # Collect results for standardized output
        actual_words.append(word)
        predicted_words.append(best_guess)

    return actual_words, predicted_words

def run_semantic_shot_mode_test(allowed_words, train_descs, client, model):
    """Run the semantic matching logic for test videos"""
    print("Running in SEMANTIC_SHOT mode for TEST videos")
    
    # Create embeddings once at startup
    print("[DEBUG] Creating embeddings for train descriptions...")
    word_embeddings = create_embeddings(train_descs)

    # Collect results for standardized output
    actual_words = []
    predicted_words = []

    for word in allowed_words:
        if stop_flag:
            print("Stopping as requested by user.")
            break

        # Use test videos
        test_filename = get_test_filename(word)
        test_gcs_uri = f"{gcs_prefix_test}{test_filename}"
        print(f"[DEBUG] Processing word: '{word}' | Test filename: '{test_filename}' | GCS URI: '{test_gcs_uri}'")

        # Check if local test file exists (for debugging naming issues)
        local_test_path = os.path.join(video_dir_test, test_filename)
        if not os.path.exists(local_test_path):
            print(f"[WARNING] Local test video not found: {local_test_path}")

        generate_content_config = types.GenerateContentConfig(
            temperature=0.0,      # deterministic
            top_p=1,
            seed=0,
            max_output_tokens=12000,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
            ],
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            media_resolution="MEDIA_RESOLUTION_LOW",
        )
        
        # Step 1: Get initial description
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                file_data=types.FileData(
                                    file_uri=test_gcs_uri,
                                    mime_type="video/MOV",
                                )
                            ),
                            types.Part(text=DESCRIPTION_PROMPT),
                        ],
                    )
                ],
                config=generate_content_config,
            )
            
            initial_desc = resp.text.split("Description:", 1)[-1].strip()

            # Step 2: Get similar descriptions
            print("\n[DEBUG] Top similar descriptions retrieved:")
            top_matches = get_top_n_similar(initial_desc, word_embeddings, train_descs, n=15)
            for matched_word in top_matches.keys():
                print(f"  - {matched_word}")
            
            reference_block = build_reference_block(top_matches, allowed_words)
            # Compose the full instruction (references + allowed words)
            instruction = (
                PROMPT_HEADER_SEMANTIC
                + "\nREFERENCE DESCRIPTIONS (one per word):\n"
                + reference_block
                + f"\nTest description: {initial_desc}"
            )

            # Step 3: Make final prediction
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=instruction),
                        ],
                    )
                ],
                config=generate_content_config,
            )
            
            output_text = resp.text or ""
            
            if not output_text:
                print(f"[ERROR] Empty response from model for word: {word}")
                continue
                
        except Exception as e:
            print(f"[ERROR] API error for word {word}: {str(e)}")
            continue

        test_desc, best_guess, confidence = parse_model_guess(output_text)
        
        # Keep only the final result summary
        print(f"\n=== {word} [test] ===")
        print(f"Best guess: {best_guess}")
        print("=====================\n")

        # Collect results for standardized output
        actual_words.append(word)
        predicted_words.append(best_guess)

    return actual_words, predicted_words

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='PSL Video Classification with Gemini')
    parser.add_argument('--mode', choices=['zero_shot', 'semantic_shot'], default='semantic_shot',
                       help='Mode to run: zero_shot (testing_2 logic) or semantic_shot (semantic matching logic)')
    parser.add_argument('--num_words', type=int, default=None, help='Number of words to process (default: all)')
    
    args = parser.parse_args()
    
    # Load vocab and references
    allowed_words = get_vocab_from_train_videos(video_dir_train)
    train_descs = load_train_descriptions(train_desc_file)

    # Limit words if specified
    if args.num_words and args.num_words < len(allowed_words):
        allowed_words = allowed_words[:args.num_words]

    client = genai.Client(vertexai=True, project="finetuning-gemini-on-psl", location="us-central1")
    model = "publishers/google/models/gemini-2.5-pro"

    # Run based on mode
    if args.mode == 'zero_shot':
        actual_words, predicted_words = run_zero_shot_mode(allowed_words, train_descs, client, model)
        method_name = "zero_shot"
    else:  # semantic_shot
        actual_words, predicted_words = run_semantic_shot_mode(allowed_words, train_descs, client, model)
        method_name = "semantic_shot"


def run_zero_shot_matching(num_words=1, seed: int = 42, out_dir: str = "results", confusion=False):
    """Wrapper function for main.py integration"""
    # Process both training and test data for proper accuracy calculation
    
    # Get vocabulary and references
    allowed_words = get_vocab_from_train_videos(video_dir_train)
    train_descs = load_train_descriptions(train_desc_file)
    
    # Limit words if specified
    if num_words and num_words < len(allowed_words):
        allowed_words = allowed_words[:num_words]
    
    client = genai.Client(vertexai=True, project="finetuning-gemini-on-psl", location="us-central1")
    model = "publishers/google/models/gemini-2.5-pro"
    
    # Process training videos
    print(f"\n{'='*20} Processing Training Videos {'='*20}")
    train_actual, train_predicted = run_zero_shot_mode(allowed_words, train_descs, client, model)
    
    # Process test videos  
    print(f"\n{'='*20} Processing Test Videos {'='*20}")
    test_actual, test_predicted = run_zero_shot_mode_test(allowed_words, train_descs, client, model)
    
    # Calculate accuracies
    train_accuracy = 0.0
    test_accuracy = 0.0
    
    if train_actual and train_predicted and len(train_actual) == len(train_predicted):
        train_correct = sum(1 for a, p in zip(train_actual, train_predicted) if a.lower() == p.lower())
        train_accuracy = train_correct / len(train_actual)
    
    if test_actual and test_predicted and len(test_actual) == len(test_predicted):
        test_correct = sum(1 for a, p in zip(test_actual, test_predicted) if a.lower() == p.lower())
        test_accuracy = test_correct / len(test_actual)
    
    # Generate confusion matrix if requested
    if confusion:
        # Use test data for confusion matrix (more meaningful for zero-shot)
        if test_actual and test_predicted:
            create_confusion_matrix(test_actual, test_predicted, "zero_shot")
    
    return {
        "method": "zero_shot",
        "num_words": len(allowed_words),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

def run_semantic_matching(num_words=1, seed: int = 42, out_dir: str = "results", confusion=False):
    """Wrapper function for main.py integration"""
    # Process both training and test data for proper accuracy calculation
    
    # Get vocabulary and references
    allowed_words = get_vocab_from_train_videos(video_dir_train)
    train_descs = load_train_descriptions(train_desc_file)
    
    # Limit words if specified
    if num_words and num_words < len(allowed_words):
        allowed_words = allowed_words[:num_words]
    
    client = genai.Client(vertexai=True, project="finetuning-gemini-on-psl", location="us-central1")
    model = "publishers/google/models/gemini-2.5-pro"
    
    # Process training videos
    print(f"\n{'='*20} Processing Training Videos {'='*20}")
    train_actual, train_predicted = run_semantic_shot_mode(allowed_words, train_descs, client, model)
    
    # Process test videos  
    print(f"\n{'='*20} Processing Test Videos {'='*20}")
    test_actual, test_predicted = run_semantic_shot_mode_test(allowed_words, train_descs, client, model)
    
    # Calculate accuracies
    train_accuracy = 0.0
    test_accuracy = 0.0
    
    if train_actual and train_predicted and len(train_actual) == len(train_predicted):
        train_correct = sum(1 for a, p in zip(train_actual, train_predicted) if a.lower() == p.lower())
        train_accuracy = train_correct / len(train_actual)
    
    if test_actual and test_predicted and len(test_actual) == len(test_predicted):
        test_correct = sum(1 for a, p in zip(test_actual, test_predicted) if a.lower() == p.lower())
        test_accuracy = test_correct / len(test_actual)
    
    # Generate confusion matrix if requested
    if confusion:
        # Use test data for confusion matrix (more meaningful for semantic matching)
        if test_actual and test_predicted:
            create_confusion_matrix(test_actual, test_predicted, "semantic_shot")
    
    return {
        "method": "semantic_shot",
        "num_words": len(allowed_words),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

if __name__ == "__main__":
    listener = threading.Thread(target=listen_for_stop, daemon=True)
    listener.start()
    main()
