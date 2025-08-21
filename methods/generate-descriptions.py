from google import genai
from google.genai import types
import os
import threading
import time
import signal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
FINETUNED_MODEL_NAME = os.getenv("FINETUNED_MODEL_NAME")

# Ensure GOOGLE_APPLICATION_CREDENTIALS is set
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    default_adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(default_adc):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_adc
        print(f"Using ADC file: {default_adc}")
        print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    else:
        raise RuntimeError("No GOOGLE_APPLICATION_CREDENTIALS set and no ADC file found.")

video_dir_train = "data/Words_train"
video_dir_test = "data/Words_test"
gcs_prefix_train = "gs://psl-train-clipped/train/"
gcs_prefix_test = "gs://psl-test/"

os.makedirs("outputs", exist_ok=True)
SAVE_FILENAME = "outputs/descriptions_train-clipped-32.txt"

PROMPT_TEMPLATE = """
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

# Collect all possible words (labels) from train
all_words = [
    os.path.splitext(f)[0]
    for f in os.listdir(video_dir_train)
    if f.endswith(".MOV")
]

def get_test_filename(word):
    # Capitalize first letter for test set
    return word[0].upper() + word[1:] + ".MOV"

stop_flag = False

def save_description(word, split, description):
    filename = SAVE_FILENAME
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{word}: {description.strip()}\n")

def generate_for_all_videos():
    global stop_flag

    # Clear the file at the start of each run
    with open(SAVE_FILENAME, "w", encoding="utf-8") as f:
        pass

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

    model = "publishers/google/models/gemini-2.5-pro"

    for word in sorted(all_words):
        if stop_flag:
            print("Stopping as requested by user.")
            break

        train_filename = word + ".MOV"
        train_path = os.path.join(video_dir_train, train_filename)

        test_filename = get_test_filename(word)
        test_path = os.path.join(video_dir_test, test_filename)

        for split, video_path in [("train", train_path)]:
            if stop_flag:
                print("Stopping as requested by user.")
                break
            
            # Use GCS URI like zero_shot_semantic_matching.py does
            train_gcs_uri = f"{gcs_prefix_train}{train_filename}"
            print(f"Using GCS URI: {train_gcs_uri}")
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=train_gcs_uri,
                                mime_type="video/MOV"
                            )
                        ),
                        types.Part(text=PROMPT_TEMPLATE),
                    ]
                )
            ]

            generate_content_config = types.GenerateContentConfig(
                temperature=0.0,
                top_p=1,
                seed=0,
                max_output_tokens=12000,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                media_resolution="MEDIA_RESOLUTION_LOW",
            )

            output = ""
            try:
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if stop_flag:
                        print("Stopping as requested by user.")
                        break
                    if chunk.text is not None:
                        output += chunk.text
                print(f"{word} [{split}] -> {output.strip()}\n\n\n")
                save_description(word, split, output)
            except Exception as e:
                print(f"Error processing {word} [{split}]: {e}")

def handle_sigint(sig, frame):
    global stop_flag
    stop_flag = True
    print("\nCtrl-C pressed. Stopping gracefully...")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    print("Press Ctrl-C to stop the script.")
    generate_for_all_videos()
