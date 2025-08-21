import os
import cv2
import torch
import mediapipe as mp
import numpy as np
import math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from scipy.signal import find_peaks

# ========== Configuration ==========
VIDEO_DIR = "data/Words_train"
SINGLE_TEST_VIDEO = "massage.mp4"
CONTINUOUS_TEST_VIDEO = "body massage.mp4"
MAX_FRAMES = 30
KEYPOINT_DIM = (33 * 4) + (21 * 3) + (21 * 3) + (468 * 3)  # 1662
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 2
MODEL_DIM = 512
NUM_HEADS = 4
NUM_LAYERS = 2
MODEL_PATH = "sign_transformer.pth"

# Continuous recognition parameters
WORD_THRESHOLD = 0.8
MIN_WORD_DURATION = 8
MAX_TRANSITION = 15
SENTENCE_SMOOTHING = 5
MOTION_THRESHOLD = 0.15
STATIC_THRESHOLD = 0.02
BUFFER_SIZE = 30

"""MediaPipe baseline for PSL signs with Transformer and LSTM options."""

# ========== MediaPipe Keypoint Extractor ==========
class KeypointExtractor:
    def __init__(self):
        self.holistic = None
        self._init_failed = False

    def _ensure_init(self):
        if self.holistic is None and not self._init_failed:
            try:
                self.holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=False
                )
            except Exception:
                # Fallback to zeros if mediapipe initialization fails (e.g., platform issues)
                self._init_failed = True

    def extract(self, frame):
        self._ensure_init()
        if self._init_failed:
            return np.zeros(KEYPOINT_DIM, dtype=np.float32)
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        def get_landmarks(landmark_list, expected_len, dims=4):
            if landmark_list:
                data = []
                for lm in landmark_list.landmark:
                    if dims == 4:
                        data.extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)])
                    else:
                        data.extend([lm.x, lm.y, lm.z])
                if len(data) < expected_len:
                    data.extend([0.0] * (expected_len - len(data)))
                return data
            return [0.0] * expected_len

        pose = get_landmarks(results.pose_landmarks, 33 * 4, dims=4)
        left_hand = get_landmarks(results.left_hand_landmarks, 21 * 3, dims=3)
        right_hand = get_landmarks(results.right_hand_landmarks, 21 * 3, dims=3)
        face = get_landmarks(results.face_landmarks, 468 * 3, dims=3)

        keypoints = pose + left_hand + right_hand + face
        return np.array(keypoints, dtype=np.float32)

# ========== Dataset ==========
class SignDataset(Dataset):
    def __init__(self, video_dir):
        # Support flat .mp4 layout and class subfolders
        self.video_paths = []
        self.labels = []
        entries = sorted(os.listdir(video_dir))
        for e in entries:
            path = os.path.join(video_dir, e)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if f.lower().endswith('.mov'):
                        self.video_paths.append(os.path.join(path, f))
                        self.labels.append(e)
            elif os.path.isfile(path) and e.lower().endswith('.mov'):
                self.video_paths.append(path)
                self.labels.append(os.path.splitext(e)[0])

        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)
        self.extractor = KeypointExtractor()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx])
        frames = []
        while len(frames) < MAX_FRAMES and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            keypoints = self.extractor.extract(frame)
            frames.append(keypoints)
        cap.release()

        if len(frames) < MAX_FRAMES:
            padding = [np.zeros(KEYPOINT_DIM, dtype=np.float32)
                      for _ in range(MAX_FRAMES - len(frames))]
            frames.extend(padding)
        frames = np.stack(frames)
        label = self.encoded_labels[idx]
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ========== PyTorch Decoder Components ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        
    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_k, d_v, d_model):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, h * d_k)
        self.W_k = nn.Linear(d_model, h * d_k)
        self.W_v = nn.Linear(d_model, h * d_v)
        self.W_o = nn.Linear(h * d_v, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)
        return self.W_o(output)

class AddNormalization(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)

class FeedForward(nn.Module):
    def __init__(self, d_ff, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, h, d_k, d_v, d_model, d_ff, dropout_rate):
        super().__init__()
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.add_norm1 = AddNormalization(d_model)

        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.add_norm2 = AddNormalization(d_model)

        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.add_norm3 = AddNormalization(d_model)

    def forward(self, x, encoder_output, lookahead_mask=None, padding_mask=None):
        attn1 = self.multihead_attention1(x, x, x, lookahead_mask)
        attn1 = self.dropout1(attn1)
        addnorm1 = self.add_norm1(x, attn1)

        attn2 = self.multihead_attention2(addnorm1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2)
        addnorm2 = self.add_norm2(addnorm1, attn2)

        ff_output = self.feed_forward(addnorm2)
        ff_output = self.dropout3(ff_output)
        return self.add_norm3(addnorm2, ff_output)

class Decoder(nn.Module):
    def __init__(self, d_model, max_seq_len, h, d_k, d_v, d_ff, n_layers, dropout_rate):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(h, d_k, d_v, d_model, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])

    def forward(self, x, encoder_output, lookahead_mask=None, padding_mask=None):
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, lookahead_mask, padding_mask)
        return x

# ========== Model ==========
class SignTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes,
                 h=4, d_k=128, d_v=128, d_ff=2048,
                 n_layers=2, dropout_rate=0.1, max_seq_len=MAX_FRAMES):
        super().__init__()
        self.src_proj = nn.Linear(input_dim, model_dim)
        self.dummy_encoder = nn.Parameter(torch.randn(1, MAX_FRAMES, model_dim))
        self.decoder = Decoder(
            d_model=model_dim,
            max_seq_len=max_seq_len,
            h=h,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, src):
        x = self.src_proj(src)
        encoder_output = self.dummy_encoder.repeat(x.size(0), 1, 1)
        decoder_output = self.decoder(x, encoder_output)
        pooled = decoder_output.mean(dim=1)
        return self.classifier(pooled)


class SignLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])

# ========== Transition Detector ==========
class TransitionDetector:
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)
        self.motion_threshold = MOTION_THRESHOLD
        self.static_threshold = STATIC_THRESHOLD

    def update(self, keypoints):
        if not self.buffer:
            self.buffer.append(keypoints)
            return "initializing"

        # Calculate motion magnitude
        prev = np.array(self.buffer[-1])
        curr = np.array(keypoints)
        motion = np.mean(np.abs(curr - prev))

        # Update buffer
        self.buffer.append(keypoints)

        # Classify state
        if motion > self.motion_threshold:
            return "transition"
        elif motion < self.static_threshold:
            return "static"
        return "signing"

# ========== Sentence Predictor ==========
class SentencePredictor:
    def __init__(self, word_model, vocab, device):
        self.word_model = word_model
        self.vocab = vocab
        self.device = device
        self.reset()

    def reset(self):
        self.frame_buffer = []
        self.word_predictions = []
        self.confidence_scores = []
        self.transition_frames = 0
        self.current_word_start = None
        self.transition_detector = TransitionDetector()

    def process_frame(self, frame):
        # Extract keypoints
        extractor = KeypointExtractor()
        keypoints = extractor.extract(frame)

        # Update transition detector
        state = self.transition_detector.update(keypoints)

        # Add to frame buffer
        self.frame_buffer.append(keypoints)
        if len(self.frame_buffer) > MAX_FRAMES:
            self.frame_buffer.pop(0)

        # Check state conditions
        if state == "static" and self.current_word_start is not None:
            self._process_word_candidate()
        elif state == "transition":
            self.transition_frames += 1
        elif state == "signing":
            self.transition_frames = 0

        # Handle long transitions as word boundaries
        if self.transition_frames > MAX_TRANSITION and self.current_word_start is not None:
            self._process_word_candidate()

        return self.get_current_sentence()

    def _process_word_candidate(self):
        # Pad or trim to MAX_FRAMES
        frames = list(self.frame_buffer)
        if len(frames) < MAX_FRAMES:
            padding = [np.zeros_like(frames[0]) for _ in range(MAX_FRAMES - len(frames))]
            frames = padding + frames
        else:
            frames = frames[-MAX_FRAMES:]

        # Convert to tensor and predict
        input_tensor = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.word_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        # Only accept high-confidence predictions
        if conf.item() > WORD_THRESHOLD:
            word = self.vocab.inverse_transform([pred.item()])[0]
            self.word_predictions.append(word)
            self.confidence_scores.append(conf.item())

        # Reset state
        self.current_word_start = None
        self.frame_buffer = []
        self.transition_frames = 0

    def get_current_sentence(self, smoothing=SENTENCE_SMOOTHING):
        if not self.word_predictions:
            return ""

        # Apply smoothing to remove flickering
        if len(self.word_predictions) > smoothing:
            window = self.word_predictions[-smoothing:]
            counts = {}
            for word in window:
                counts[word] = counts.get(word, 0) + 1
            stable_word = max(counts, key=counts.get)

            if counts[stable_word] > smoothing // 2:
                self.word_predictions = self.word_predictions[:-smoothing] + [stable_word]

        return " ".join(self.word_predictions)

    def get_final_sentence(self):
        """Return the final predicted sentence after processing all frames"""
        # Process any remaining frames in the buffer
        if self.frame_buffer:
            self._process_word_candidate()

        # Apply final smoothing
        return self.get_current_sentence(smoothing=len(self.word_predictions))

# ========== Train ==========
def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# ========== Evaluation ==========
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return all_labels, all_preds

def plot_confusion_matrix(true_labels, pred_labels, classes, title='Confusion Matrix'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=classes))

# ========== Test Functions ==========
def test_single_video(model, test_path, label_encoder):
    extractor = KeypointExtractor()
    cap = cv2.VideoCapture(test_path)
    frames = []
    while len(frames) < MAX_FRAMES and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extractor.extract(frame)
        frames.append(keypoints)
    cap.release()
    
    if len(frames) < MAX_FRAMES:
        padding = [np.zeros(KEYPOINT_DIM, dtype=np.float32) 
                  for _ in range(MAX_FRAMES - len(frames))]
        frames.extend(padding)
    frames = np.stack(frames)
    input_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()
        print(f"Predicted Label: {label_encoder.inverse_transform([pred_label])[0]}")

def test_all_videos(model, dataset):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Testing on all videos in dataset...")
    true_labels, pred_labels = evaluate_model(model, dataloader)
    
    plot_confusion_matrix(true_labels, pred_labels, 
                         dataset.encoder.classes_,
                         title='Full Dataset Confusion Matrix')
    
    print("\nPer-video Results:")
    for i, video_path in enumerate(dataset.video_paths):
        true_label = dataset.labels[i]
        pred_label = dataset.encoder.inverse_transform([pred_labels[i]])[0]
        result = "CORRECT" if true_label == pred_label else "WRONG"
        print(f"{os.path.basename(video_path)}: True={true_label}, Pred={pred_label} -> {result}")

def test_continuous_signing(model, video_path, vocab):
    """Test model on continuous signing video without live display"""
    predictor = SentencePredictor(model, vocab, DEVICE)
    cap = cv2.VideoCapture(video_path)
    
    # Process all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        predictor.process_frame(frame)
            
    cap.release()
    
    # Get final predicted sentence
    sentence = predictor.get_final_sentence()
    print(f"Predicted Sentence: '{sentence}'")
    return sentence

# ========== Main ==========
def run_mediapipe(num_words=1, backend="transformer", seed: int = 42, epochs: int = 20, batch_size: int = 1, out_dir: str = "results"):
    """Run MediaPipe baseline with specified backend: transformer or lstm."""
    import json
    import time
    
    train_root = "data/Words_train"
    test_root = "data/Words_test"
    
    # Load training dataset - ALL words from Words_train
    train_dataset = SignDataset(train_root)
    if len(train_dataset) == 0:
        print(f"No .mov files found in {train_root}")
        return

    print(f"Training dataset: {len(train_dataset)} videos, {len(set(train_dataset.encoded_labels))} classes")
    print(f"Classes: {list(set(train_dataset.encoded_labels))}")
    
    # Create training loader with ALL data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    num_classes = len(set(train_dataset.encoded_labels))

    # Create model
    if backend.lower() == "lstm":
        model = SignLSTM(input_dim=KEYPOINT_DIM, hidden_dim=MODEL_DIM, num_classes=num_classes).to(DEVICE)
    else:
        model = SignTransformer(
            input_dim=KEYPOINT_DIM,
            model_dim=MODEL_DIM,
            num_classes=num_classes,
            h=NUM_HEADS,
            d_k=MODEL_DIM // NUM_HEADS,
            d_v=MODEL_DIM // NUM_HEADS,
            d_ff=MODEL_DIM * 4,
            n_layers=NUM_LAYERS,
            dropout_rate=0.1
        ).to(DEVICE)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"Training MediaPipe-{backend} model for {epochs} epochs on {num_classes} classes...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for each epoch
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for x, y in pbar:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                # Update progress bar with loss
                pbar.set_postfix({
                    'Loss': f'{total_loss/total:.4f}',
                    'Acc': f'{correct/total:.3f}' if total > 0 else '0.000'
                })
        
        # Print epoch summary
        epoch_loss = total_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.3f}")
    
    model.eval()
    
    # Test on training set
    print("Evaluating on training set...")
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
    
    train_acc = (train_correct / train_total) if train_total > 0 else 0.0
    
    # Test on test set (different videos, same classes)
    print("Evaluating on test set...")
    test_dataset = SignDataset(test_root)
    if len(test_dataset) == 0:
        print(f"No .mov files found in {test_root}")
        return
    
    # Important: Use the same class mapping as training
    test_dataset.encoder = train_dataset.encoder
    test_dataset.encoded_labels = train_dataset.encoded_labels
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
    
    test_acc = (test_correct / test_total) if test_total > 0 else 0.0
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.3f} ({train_correct}/{train_total})")
    print(f"Test Accuracy: {test_acc:.3f} ({test_correct}/{test_total})")
    print(f"Classes: {num_classes}")

    # Save results
    method_dir = os.path.join(out_dir, f"mediapipe_{backend}")
    os.makedirs(method_dir, exist_ok=True)
    out_path = os.path.join(method_dir, f"accuracy_seed{seed}_n{num_classes}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "method": f"mediapipe_{backend}",
            "num_words": num_classes,  # Use actual number of classes
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "epochs": epochs
        }, f, indent=2)
    print(f"Saved results -> {out_path}")
    
    # Return results for main.py to handle
    return {
        "method": f"mediapipe_{backend}",
        "num_words": num_classes,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "epochs": epochs
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_words", type=int, default=1)
    ap.add_argument("--backend", type=str, choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()
    run_mediapipe(num_words=args.num_words, backend=args.backend, seed=args.seed, out_dir=args.out_dir)