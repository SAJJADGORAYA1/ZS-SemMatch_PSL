"""
MediaPipe Baseline for PSL Sign Language Recognition

This is a baseline method using MediaPipe keypoint extraction with Transformer and LSTM backends.
It is used for comparing our main method (zero-shot semantic matching) with traditional 
computer vision approaches.

Method: Extracts pose, hand, and face keypoints using MediaPipe, then processes them with
either a Transformer encoder or LSTM for classification.
"""

import os
import cv2
import torch
import mediapipe as mp
import numpy as np
import math
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

VIDEO_DIR = "data/Words_train"
SINGLE_TEST_VIDEO = "massage.mp4"
MAX_FRAMES = 30
KEYPOINT_DIM = (33 * 4) + (21 * 3) + (21 * 3) + (468 * 3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 2
MODEL_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 1
MODEL_PATH = "sign_transformer.pth"

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

class SignDataset(Dataset):
    def __init__(self, video_dir):
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

class SignTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes,
                 h=4, d_k=128, d_v=128, d_ff=2048,
                 n_layers=2, dropout_rate=0.1, max_seq_len=MAX_FRAMES):
        super().__init__()
        self.src_proj = nn.Linear(input_dim, model_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, MAX_FRAMES, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=h, dim_feedforward=d_ff,
            dropout=dropout_rate, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, src):
        x = self.src_proj(src)
        encoder_output = self.positional_embedding.repeat(x.size(0), 1, 1)
        encoded = self.encoder(x + encoder_output)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

class SignLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])

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

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
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

def run_mediapipe(num_words=None, backend="transformer", seed=42, epochs=20, batch_size=1, out_dir="results", confusion=False, loss=False, debug=False):
    import json
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    if debug:
        print(f"Using device: {DEVICE}")
        print(f"Keypoint dimension: {KEYPOINT_DIM}")
        print(f"Max frames: {MAX_FRAMES}")
    
    train_root = "data/Words_train"
    test_root = "data/Words_test"
    train_dataset = SignDataset(train_root)
    if len(train_dataset) == 0:
        print(f"No .mov files found in {train_root}")
        return
    
    if num_words is not None and num_words < len(train_dataset):
        unique_classes = list(set(train_dataset.labels))
        selected_classes = unique_classes[:num_words]
        
        filtered_indices = [i for i, label in enumerate(train_dataset.labels) if label in selected_classes]
        train_dataset.video_paths = [train_dataset.video_paths[i] for i in filtered_indices]
        train_dataset.labels = [train_dataset.labels[i] for i in filtered_indices]
        
        train_dataset.encoder = LabelEncoder()
        train_dataset.encoded_labels = train_dataset.encoder.fit_transform(train_dataset.labels)
        
        if debug:
            print(f"Limited dataset to {num_words} classes: {selected_classes}")
    
    print(f"Training dataset: {len(train_dataset)} videos, {len(set(train_dataset.encoded_labels))} classes")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    num_classes = len(set(train_dataset.encoded_labels))
    if backend.lower() == "lstm":
        model = SignLSTM(input_dim=KEYPOINT_DIM, hidden_dim=128, num_classes=num_classes).to(DEVICE)
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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    epoch_losses = []
    
    print(f"Training MediaPipe-{backend} model for {epochs} epochs on {num_classes} classes...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for x, y in pbar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                pbar.set_postfix({'Loss': f'{total_loss/total:.4f}',
                                  'Acc': f'{correct/total:.3f}' if total > 0 else '0.000'})
        
        epoch_loss = total_loss / total if total > 0 else 0.0
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={correct/total:.3f}")
    
    print("Evaluating on training set...")
    train_correct, train_total = 0, 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    print(f"Training Accuracy: {train_acc:.3f} ({train_correct}/{train_total})")
    test_dataset = SignDataset(test_root)
    if len(test_dataset) == 0:
        print(f"No .mov files found in {test_root}")
        return
    
    if num_words is not None:
        train_classes = set(train_dataset.labels)
        
        filtered_test_indices = [i for i, label in enumerate(test_dataset.labels) if label in train_classes]
        test_dataset.video_paths = [test_dataset.video_paths[i] for i in filtered_test_indices]
        test_dataset.labels = [test_dataset.labels[i] for i in filtered_test_indices]
        
        if debug:
            print(f"Filtered test dataset to {len(test_dataset)} videos from training classes")
    
    test_dataset.encoder = train_dataset.encoder
    test_dataset.encoded_labels = train_dataset.encoder.transform(test_dataset.labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.3f} ({test_correct}/{test_total})")
    
    if confusion or loss:
        actual_words = []
        predicted_words = []
        
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                pred = logits.argmax(1)
                
                for i in range(len(y)):
                    actual_word = test_dataset.encoder.inverse_transform([y[i].item()])[0]
                    predicted_word = test_dataset.encoder.inverse_transform([pred[i].item()])[0]
                    actual_words.append(actual_word)
                    predicted_words.append(predicted_word)
        
        if confusion:
            from utils.plot_utils import create_confusion_matrix
            create_confusion_matrix(actual_words, predicted_words, f"mediapipe_{backend}")
        
        if loss and epoch_losses:
            from utils.plot_utils import create_loss_graph
            create_loss_graph(epoch_losses, f"mediapipe_{backend}")
    
    return {"train_accuracy": train_acc, "test_accuracy": test_acc}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()
    run_mediapipe(backend=args.backend, seed=args.seed, out_dir=args.out_dir)
