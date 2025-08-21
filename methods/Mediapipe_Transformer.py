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

# Config
VIDEO_DIR = "Words"
SINGLE_TEST_VIDEO = "massage.mp4"
MAX_FRAMES = 30
KEYPOINT_DIM = (33 * 4) + (21 * 3) + (21 * 3) + (468 * 3)  # 1662
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 80
BATCH_SIZE = 2
MODEL_DIM = 512
NUM_HEADS = 4
NUM_LAYERS = 2
MODEL_PATH = "sign_transformer.pth"

# ========== MediaPipe Keypoint Extractor ==========
class KeypointExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False)
    
    def extract(self, frame):
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        def get_landmarks(landmark_list, expected_len, dims=4):
            if landmark_list:
                data = []
                for lm in landmark_list.landmark:
                    if dims == 4:
                        data.extend([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
                    else:
                        data.extend([lm.x, lm.y, lm.z])
                if len(data) < expected_len:
                    data.extend([0.0] * (expected_len - len(data)))
                return data
            else:
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
        self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                            if f.endswith(".mp4")]
        self.labels = [os.path.splitext(os.path.basename(v))[0] for v in self.video_paths]
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

# FIXED: Changed nn.Mod极 to nn.Module
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

# ========== Test ==========
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

# ========== Test All Videos ==========
def test_all_videos(model, dataset):
    # Create dataloader without shuffling
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Testing on all videos in dataset...")
    true_labels, pred_labels = evaluate_model(model, dataloader)
    
    # Calculate and display metrics
    plot_confusion_matrix(true_labels, pred_labels, 
                         dataset.encoder.classes_,
                         title='Full Dataset Confusion Matrix')
    
    # Print per-video results
    print("\nPer-video Results:")
    for i, video_path in enumerate(dataset.video_paths):
        true_label = dataset.labels[i]
        pred_label = dataset.encoder.inverse_transform([pred_labels[i]])[0]
        result = "CORRECT" if true_label == pred_label else "WRONG"
        print(f"{os.path.basename(video_path)}: True={true_label}, Pred={pred_label} -> {result}")

# ========== Main ==========
def main():
    dataset = SignDataset(VIDEO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SignTransformer(
        input_dim=KEYPOINT_DIM,
        model_dim=MODEL_DIM,
        num_classes=len(set(dataset.encoded_labels)),
        h=NUM_HEADS,
        d_k=MODEL_DIM // NUM_HEADS,
        d_v=MODEL_DIM // NUM_HEADS,
        d_ff=MODEL_DIM * 4,
        n_layers=NUM_LAYERS,
        dropout_rate=0.1
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        train_model(model, dataloader, optimizer, criterion, EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    test_all_videos(model, dataset)
    
    # Test single video
    print("\nTesting on single test video:")
    test_single_video(model, SINGLE_TEST_VIDEO, dataset.encoder)

if __name__ == "__main__":
    main()