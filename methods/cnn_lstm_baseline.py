"""
CNN-LSTM Baseline for PSL Sign Language Recognition

This is a baseline method using CNN-LSTM architecture for video classification.
It is used for comparing our main method (zero-shot semantic matching) with traditional 
deep learning approaches.

Method: Extracts frame-level features using InceptionV3 CNN, then processes temporal
sequences with LSTM for sign classification.
"""

import os, random, cv2, torch, torch.nn as nn, torch.optim as optim
import json, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from utils.device_utils import get_best_device, to_device, device_info
from utils.plot_utils import create_confusion_matrix, create_loss_graph
from tqdm import tqdm

class VideoISLRDataset(Dataset):
    def __init__(self, root, clip_len=24, size=299):
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        entries = sorted(os.listdir(root))
        for c in [d for d in entries if os.path.isdir(os.path.join(root, d))]:
            self.class_to_idx.setdefault(c, len(self.classes))
            if c not in self.classes:
                self.classes.append(c)
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        for f in entries:
            fpath = os.path.join(root, f)
            if os.path.isfile(fpath) and f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                stem = os.path.splitext(f)[0]
                self.class_to_idx.setdefault(stem, len(self.classes))
                if stem not in self.classes:
                    self.classes.append(stem)
                self.samples.append((fpath, self.class_to_idx[stem]))

        self.clip_len = clip_len
        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def _read_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_indices(self, n):
        if n <= self.clip_len:
            idx = list(range(n))
            idx += [n-1]*(self.clip_len-n)
            return idx
        step = n / self.clip_len
        return [int(i*step) for i in range(self.clip_len)]

    def __getitem__(self, i):
        vp, y = self.samples[i]
        frames = self._read_frames(vp)
        idxs = self._sample_indices(len(frames))
        clip = torch.stack([self.t(frames[j]) for j in idxs], dim=0)
        return clip, y

    def __len__(self): return len(self.samples)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, use_pretrained=True):
        super().__init__()
        if use_pretrained:
            m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        else:
            m = models.inception_v3(weights=None, aux_logits=True)
        self.backbone = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            m.Mixed_5b,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_dim = 256

    def forward(self, x):
        f = self.backbone(x)
        return torch.flatten(f, 1)

class CNNLSTM(nn.Module):
    def __init__(self, feat_dim=256, hidden=256, num_classes=10, use_pretrained=True):
        super().__init__()
        self.feat = InceptionFeatureExtractor(use_pretrained=use_pretrained)
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, clip):
        B,T,_,_,_ = clip.shape
        x = clip.view(B*T, 3, 299, 299)
        f = self.feat(x)
        f = f.view(B, T, -1)
        out, _ = self.lstm(f)
        logits = self.head(out[:, -1])
        return logits

def _remap_subset(samples):
    mapping = {}
    next_id = 0
    remapped = []
    for path, y in samples:
        if y not in mapping:
            mapping[y] = next_id
            next_id += 1
        remapped.append((path, mapping[y]))
    return remapped, next_id

def run_cnn_lstm(num_words=1, seed: int = 42, epochs: int = 20, batch_size: int = 1, use_pretrained=True, confusion=False, loss=False):
    train_root = "data/Words_train"
    test_root = "data/Words_test"
    
    train_ds = VideoISLRDataset(train_root, clip_len=24, size=299)
    if len(train_ds) == 0:
        print(f"No videos found in {train_root}")
        return
    
    print(f"Training dataset: {len(train_ds)} videos, {len(train_ds.classes)} classes")
    print(f"Classes: {train_ds.classes}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    num_classes = len(train_ds.classes)

    device = get_best_device(method_name="cnn_lstm")
    device_info()
    model = CNNLSTM(num_classes=num_classes, use_pretrained=use_pretrained).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = []
    
    print(f"Training CNN-LSTM model for {epochs} epochs on {num_classes} classes...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for clips, y in pbar:
                clips = to_device(clips, device)
                y = to_device(torch.tensor(y), device)
                
                optimizer.zero_grad()
                logits = model(clips)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                pbar.set_postfix({
                    'Loss': f'{total_loss/total:.4f}',
                    'Acc': f'{correct/total:.3f}' if total > 0 else '0.000'
                })
        
        epoch_loss = total_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.3f}")
    
    model.eval()
    
    train_correct=0; train_total=0
    with torch.no_grad():
        for clips, y in train_loader:
            clips = to_device(clips, device); y = to_device(torch.tensor(y), device)
            logits = model(clips)
            pred = logits.argmax(1)
            train_correct += (pred==y).sum().item(); train_total += y.size(0)
    train_acc = (train_correct/train_total) if train_total>0 else 0.0
    
    print("Evaluating on test set...")
    test_ds = VideoISLRDataset(test_root, clip_len=24, size=299)
    if len(test_ds) == 0:
        print(f"No videos found in {test_root}")
        return
    
    test_ds.class_to_idx = train_ds.class_to_idx
    test_ds.classes = train_ds.classes
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for clips, y in test_loader:
            clips = to_device(clips, device)
            y = to_device(torch.tensor(y), device)
            logits = model(clips)
            pred = logits.argmax(1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
    test_acc = (test_correct / test_total) if test_total > 0 else 0.0
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.3f} ({train_correct}/{train_total})")
    print(f"Test Accuracy: {test_acc:.3f} ({test_correct}/{test_total})")
    print(f"Classes: {num_classes}")

    if confusion or loss:
        actual_words = []
        predicted_words = []
        
        model.eval()
        with torch.no_grad():
            for clips, y in test_loader:
                clips = to_device(clips, device)
                y = to_device(torch.tensor(y), device)
                logits = model(clips)
                pred = logits.argmax(1)
                
                for i in range(len(y)):
                    actual_word = test_ds.classes[y[i].item()]
                    predicted_word = test_ds.classes[pred[i].item()]
                    actual_words.append(actual_word)
                    predicted_words.append(predicted_word)
        
        if confusion:
            create_confusion_matrix(actual_words, predicted_words, "cnn_lstm")
        
        if loss and epoch_losses:
            create_loss_graph(epoch_losses, "cnn_lstm")

    return {
        "method": "cnn_lstm",
        "num_words": num_classes,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "epochs": epochs
    }

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for clips, y in loader:
        clips, y = to_device(clips, device), to_device(torch.tensor(y), device)
        opt.zero_grad()
        logits = model(clips)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()*clips.size(0)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred==y).sum().item()
    return loss_sum/total, correct/total

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_words", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()
    run_cnn_lstm(num_words=args.num_words, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
