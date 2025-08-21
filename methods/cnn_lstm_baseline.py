# pip install torch torchvision opencv-python
import os, random, cv2, torch, torch.nn as nn, torch.optim as optim
import json, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from device_utils import get_best_device, to_device, device_info
from tqdm import tqdm

class VideoISLRDataset(Dataset):
    def __init__(self, root, clip_len=24, size=299):
        self.samples = []  # (video_path, class_idx)
        self.classes = []
        self.class_to_idx = {}

        entries = sorted(os.listdir(root))
        # Nested layout support
        for c in [d for d in entries if os.path.isdir(os.path.join(root, d))]:
            self.class_to_idx.setdefault(c, len(self.classes))
            if c not in self.classes:
                self.classes.append(c)
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        # Flat layout support
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
            # loop last frame if short
            idx += [n-1]*(self.clip_len-n)
            return idx
        step = n / self.clip_len
        return [int(i*step) for i in range(self.clip_len)]

    def __getitem__(self, i):
        vp, y = self.samples[i]
        frames = self._read_frames(vp)
        idxs = self._sample_indices(len(frames))
        clip = torch.stack([self.t(frames[j]) for j in idxs], dim=0)  # [T,3,H,W]
        return clip, y

    def __len__(self): return len(self.samples)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, use_pretrained=True):
        super().__init__()
        if use_pretrained:
            m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        else:
            m = models.inception_v3(weights=None, aux_logits=True)
        # keep everything up to the final pooling (2048-d)
        self.backbone = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out_dim = 2048

    def forward(self, x):                   # x: [B,3,299,299]
        f = self.backbone(x)                # [B,2048,1,1]
        return torch.flatten(f, 1)          # [B,2048]

class CNNLSTM(nn.Module):
    def __init__(self, feat_dim=2048, hidden=512, num_classes=10, use_pretrained=True):
        super().__init__()
        self.feat = InceptionFeatureExtractor(use_pretrained=use_pretrained)
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, clip):                # clip: [B,T,3,299,299]
        B,T,_,_,_ = clip.shape
        x = clip.view(B*T, 3, 299, 299)
        with torch.no_grad():               # freeze CNN for a quick baseline
            f = self.feat(x)                # [B*T,2048]
        f = f.view(B, T, -1)
        out, _ = self.lstm(f)               # [B,T,H]
        logits = self.head(out[:, -1])      # last timestep
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


def run_cnn_lstm(num_words=1, seed: int = 42, epochs: int = 20, batch_size: int = 1, use_pretrained=True):
    """Run CNN-LSTM pipeline on all words from Words_train, test on Words_test."""
    train_root = "data/Words_train"
    test_root = "data/Words_test"
    
    # Load training dataset - ALL words from Words_train
    train_ds = VideoISLRDataset(train_root, clip_len=24, size=299)
    if len(train_ds) == 0:
        print(f"No videos found in {train_root}")
        return
    
    print(f"Training dataset: {len(train_ds)} videos, {len(train_ds.classes)} classes")
    print(f"Classes: {train_ds.classes}")
    
    # Create training loader with ALL data
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    num_classes = len(train_ds.classes)

    # Get best available device (CUDA > MPS > CPU)
    device = get_best_device(method_name="cnn_lstm")
    device_info()
    model = CNNLSTM(num_classes=num_classes, use_pretrained=use_pretrained).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"Training CNN-LSTM model for {epochs} epochs on {num_classes} classes...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for each epoch
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
    train_correct=0; train_total=0
    with torch.no_grad():
        for clips, y in train_loader:
            clips = to_device(clips, device); y = to_device(torch.tensor(y), device)
            logits = model(clips)
            pred = logits.argmax(1)
            train_correct += (pred==y).sum().item(); train_total += y.size(0)
    train_acc = (train_correct/train_total) if train_total>0 else 0.0
    
    # Test on test set (different videos, same classes)
    print("Evaluating on test set...")
    test_ds = VideoISLRDataset(test_root, clip_len=24, size=299)
    if len(test_ds) == 0:
        print(f"No videos found in {test_root}")
        return
    
    # Important: Use the same class mapping as training
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

    # Return results for main.py to handle
    return {
        "method": "cnn_lstm",
        "num_words": num_classes,  # Use actual number of classes
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
