"""
C3D Baseline for PSL Sign Language Recognition

This is a baseline method using 3D Convolutional Networks for video classification.
It is used for comparing our main method (zero-shot semantic matching) with traditional 
deep learning approaches.

Method: Uses 3D CNNs (ResNet3D-18 or custom compact architecture) to process video
clips directly, learning spatiotemporal features for sign classification.
"""

import os, cv2, torch, torch.nn as nn, torch.optim as optim
import random, json, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from utils.device_utils import get_best_device, to_device, device_info
from utils.plot_utils import create_confusion_matrix, create_loss_graph
from tqdm import tqdm

class VideoISLR3D(Dataset):
    def __init__(self, root, clip_len=16, size=112):
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        entries = sorted(os.listdir(root))
        dir_classes = [d for d in entries if os.path.isdir(os.path.join(root, d))]
        for c in dir_classes:
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
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.43216, 0.394666, 0.37645],
                                         [0.22803, 0.22145, 0.216989])

    def _read_rgb(self, path):
        cap = cv2.VideoCapture(path); frames=[]
        while True:
            ok, f = cap.read()
            if not ok: break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        cap.release(); return frames

    def _sample_idx(self, n):
        if n <= self.clip_len: return list(range(n)) + [n-1]*(self.clip_len-n)
        step = n / self.clip_len
        return [int(i*step) for i in range(self.clip_len)]

    def __getitem__(self, i):
        vp, y = self.samples[i]
        frames = self._read_rgb(vp)
        idxs = self._sample_idx(len(frames))
        clip = []
        for j in idxs:
            img = self.to_tensor(frames[j])
            img = self.resize(img)
            img = self.norm(img)
            clip.append(img)
        x = torch.stack(clip, dim=1)
        return x, y

    def __len__(self): return len(self.samples)

def make_model(num_classes, pretrained=True):
    if pretrained:
        m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    else:
        m = r3d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

class CompactC3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(64,128,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(128,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(256,256,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(256,512,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(512,512,3,padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f.view(x.size(0), -1))

def _remap_subset(samples):
    old_to_new = {}
    new_samples = []
    next_idx = 0
    for path, y in samples:
        if y not in old_to_new:
            old_to_new[y] = next_idx
            next_idx += 1
        new_samples.append((path, old_to_new[y]))
    return new_samples, next_idx

def run_c3d(num_words=1, use_pretrained=True, seed: int = 42, epochs: int = 20, batch_size: int = 1, confusion=False, loss=False):
    train_root = "data/Words_train"
    test_root = "data/Words_test"
    
    train_ds = VideoISLR3D(train_root, clip_len=16, size=112)
    if len(train_ds) == 0:
        print(f"No videos found in {train_root}")
        return
    
    print(f"Training dataset: {len(train_ds)} videos, {len(train_ds.classes)} classes")
    print(f"Classes: {train_ds.classes}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    num_classes = len(train_ds.classes)
    
    device = get_best_device(method_name="c3d")
    device_info()
    model = make_model(num_classes, pretrained=use_pretrained).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = []
    
    print(f"Training C3D model for {epochs} epochs on {num_classes} classes...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for x, y in pbar:
                x = to_device(x, device)
                y = to_device(y, device)
                
                optimizer.zero_grad()
                logits = model(x)
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
    
    print("\nEvaluating on training set...")
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for x, y in train_loader:
            x = to_device(x, device)
            y = to_device(y, device)
            logits = model(x)
            pred = logits.argmax(1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
    train_acc = (train_correct / train_total) if train_total > 0 else 0.0
    
    print("Evaluating on test set...")
    test_ds = VideoISLR3D(test_root, clip_len=16, size=112)
    if len(test_ds) == 0:
        print(f"No videos found in {test_root}")
        return
    
    test_ds.class_to_idx = train_ds.class_to_idx
    test_ds.classes = train_ds.classes
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = to_device(x, device)
            y = to_device(y, device)
            logits = model(x)
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
            for x, y in test_loader:
                x = to_device(x, device)
                y = to_device(y, device)
                logits = model(x)
                pred = logits.argmax(1)
                
                for i in range(len(y)):
                    actual_word = test_ds.classes[y[i].item()]
                    predicted_word = test_ds.classes[pred[i].item()]
                    actual_words.append(actual_word)
                    predicted_words.append(predicted_word)
        
        if confusion:
            create_confusion_matrix(actual_words, predicted_words, "c3d")
        
        if loss and epoch_losses:
            create_loss_graph(epoch_losses, "c3d")

    return {
        "method": "c3d",
        "num_words": num_classes,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "epochs": epochs
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_words", type=int, default=1)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()
    run_c3d(num_words=args.num_words, use_pretrained=not args.no_pretrained, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
