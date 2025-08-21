# pip install torch torchvision opencv-python
import os, cv2, torch, torch.nn as nn, torch.optim as optim
import random, json, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from device_utils import get_best_device, to_device, device_info
from tqdm import tqdm

class VideoISLR3D(Dataset):
    def __init__(self, root, clip_len=16, size=112):
        self.samples = []  # List[Tuple[path, class_idx]]
        self.classes = []  # List[str]
        self.class_to_idx = {}

        entries = sorted(os.listdir(root))
        # Collect directory-structured classes
        dir_classes = [d for d in entries if os.path.isdir(os.path.join(root, d))]
        for c in dir_classes:
            self.class_to_idx.setdefault(c, len(self.classes))
            if c not in self.classes:
                self.classes.append(c)
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        # Also support flat layout: each .mp4 is its own class (video name)
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
                                         [0.22803, 0.22145, 0.216989])  # Kinetics-400 stats

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
            img = self.to_tensor(frames[j])         # [3,H,W]
            img = self.resize(img)                  # [3,112,112]
            img = self.norm(img)
            clip.append(img)
        x = torch.stack(clip, dim=1)                # [3,T,H,W]
        return x, y

    def __len__(self): return len(self.samples)

def make_model(num_classes, pretrained=True):
    if pretrained:
        m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    else:
        m = r3d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

class TinyC3D(nn.Module):
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

    def forward(self, x):              # x: [B,3,T,112,112]
        f = self.features(x)           # [B,512,1,1,1]
        return self.classifier(f.view(x.size(0), -1))


def _remap_subset(samples):
    """Remap labels in given sample list to a compact 0..K-1 space."""
    old_to_new = {}
    new_samples = []
    next_idx = 0
    for path, y in samples:
        if y not in old_to_new:
            old_to_new[y] = next_idx
            next_idx += 1
        new_samples.append((path, old_to_new[y]))
    return new_samples, next_idx


def run_c3d(num_words=1, use_pretrained=True, seed: int = 42, epochs: int = 20, batch_size: int = 1):
    """Run C3D pipeline on all words from Words_train, test on Words_test."""
    train_root = "data/Words_train"
    test_root = "data/Words_test"
    
    # Load training dataset - ALL words from Words_train
    train_ds = VideoISLR3D(train_root, clip_len=16, size=112)
    if len(train_ds) == 0:
        print(f"No videos found in {train_root}")
        return
    
    print(f"Training dataset: {len(train_ds)} videos, {len(train_ds.classes)} classes")
    print(f"Classes: {train_ds.classes}")
    
    # Create training loader with ALL data
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    num_classes = len(train_ds.classes)
    
    # Get best available device (CUDA > MPS > CPU)
    device = get_best_device(method_name="c3d")
    device_info()
    model = make_model(num_classes, pretrained=use_pretrained).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"Training C3D model for {epochs} epochs on {num_classes} classes...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for each epoch
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
    
    # Test on training set (to check overfitting)
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
    
    # Test on test set (different videos, same classes)
    print("Evaluating on test set...")
    test_ds = VideoISLR3D(test_root, clip_len=16, size=112)
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

    # Return results for main.py to handle
    return {
        "method": "c3d",
        "num_words": num_classes,  # Use actual number of classes
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
