# MHI_baseline.py
# Implementation of "Using Motion History Images with 3D Convolutional Networks in Isolated Sign Language Recognition"
# Based on Sincan & Keleş paper with three modes: attention, fusion, and baseline (plain I3D)
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from typing import List, Tuple, Optional
import math
import torch.optim as optim
from utils.device_utils import get_best_device, to_device, device_info
from utils.plot_utils import create_confusion_matrix, create_loss_graph
from tqdm import tqdm

# ========== Configuration ==========
VIDEO_DIR = "data/Words_train"
MAX_FRAMES = 32
FRAME_SIZE = 224
# Device will be set dynamically in run_mhi function
BATCH_SIZE = 1
NUM_WORKERS = 0

# MHI parameters
MHI_CHANNELS = 3  # B, G, R channels for temporal thirds
MHI_THRESHOLD = 30  # Motion threshold for MHI computation

# Model parameters
I3D_HIDDEN_DIM = 1024
MHI_HIDDEN_DIM = 1024
DROPOUT_RATE = 0.5

# Fusion weights (tune around these values)
FUSION_WEIGHTS = (0.6, 0.4)  # (w1*logits_i3d + w2*logits_mhi)

# ========== MHI Computation ==========
def compute_rgb_mhi(frames: List[np.ndarray]) -> np.ndarray:
    """
    Compute RGB Motion History Image by splitting video into three temporal thirds.
    Each channel represents motion from a different temporal segment.
    """
    if len(frames) < 3:
        # Pad with last frame if too short
        frames = frames + [frames[-1]] * (3 - len(frames))
    
    # Split into three equal temporal thirds
    third_length = len(frames) // 3
    thirds = [
        frames[:third_length],
        frames[third_length:2*third_length],
        frames[2*third_length:]
    ]
    
    # Compute MHI for each third
    mhi_channels = []
    for third in thirds:
        if len(third) < 2:
            # If third is too short, use the frame as is
            mhi = cv2.cvtColor(third[0], cv2.COLOR_RGB2GRAY)
        else:
            mhi = compute_single_mhi(third)
        mhi_channels.append(mhi)
    
    # Stack channels: B (oldest), G (middle), R (newest)
    rgb_mhi = np.stack(mhi_channels, axis=2)
    return rgb_mhi

def compute_single_mhi(frames: List[np.ndarray]) -> np.ndarray:
    """Compute single-channel MHI from a sequence of frames."""
    if len(frames) < 2:
        return cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    
    # Initialize MHI
    height, width = gray_frames[0].shape
    mhi = np.zeros((height, width), dtype=np.uint8)
    
    # Compute motion history
    for i in range(1, len(gray_frames)):
        # Frame difference
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        
        # Threshold motion
        motion_mask = diff > MHI_THRESHOLD
        
        # Update MHI: newer motion gets higher intensity
        mhi[motion_mask] = np.minimum(255, mhi[motion_mask] + 50)
        
        # Decay older motion
        mhi = cv2.erode(mhi, np.ones((3, 3), np.uint8))
    
    return mhi

# ========== Dataset ==========
class VideoISLRDataset(Dataset):
    def __init__(self, root, clip_len=MAX_FRAMES, size=FRAME_SIZE):
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
                if f.lower().endswith('.mov'):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        # Flat layout support
        for f in entries:
            fpath = os.path.join(root, f)
            if os.path.isfile(fpath) and f.lower().endswith('.mov'):
                stem = os.path.splitext(f)[0]
                self.class_to_idx.setdefault(stem, len(self.classes))
                if stem not in self.classes:
                    self.classes.append(stem)
                self.samples.append((fpath, self.class_to_idx[stem]))

        self.clip_len = clip_len
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _read_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.clip_len and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_indices(self, n):
        if n <= self.clip_len:
            idx = list(range(n))
            # Loop last frame if short
            idx += [n-1] * (self.clip_len - n)
            return idx
        step = n / self.clip_len
        return [int(i * step) for i in range(self.clip_len)]

    def __getitem__(self, i):
        vp, y = self.samples[i]
        frames = self._read_frames(vp)
        
        # Sample frames for 3D CNN
        idxs = self._sample_indices(len(frames))
        clip_frames = [frames[j] for j in idxs]
        
        # Convert to tensors and stack: [T, C, H, W] -> [C, T, H, W]
        clip_tensors = torch.stack([self.transform(frame) for frame in clip_frames], dim=0)  # [T, C, H, W]
        clip_tensors = clip_tensors.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        # Compute RGB-MHI
        rgb_mhi = compute_rgb_mhi(frames)
        mhi_tensor = self.transform(rgb_mhi)
        
        return clip_tensors, mhi_tensor, y

    def __len__(self):
        return len(self.samples)

# ========== Models ==========
class I3D(nn.Module):
    """Simplified 3D CNN backbone for video processing."""
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        
        # 3D convolution layers (simplified I3D-like architecture)
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                               stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Residual blocks (simplified)
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(512, num_classes)
        
        # Store intermediate features for attention
        self.features = {}

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        # Store intermediate features for attention mechanism
        if return_features:
            self.features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if return_features:
            self.features['conv1'] = x
        
        x = self.layer1(x)
        if return_features:
            self.features['layer1'] = x
        
        x = self.layer2(x)
        if return_features:
            self.features['layer2'] = x
        
        x = self.layer3(x)
        if return_features:
            self.features['layer3'] = x
        
        if return_features:
            return self.features
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class MHIResNet(nn.Module):
    """ResNet-50 based MHI classifier without GAP, keeping all conv blocks."""
    def __init__(self, num_classes):
        super().__init__()
        
        # Always use random initialization (no pretrained weights)
        resnet = models.resnet50(weights=None)
        
        # Remove the final layers (avgpool and fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Custom head as per paper: FC(1024, ReLU) -> FC(num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, MHI_HIDDEN_DIM)  # 2048 from ResNet-50
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(MHI_HIDDEN_DIM, num_classes)
        
        # Store intermediate features for attention
        self.intermediate_features = {}

    def forward(self, x, return_features=False):
        if return_features:
            self.intermediate_features = {}
        
        # Extract features from each stage
        x = self.features[0:5](x)  # conv1, bn1, relu, maxpool, layer1
        if return_features:
            self.intermediate_features['conv1'] = x
        
        x = self.features[5:6](x)  # layer2
        if return_features:
            self.intermediate_features['conv2'] = x
        
        x = self.features[6:7](x)  # layer3
        if return_features:
            self.intermediate_features['conv3'] = x
        
        x = self.features[7:8](x)  # layer4
        if return_features:
            self.intermediate_features['conv4'] = x
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class MHIAttentionModel(nn.Module):
    """Simplified MHI-Attention"""
    def __init__(self, num_classes):
        super().__init__()
        # Use a much smaller I3D backbone
        self.i3d = I3D(num_classes)
        
        # Simple MHI feature extractor (just a few conv layers)
        self.mhi_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # 32 channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 channels
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # Fixed size output
        )
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),   # Generate attention map
            nn.Sigmoid()  # Values between 0 and 1
        )
        
        # Simple classifier for attended features
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # I3D features are 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, video, mhi):
        # Force I3D to compute full forward pass (so weights get trained)
        _ = self.i3d(video)  
        
        # Get intermediate features
        feats = self.i3d(video, return_features=True)
        spatial_features = feats['layer3'].mean(dim=2)  # [B, 512, H, W]

        # Attention
        mhi_features = self.mhi_encoder(mhi)  
        attn = self.attention(mhi_features)
        attn = F.interpolate(attn, size=spatial_features.shape[-2:], mode='bilinear', align_corners=False)

        # Residual gating
        attended = spatial_features * (1 + attn)

        # Pool + classify
        pooled = F.adaptive_avg_pool2d(attended, (1, 1)).flatten(1)
        return self.classifier(pooled)


class MHIFusionModel(nn.Module):
    """Late-fusion variant: combine logits from I3D and MHI-ResNet."""
    def __init__(self, num_classes, fusion_weights=FUSION_WEIGHTS):
        super().__init__()
        self.i3d = I3D(num_classes)
        self.mhi_resnet = MHIResNet(num_classes)
        self.fusion_weights = fusion_weights
        
    def forward(self, video, mhi):
        # Get predictions from both models
        i3d_logits = self.i3d(video)
        mhi_logits = self.mhi_resnet(mhi)
        
        # Late fusion: w1*logits_i3d + w2*logits_mhi
        fused_logits = (self.fusion_weights[0] * i3d_logits + 
                       self.fusion_weights[1] * mhi_logits)
        
        return fused_logits

# ========== Helper Functions ==========
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

# ========== Main Function ==========
def run_mhi(num_words=1, mode="baseline", seed: int = 42, epochs: int = 20, batch_size: int = 1, confusion=False, loss=False):
    """Run MHI baseline with specified mode: attention, fusion, or baseline (plain I3D)."""
    train_root = "data/Words_train"
    test_root = "data/Words_test"
    
    # Create model based on mode
    if mode == "attention":
        method_name = "mhi_attention"
    elif mode == "fusion":
        method_name = "mhi_fusion"
    else:  # baseline
        method_name = "mhi_baseline"
    
    # Load training dataset - ALL words from Words_train
    train_dataset = VideoISLRDataset(train_root, clip_len=MAX_FRAMES, size=FRAME_SIZE)
    if len(train_dataset) == 0:
        print(f"No .mov files found in {train_root}")
        return
    
    print(f"Training dataset: {len(train_dataset)} videos, {len(train_dataset.classes)} classes")
    print(f"Classes: {train_dataset.classes}")
    
    # Create training loader with ALL data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    num_classes = len(train_dataset.classes)
    
    # Load test dataset (different videos, same classes)
    test_dataset = VideoISLRDataset(test_root, clip_len=MAX_FRAMES, size=FRAME_SIZE)
    if len(test_dataset) == 0:
        print(f"No .mov files found in {test_root}")
        return
    
    # Important: Use the same class mapping as training
    test_dataset.class_to_idx = train_dataset.class_to_idx
    test_dataset.classes = train_dataset.classes
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Get best available device (CUDA > MPS > CPU)
    device = get_best_device(method_name=method_name)
    device_info()
    
    # Create model
    if mode == "attention":
        model = MHIAttentionModel(num_classes).to(device)
    elif mode == "fusion":
        model = MHIFusionModel(num_classes).to(device)
    else:  # baseline
        model = I3D(num_classes).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate for faster overfitting
    
    # Track losses for plotting
    epoch_losses = []
    
    # Training loop
    print(f"Training MHI-{mode} model for {epochs} epochs on {num_classes} classes...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for each epoch
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for video, mhi, y in pbar:
                video = to_device(video, device)
                mhi = to_device(mhi, device)
                y = to_device(torch.tensor(y), device)
                
                optimizer.zero_grad()
                
                if mode == "baseline":
                    logits = model(video)
                else:
                    logits = model(video, mhi)
                
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
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.3f}")
    
    model.eval()
    
    # Test on training set
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for video, mhi, y in train_loader:
            video = to_device(video, device)
            mhi = to_device(mhi, device)
            y = to_device(torch.tensor(y), device)
            
            if mode == "baseline":
                logits = model(video)
            else:
                logits = model(video, mhi)
            
            pred = logits.argmax(1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
    
    train_acc = (train_correct / train_total) if train_total > 0 else 0.0
    
    # Test on test set (different videos, same classes)
    print("Evaluating on test set...")
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for video, mhi, y in test_loader:
            video = to_device(video, device)
            mhi = to_device(mhi, device)
            y = to_device(torch.tensor(y), device)
            
            if mode == "baseline":
                logits = model(video)
            else:
                logits = model(video, mhi)
            
            pred = logits.argmax(1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
    
    test_acc = (test_correct / test_total) if test_total > 0 else 0.0
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.3f} ({train_correct}/{train_total})")
    print(f"Test Accuracy: {test_acc:.3f} ({test_correct}/{test_total})")
    print(f"Classes: {num_classes}")

    # Generate plots if requested
    if confusion or loss:
        # Collect actual and predicted words for confusion matrix
        actual_words = []
        predicted_words = []
        
        # Get predictions for confusion matrix
        model.eval()
        with torch.no_grad():
            for video, mhi, y in test_loader:
                video = to_device(video, device)
                mhi = to_device(mhi, device)
                y = to_device(torch.tensor(y), device)
                
                if mode == "baseline":
                    logits = model(video)
                else:
                    logits = model(video, mhi)
                
                pred = logits.argmax(1)
                
                # Convert indices to words
                for i in range(len(y)):
                    actual_word = test_dataset.classes[y[i].item()]
                    predicted_word = test_dataset.classes[pred[i].item()]
                    actual_words.append(actual_word)
                    predicted_words.append(predicted_word)
        
        # Create confusion matrix if requested
        if confusion:
            create_confusion_matrix(actual_words, predicted_words, method_name)
        
        # Create loss graph if requested
        if loss and epoch_losses:
            create_loss_graph(epoch_losses, method_name)

    # Return results for main.py to handle
    return {
        "method": method_name,
        "num_words": num_classes,  # Use actual number of classes
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "epochs": epochs
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_words", type=int, default=1)
    ap.add_argument("--mode", type=str, choices=["attention", "fusion", "baseline"], default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()
    
    run_mhi(num_words=args.num_words, mode=args.mode, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
