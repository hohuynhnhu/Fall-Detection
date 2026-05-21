"""
train_transformer.py
Train Transformer binary classifier: fall vs non_fall
- Early Stopping
- Label Smoothing
- Cosine Annealing với Warm Restart
- Dropout + Weight Decay tối ưu
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os
import json

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR     = r"D:\luan_van\Fall-Detection\train_tranformer\dataset"
SAVE_DIR     = r"D:\luan_van\Fall-Detection\train_tranformer\dataset\checkpoints"

# Hyperparameters đã tối ưu
D_MODEL      = 128
NHEAD        = 4
NUM_LAYERS   = 2     # giảm từ 3 → 2 để tránh overfit
DROPOUT      = 0.4   # tăng từ 0.3 → 0.4
BATCH_SIZE   = 16    # giảm từ 32 → 16 để generalize tốt hơn
EPOCHS       = 100
LR           = 5e-4  # giảm từ 1e-3 → 5e-4
WEIGHT_DECAY = 5e-3  # tăng mạnh để regularize
PATIENCE     = 15    # early stopping sau 15 epochs không cải thiện
LABEL_SMOOTH = 0.1   # label smoothing tránh overconfident

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = os.path.join(DATA_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Config loaded từ extract_keypoints:")
        print(f"  num_frames = {cfg['num_frames']}")
        print(f"  input_dim  = {cfg['input_dim']}")
        if cfg.get("fusion"):
            print(f"  fusion     = MediaPipe({cfg.get('mp_kp',33)} kp) + YOLO({cfg.get('yolo_kp',17)} kp)")
        return cfg
    else:
        print("WARNING: config.json không tìm thấy → dùng giá trị mặc định")
        return {"num_frames": 30, "input_dim": 132, "fusion": False}

# ── Dataset ───────────────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(X: np.ndarray, y: np.ndarray,
            input_dim: int, n_aug: int = 3) -> tuple:
    """
    Giảm n_aug từ 4 → 3 để tránh overfit do augmented data quá giống nhau
    """
    X_aug, y_aug = [X], [y]

    for _ in range(n_aug):
        X_new = X.copy()

        # 1. Horizontal flip
        mask      = np.random.rand(len(X_new)) > 0.5
        X_flip    = X_new.copy()
        x_indices = np.arange(0, input_dim, 4)
        X_flip[mask, :, :][:, :, x_indices] = \
            1.0 - X_flip[mask, :, :][:, :, x_indices]
        X_new = X_flip

        # 2. Speed change
        factor    = np.random.choice([0.8, 0.9, 1.1, 1.2])
        T         = X_new.shape[1]
        src_idx   = np.linspace(0, T-1, int(T * factor))
        src_idx   = np.clip(src_idx, 0, T-1).astype(int)
        final_idx = np.linspace(0, len(src_idx)-1, T, dtype=int)
        X_new     = X_new[:, src_idx, :][:, final_idx, :]

        # 3. Gaussian noise — tăng sigma để model robust hơn
        X_new = X_new + np.random.normal(0, 0.01, X_new.shape)

        # 4. Random frame dropout — mask 10% frames về 0
        drop_mask = np.random.rand(X_new.shape[0], X_new.shape[1]) < 0.1
        X_new[drop_mask] = 0.0

        X_aug.append(X_new.astype(np.float32))
        y_aug.append(y.copy())

    return np.concatenate(X_aug), np.concatenate(y_aug)

# ── Model ─────────────────────────────────────────────────────────────────────

class PoseTransformer(nn.Module):
    def __init__(self, input_dim: int, num_frames: int,
                 d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dropout=DROPOUT,
                 num_classes=2):
        super().__init__()

        # Input projection với dropout ngay từ đầu
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # dropout nhẹ ở input
        )

        self.pos_emb   = nn.Parameter(
            torch.randn(1, num_frames + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        B   = x.size(0)
        x   = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_emb
        x   = self.transformer(x)
        return self.head(x[:, 0])

# ── Train ─────────────────────────────────────────────────────────────────────

def train():
    # Bước 1: Load config
    cfg        = load_config()
    num_frames = cfg["num_frames"]
    input_dim  = cfg["input_dim"]

    # Bước 2: Load data
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    N, T, D = X.shape

    print(f"\nDataset loaded:")
    print(f"  X shape  : {X.shape}")
    print(f"  fall     : {int((y==1).sum())}")
    print(f"  non_fall : {int((y==0).sum())}")

    assert T == num_frames, f"Frame mismatch: X={T} vs config={num_frames}"
    assert D == input_dim,  f"Dim mismatch: X={D} vs config={input_dim}"

    # Bước 3: Normalize
    X_flat = X.reshape(-1, D)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X      = X_flat.reshape(N, T, D).astype(np.float32)
    np.save(os.path.join(SAVE_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(SAVE_DIR, "scaler_std.npy"),  scaler.scale_)
    print("Scaler saved.")

    # Bước 4: Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split: Train={len(X_train)}  Val={len(X_val)}")

    # Bước 5: Augment (n_aug=3 thay vì 4)
    X_train, y_train = augment(X_train, y_train, input_dim=input_dim, n_aug=3)
    print(f"After augmentation: Train={len(X_train)}")
    print(f"  fall={int((y_train==1).sum())}  non_fall={int((y_train==0).sum())}")

    # Bước 6: DataLoader
    train_dl = DataLoader(
        PoseDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(
        PoseDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Bước 7: Model + Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = PoseTransformer(
        input_dim=input_dim,
        num_frames=num_frames,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")
    print(f"Model input: ({BATCH_SIZE}, {num_frames}, {input_dim})")

    # Bước 8: Loss với Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # AdamW + Cosine Annealing Warm Restart
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # Bước 9: Train loop với Early Stopping
    best_val_acc     = 0.0
    patience_counter = 0
    history          = {"train_acc": [], "val_acc": [], "train_loss": [], "lr": []}

    print(f"\n{'='*70}")
    print(f"{'Epoch':>6} {'Loss':>8} {'Train%':>8} {'Val%':>8} {'Best%':>8} {'LR':>10}")
    print(f"{'='*70}")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == yb).sum().item()

        scheduler.step()
        train_acc  = correct / len(train_dl.dataset) * 100
        train_loss = total_loss / len(train_dl)
        cur_lr     = optimizer.param_groups[0]["lr"]

        # Validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_correct += model(xb).argmax(1).eq(yb).sum().item()

        val_acc = val_correct / len(val_dl.dataset) * 100

        history["train_acc"].append(round(train_acc, 2))
        history["val_acc"].append(round(val_acc, 2))
        history["train_loss"].append(round(train_loss, 4))
        history["lr"].append(round(cur_lr, 6))

        # Early Stopping check
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            marker           = " ←"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "config": {
                    "input_dim":  input_dim,
                    "num_frames": num_frames,
                    "d_model":    D_MODEL,
                    "nhead":      NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dropout":    DROPOUT,
                }
            }, os.path.join(SAVE_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            marker            = ""

        print(f"{epoch:>6} {train_loss:>8.4f} {train_acc:>7.1f}% "
              f"{val_acc:>7.1f}% {best_val_acc:>7.1f}% {cur_lr:>10.6f}{marker}")

        # Dừng nếu không cải thiện sau PATIENCE epochs
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping tại epoch {epoch} "
                  f"(không cải thiện sau {PATIENCE} epochs)")
            break

    # Lưu history
    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy : {best_val_acc:.1f}%")
    print(f"Saved             : {SAVE_DIR}/best_model.pth")

    # Load best model để evaluate
    checkpoint = torch.load(os.path.join(SAVE_DIR, "best_model.pth"),
                            map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    evaluate(model, val_dl, device)

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, val_dl, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb    = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (best model)")
    print("="*50)
    print(classification_report(
        all_labels, all_preds,
        target_names=["non_fall", "fall"]
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print("CONFUSION MATRIX")
    print(f"               Pred non_fall  Pred fall")
    print(f"True non_fall:     {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"True fall:         {cm[1][0]:>6}       {cm[1][1]:>6}")

    # Tính thêm metrics quan trọng cho fall detection
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) * 100  # recall của fall
    specificity = tn / (tn + fp) * 100  # recall của non_fall
    print(f"\nFall Detection Metrics:")
    print(f"  Sensitivity (fall recall) : {sensitivity:.1f}%")
    print(f"  Specificity (non_fall rec): {specificity:.1f}%")
    print(f"  False Alarm Rate          : {fp/(fp+tn)*100:.1f}%")
    print(f"  Miss Rate                 : {fn/(fn+tp)*100:.1f}%")


if __name__ == "__main__":
    train()