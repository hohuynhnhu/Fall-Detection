"""
inspect_model.py
----------------
Kiểm tra toàn bộ nội dung checkpoint best_model.pth

Cách dùng:
    python inspect_model.py --ckpt train_tranformer/dataset/checkpoints1/best_model.pth
"""
import torch
import numpy as np
import argparse
import os

def inspect(ckpt_path):
    print(f"{'='*60}")
    print(f"File: {ckpt_path}")
    print(f"Size: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── Keys trong checkpoint ────────────────────────────────────────────
    print(f"\n[1] Keys trong checkpoint:")
    for k, v in ckpt.items():
        if k == "model_state":
            print(f"  {k}: dict với {len(v)} layers")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # ── Config ───────────────────────────────────────────────────────────
    print(f"\n[2] Config:")
    cfg = ckpt.get("config", {})
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # ── Model state weights ───────────────────────────────────────────────
    print(f"\n[3] Layer weights:")
    state = ckpt["model_state"]
    total_params = 0
    for name, tensor in state.items():
        n = tensor.numel()
        total_params += n
        mean = tensor.float().mean().item()
        std  = tensor.float().std().item()
        vmin = tensor.float().min().item()
        vmax = tensor.float().max().item()
        print(f"  {name:<45} shape={str(list(tensor.shape)):<20} "
              f"mean={mean:>7.4f}  std={std:>6.4f}  "
              f"[{vmin:.3f}, {vmax:.3f}]")
    print(f"\n  Tổng params: {total_params:,}")

    # ── Kiểm tra trọng số có bình thường không ────────────────────────────
    print(f"\n[4] Kiểm tra sức khoẻ trọng số:")
    issues = []
    for name, tensor in state.items():
        t = tensor.float()
        if torch.isnan(t).any():
            issues.append(f"  NaN trong {name}")
        if torch.isinf(t).any():
            issues.append(f"  Inf trong {name}")
        if t.std() < 1e-6 and t.numel() > 1:
            issues.append(f"  Dead weights (std≈0) trong {name}")
        if t.abs().max() > 100:
            issues.append(f"  Trọng số rất lớn (max={t.abs().max():.1f}) trong {name}")

    if issues:
        print("  [WARN] Phát hiện vấn đề:")
        for i in issues:
            print(i)
    else:
        print("  [OK] Không có NaN, Inf, dead weights")

    # ── Kiểm tra scaler ───────────────────────────────────────────────────
    ckpt_dir = os.path.dirname(ckpt_path)
    mean_path = os.path.join(ckpt_dir, "scaler_mean.npy")
    std_path  = os.path.join(ckpt_dir, "scaler_std.npy")

    print(f"\n[5] Scaler files:")
    if os.path.exists(mean_path):
        m = np.load(mean_path)
        print(f"  scaler_mean.npy  shape={m.shape}  "
              f"range=[{m.min():.4f}, {m.max():.4f}]  mean={m.mean():.4f}")
    else:
        print(f"  [MISS] scaler_mean.npy không tìm thấy")

    if os.path.exists(std_path):
        s = np.load(std_path)
        print(f"  scaler_std.npy   shape={s.shape}  "
              f"range=[{s.min():.4f}, {s.max():.4f}]  mean={s.mean():.4f}")
        # Kiểm tra std = 0 → chia 0 khi inference
        zero_std = (s < 1e-6).sum()
        if zero_std > 0:
            print(f"  [WARN] {zero_std} features có std≈0 → sẽ bị clip về 1.0 khi normalize")
    else:
        print(f"  [MISS] scaler_std.npy không tìm thấy")

    # ── Kiểm tra X.npy đã scaled chưa ────────────────────────────────────
    x_path = os.path.join(os.path.dirname(ckpt_dir), "X.npy")
    if os.path.exists(x_path):
        X = np.load(x_path)
        print(f"\n[6] X.npy kiểm tra normalize:")
        print(f"  shape : {X.shape}")
        print(f"  mean  : {X.mean():.4f}  (nếu StandardScaled → gần 0)")
        print(f"  std   : {X.std():.4f}   (nếu StandardScaled → gần 1)")
        print(f"  min   : {X.min():.4f}")
        print(f"  max   : {X.max():.4f}")
        if abs(X.mean()) < 0.1 and 0.8 < X.std() < 1.2:
            print(f"  [OK] X.npy đã được StandardScaler transform")
        else:
            print(f"  [WARN] X.npy có vẻ CHƯA được StandardScaler transform")
            print(f"         mean xa 0 hoặc std xa 1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="dataset/checkpoints1/best_model.pth"
    )
    args = parser.parse_args()
    inspect(args.ckpt)