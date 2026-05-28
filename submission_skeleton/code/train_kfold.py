
"""
5-fold training script with Temperature Scaling (TS) calibration.
- Uses OFFICIAL columns: filename, category_id
- Backbone default: convnext_tiny
- Saves fold checkpoints: model/best_model_fold{k}.pth
- Writes model/config.json with:
    {
      "model_name": "...",
      "num_classes": 100,
      "input_size": 600,
      "ckpt": ["model/best_model_fold0.pth", ...],
      "temperature": 1.0
    }

Usage:
    python code/train_kfold.py /path/to/train_dataset /path/to/train_labels.csv \
        --model_name convnext_tiny --size 600 --epochs 50 --batch_size 32
"""
import argparse, os, json, math
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from utils import set_seed, get_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import create_model

class FlowerDataset(Dataset):
    def __init__(self, root, df, size=600, train=False):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.tf = get_transforms(size=size, train=train, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        try:
            img = Image.open(os.path.join(self.root, r['filename'])).convert('RGB')
        except Exception as e:
            print(f"⚠️ Skipping broken image: {r['filename']} ({e})")
            return None
        x = self.tf(img)
        y = int(r['category_id'])
        return x, y

    # 用于过滤坏样本
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class _TS(nn.Module):
    """Temperature scaling module for logits calibration."""
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor([math.log(init_temp)], dtype=torch.float32))
    def forward(self, logits):
        t = torch.exp(self.log_t)
        return logits / t
    def temperature(self):
        return float(torch.exp(self.log_t).detach().cpu().item())

def fit_temperature(model, loader, device):
    """Fit temperature on validation set minimizing NLL."""
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            logits_list.append(logits)
            labels_list.append(y)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    ts = _TS(1.0).to(device)
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([ts.log_t], lr=0.5, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nll(ts(logits), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return ts.temperature()

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    tot, correct, total = 0.0, 0, 0
    for x,y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return tot/total, correct/total

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    tot, correct, total = 0.0, 0, 0
    for x,y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        tot += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return tot/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", type=str, help="path to train_dataset/")
    ap.add_argument("labels_csv", type=str, help="train_labels.csv")
    ap.add_argument("--model_name", type=str, default="efficientnet_b2")
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("model", exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    assert {'filename', 'category_id'}.issubset(df.columns), \
        "labels csv must have columns: filename, category_id"

    # === 🔹 自动映射 category_id → 0..num_classes-1 ===
    unique_ids = sorted(df['category_id'].unique().tolist())
    id2idx = {cid: i for i, cid in enumerate(unique_ids)}  # 原始ID→训练用索引
    idx2id = [int(cid) for cid in unique_ids]  # 训练用索引→原始ID

    df['category_id'] = df['category_id'].map(id2idx)  # 直接覆盖
    num_classes = len(unique_ids)

    print(f"[INFO] 类别数={num_classes} | 映射后范围={df['category_id'].min()}~{df['category_id'].max()}")

    # === 保存映射到 config.json ===
    # （在最后保存 cfg 的地方追加一行）
    # cfg["id2label"] = idx2id

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    ckpts = []
    temps = []

    for k,(tr_idx, va_idx) in enumerate(kf.split(df['filename'], df['category_id'])):
        print(f"\n==== Fold {k}/{args.folds} ====")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)
        train_ds = FlowerDataset(args.train_dir, train_df, size=args.size, train=True)
        val_ds   = FlowerDataset(args.train_dir, val_df,   size=args.size, train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, collate_fn=collate_fn)

        model = create_model(args.model_name, num_classes=num_classes, pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        best_acc, best_path = 0.0, f"model/best_model_fold{k}.pth"
        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
            scheduler.step()
            print(f"[Fold {k}] Epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(model.state_dict(), best_path)
        print(f"[Fold {k}] best val acc = {best_acc:.4f} | saved {best_path}")
        # reload best and fit temperature
        best_model = create_model(args.model_name, num_classes=num_classes, pretrained=False).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        temperature = fit_temperature(best_model, val_loader, device)
        print(f"[Fold {k}] calibrated temperature = {temperature:.4f}")
        ckpts.append(best_path); temps.append(temperature)

    # save a config using averaged temperature
    avg_temp = float(sum(temps)/len(temps)) if temps else 1.0
    cfg = {
        "model_name": args.model_name,
        "num_classes": num_classes,
        "input_size": args.size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "ckpt": ckpts,           # list of folds
        "temperature": avg_temp  # global temperature
    }
    cfg["id2label"] = idx2id

    with open("model/config.json","w",encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("Saved model/config.json with", len(ckpts), "checkpoints. Avg temperature:", avg_temp)

if __name__ == "__main__":
    main()
