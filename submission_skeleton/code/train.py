"""
Stable training script (with early stopping & 9:1 split).
- 默认9:1划分防过拟合验证集
- 可选 --val_csv 手动指定验证集
- 自动保存最佳模型 best_model.pth
"""

import argparse, os, json
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import set_seed, get_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import create_model


# ====================== Dataset ======================
class FlowerDataset(Dataset):
    def __init__(self, root, df, size=600, train=False):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.tf = get_transforms(size=size, train=train,
                                 mean=IMAGENET_MEAN, std=IMAGENET_STD)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root, r['filename'])).convert('RGB')
        x = self.tf(img)
        y = int(r['category_id'])
        return x, y


# ====================== Train & Eval ======================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    tot, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return tot / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    tot, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        tot += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return tot / total, correct / total


# ====================== Main ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", type=str)
    ap.add_argument("labels_csv", type=str)
    ap.add_argument("--val_csv", type=str, default=None,
                    help="optional validation CSV (if specified, overrides auto split)")
    ap.add_argument("--model_name", type=str, default="efficientnet_b2")
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8, help="early stopping patience")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("model", exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    assert {'filename', 'category_id'}.issubset(df.columns)
    num_classes = df['category_id'].nunique()

    # --------- Split train/val (9:1) ---------
    if args.val_csv:
        val_df = pd.read_csv(args.val_csv)
        train_df = df[~df['filename'].isin(val_df['filename'])].reset_index(drop=True)
        print(f"使用外部验证集: {len(val_df)} 张")
    else:
        train_df, val_df = train_test_split(
            df, test_size=0.1, stratify=df['category_id'], random_state=args.seed
        )
        print(f"自动划分验证集: {len(train_df)} 训练, {len(val_df)} 验证 (9:1)")

    # --------- Datasets & Loaders ---------
    train_ds = FlowerDataset(args.train_dir, train_df, size=args.size, train=True)
    val_ds = FlowerDataset(args.train_dir, val_df, size=args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # --------- Model & Optimizer ---------
    model = create_model(args.model_name, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc, best_path = 0.0, "model/best_model.pth"
    patience_counter = 0

    # --------- Training Loop ---------
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停触发：连续 {args.patience} 轮验证未提升。")
                break

    cfg = {
        "model_name": args.model_name,
        "num_classes": num_classes,
        "input_size": args.size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "ckpt": best_path
    }
    with open("model/config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"训练完成，最佳验证准确率 {best_acc:.4f}，权重保存在 {best_path}")


if __name__ == "__main__":
    main()
