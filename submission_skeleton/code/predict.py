"""
推理脚本（评测执行时：python code/predict.py test_dataset results/submission.csv）
- 支持多模型平均 (logit 平均)
- 自动计算平均推理时间
- 避免一次性加载所有模型（顺序加载）
"""

import os
import sys
import json
import time
import torch
import timm
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F


# ============  数据加载  ============
def load_image(path, size=600, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)


# ============  模型创建  ============
def create_model(model_name, num_classes, ckpt_path):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.eval()
    return model


# ============  主逻辑  ============
def main():
    test_dir = sys.argv[1]
    output_csv = sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取 config.json
    with open("model/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]
    size = cfg.get("input_size", 600)
    ckpts = cfg["ckpt"]
    temp = cfg.get("temperature", 1.0)
    mean, std = tuple(cfg.get("mean", (0.485,0.456,0.406))), tuple(cfg.get("std", (0.229,0.224,0.225)))

    img_list = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))])
    all_preds, all_confs = [], []

    start_all = time.time()
    with torch.no_grad():
        for idx, img_name in enumerate(img_list):
            path = os.path.join(test_dir, img_name)
            x = load_image(path, size, mean, std).to(device)

            logits_list = []
            # 顺序加载每个模型 → 避免显存 & 时间过大
            for ckpt in ckpts:
                model = create_model(model_name, num_classes, ckpt).to(device)
                logits = model(x) / temp
                logits_list.append(logits)
                del model
                torch.cuda.empty_cache()

            # 平均 logits
            avg_logits = torch.mean(torch.stack(logits_list), dim=0)
            probs = F.softmax(avg_logits, dim=1)
            id2label = cfg.get("id2label", None)  # 从config读映射
            conf, pred = torch.max(probs, dim=1)

            # 🔹 反映射回官方编号
            raw_id = int(id2label[pred.item()]) if id2label is not None else int(pred.item())

            all_preds.append(raw_id)
            all_confs.append(conf.item())

    end_all = time.time()
    total_time = end_all - start_all
    avg_time = total_time / len(img_list)
    print(f"✅ 推理完成，共 {len(img_list)} 张图，总耗时 {total_time:.2f}s，平均 {avg_time*1000:.2f} ms/图")

    # 输出 CSV
    df = pd.DataFrame({
        "filename": img_list,
        "category_id": all_preds,
        "confidence": all_confs
    })
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"📁 已保存结果到: {output_csv}")


if __name__ == "__main__":
    main()
