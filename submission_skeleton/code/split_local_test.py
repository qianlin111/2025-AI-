"""
按类别分层把官方 train_labels.csv 划成训练集/验证集两份。
默认验证集10%，可调 --ratio。
"""
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def split_dataset(csv_path, ratio=0.1, seed=42):
    df = pd.read_csv(csv_path)
    assert {"filename", "category_id"}.issubset(df.columns), \
        "train_labels.csv 必须包含 filename, category_id 两列"

    sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    tr_idx, va_idx = next(sss.split(df["filename"], df["category_id"]))
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    df_tr.to_csv("train_split.csv", index=False, encoding="utf-8")
    df_va.to_csv("local_test_split.csv", index=False, encoding="utf-8")
    print(f"✅ 训练集: {len(df_tr)}  验证集: {len(df_va)} (ratio={ratio})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--ratio", type=float, default=0.1)
    args = parser.parse_args()
    split_dataset(args.csv_path, args.ratio)
