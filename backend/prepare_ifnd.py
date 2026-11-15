"""Convert the Kaggle IFND file (Statement, Label, Web, Category, ...) into data/ifnd.csv
with the columns (text, label) that train.py picks up automatically. 0 = fake, 1 = real.

Put this file in backend/ and run from the project root:
  python backend/prepare_ifnd.py path/to/IFND.csv
  python backend/prepare_ifnd.py path/to/IFND.csv --export-splits   # also writes fake/real and train/test CSVs
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = {"true": 1, "real": 1, "fake": 0, "false": 0}


def read_any(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):         # Kaggle files are not always UTF-8
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit("Could not decode the CSV file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--export-splits", action="store_true")
    a = ap.parse_args()

    df = read_any(a.csv)
    print("Columns:", list(df.columns), "| rows:", len(df))
    print("Raw labels:\n", df["Label"].value_counts(dropna=False), "\n")

    # 1. keep only what we need and give it the names train.py expects
    df = df.rename(columns={"Statement": "text", "Label": "label", "Category": "category", "Web": "web"})
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABELS)
    if df["label"].isna().any():
        raise SystemExit("Unknown label values found. Add them to LABELS at the top of this file.")
    df["label"] = df["label"].astype(int)

    # 2. clean the text, drop empty / very short statements
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.split().str.len() >= 3]

    # 3. remove duplicates; remove statements that appear with BOTH labels (they confuse the model)
    conflict = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(conflict[conflict > 1].index)].drop_duplicates(subset="text")

    # 4. leakage checks: if one site or one opening word almost decides the label, the model will cheat
    print("Top sources per class (web):")
    print(pd.crosstab(df["web"], df["label"]).sort_values(0, ascending=False).head(6), "\n")
    first = df["text"].str.split().str[0].str.lower()
    print("Most common first word per class:")
    for lab, name in ((0, "fake"), (1, "real")):
        print(" ", name, first[df["label"] == lab].value_counts().head(5).to_dict())

    DATA.mkdir(exist_ok=True)
    out = df[["text", "label", "category"]]
    out.to_csv(DATA / "ifnd.csv", index=False)
    print(f"\nSaved {len(out)} rows -> data/ifnd.csv")
    print("Class balance:", out["label"].value_counts().to_dict(), "(0 = fake, 1 = real)")

    # 5. optional: separate fake/real files and a stratified 80:20 train/test split for your report.
    #    train.py makes its own split, so these are for documentation and manual experiments.
    if a.export_splits:
        sp = DATA / "splits"
        sp.mkdir(exist_ok=True)
        out[out["label"] == 0].to_csv(sp / "ifnd_fake.csv", index=False)
        out[out["label"] == 1].to_csv(sp / "ifnd_real.csv", index=False)
        tr, te = train_test_split(out, test_size=0.2, random_state=42, stratify=out["label"])
        tr.to_csv(sp / "ifnd_train.csv", index=False)
        te.to_csv(sp / "ifnd_test.csv", index=False)
        print(f"Splits written to data/splits/ (train {len(tr)}, test {len(te)})")


if __name__ == "__main__":
    main()  