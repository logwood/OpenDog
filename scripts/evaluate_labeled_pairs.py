#!/usr/bin/env python3
import argparse, csv, json
from sklearn.metrics import roc_auc_score

def main():
    ap=argparse.ArgumentParser(description="Compute ROC-AUC if you have a labeled pair CSV.")
    ap.add_argument("csv_path")
    ap.add_argument("--label", default="label")
    ap.add_argument("--score", default="prediction")
    args=ap.parse_args()
    y=[]; s=[]
    with open(args.csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            y.append(int(float(r[args.label]))); s.append(float(r[args.score]))
    print(json.dumps({"n":len(y), "roc_auc":float(roc_auc_score(y,s))}, indent=2))
if __name__ == "__main__": main()
