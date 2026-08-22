#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

EXPECTED_WEIGHTS = [
    "logs/s101_224/model_final.pth",
    "logs/s101_256/model_final.pth",
    "logs/s101_288/model_final.pth",
    "logs/s200_224/model_final.pth",
]
EXPECTED_FEATURES = [
    f"logs/{name}/{kind}_f.npy"
    for name in ("s101_224", "s101_256", "s101_288", "s200_224")
    for kind in ("query", "gallery")
]
EXPECTED_DATA = ["data/test/test_data.csv", "data/test/test"]

def main():
    ap = argparse.ArgumentParser(description="Audit Pet-ReID-IMAG reproduction assets.")
    ap.add_argument("repo", nargs="?", default=".", help="Path to Pet-ReID-IMAG repo")
    ap.add_argument("--json", dest="json_path", help="Write JSON report")
    args = ap.parse_args()
    root = Path(args.repo).resolve()
    def stat(rel):
        p = root / rel
        return {"path": rel, "exists": p.exists(), "is_file": p.is_file(), "size": p.stat().st_size if p.is_file() else None}
    report = {
        "repo": str(root),
        "weights": [stat(x) for x in EXPECTED_WEIGHTS],
        "features": [stat(x) for x in EXPECTED_FEATURES],
        "data": [stat(x) for x in EXPECTED_DATA],
    }
    report["ready_for_four_model_inference"] = all(x["exists"] for x in report["weights"]) and all(x["exists"] for x in report["data"])
    report["ready_for_fusion_only"] = all(x["exists"] for x in report["features"]) and (root / "data/test/test_data.csv").exists()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
