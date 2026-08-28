#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = WORKSPACE_ROOT / "artifacts" / "runs" / "legacy"
DATA_ROOT = WORKSPACE_ROOT / "data" / "processed" / "pet-reid-imag"
EXPECTED_WEIGHTS = [f"{name}/model_final.pth" for name in (
    "s101_224", "s101_256", "s101_288", "s200_224"
)]
EXPECTED_FEATURES = [
    f"{name}/{kind}_f.npy"
    for name in ("s101_224", "s101_256", "s101_288", "s200_224")
    for kind in ("query", "gallery")
]
EXPECTED_DATA = ["test/test_data.csv", "test/test"]

def main():
    ap = argparse.ArgumentParser(description="Audit Pet-ReID-IMAG reproduction assets.")
    ap.add_argument("workspace", nargs="?", type=Path, default=WORKSPACE_ROOT)
    ap.add_argument("--json", dest="json_path", help="Write JSON report")
    args = ap.parse_args()
    workspace = args.workspace.expanduser().resolve()
    runs_root = workspace / RUNS_ROOT.relative_to(WORKSPACE_ROOT)
    data_root = workspace / DATA_ROOT.relative_to(WORKSPACE_ROOT)
    def stat(root, rel):
        p = root / rel
        return {"path": rel, "exists": p.exists(), "is_file": p.is_file(), "size": p.stat().st_size if p.is_file() else None}
    report = {
        "workspace": str(workspace),
        "weights": [stat(runs_root, x) for x in EXPECTED_WEIGHTS],
        "features": [stat(runs_root, x) for x in EXPECTED_FEATURES],
        "data": [stat(data_root, x) for x in EXPECTED_DATA],
    }
    report["ready_for_four_model_inference"] = all(x["exists"] for x in report["weights"]) and all(x["exists"] for x in report["data"])
    report["ready_for_fusion_only"] = all(x["exists"] for x in report["features"]) and (data_root / "test/test_data.csv").exists()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
