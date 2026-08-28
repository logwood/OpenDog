#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, tempfile
from pathlib import Path
import numpy as np

MODELS = ("s101_224", "s101_256", "s101_288", "s200_224")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)

def load_names(path: Path):
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def fuse(runs_root: Path, model_dirs=MODELS):
    qs, gs = [], []
    for m in model_dirs:
        q = np.load(runs_root / m / "query_f.npy")
        g = np.load(runs_root / m / "gallery_f.npy")
        if q.ndim != 2 or g.ndim != 2:
            raise ValueError(f"{m}: features must be 2D, got {q.shape=} {g.shape=}")
        if q.shape[1] != g.shape[1]:
            raise ValueError(f"{m}: query/gallery dim mismatch")
        qs.append(q.astype(np.float32, copy=False))
        gs.append(g.astype(np.float32, copy=False))
    qf = l2n(np.concatenate(qs, axis=1))
    gf = l2n(np.concatenate(gs, axis=1))
    return qf, gf

def score_pairs(runs_root: Path, pairs_csv: Path, query_names: Path, gallery_names: Path, output: Path, model_dirs=MODELS):
    qf, gf = fuse(runs_root, model_dirs)
    qnames, gnames = load_names(query_names), load_names(gallery_names)
    if len(qnames) != qf.shape[0] or len(gnames) != gf.shape[0]:
        raise ValueError(f"filename/feature count mismatch: {len(qnames)=}, {qf.shape[0]=}, {len(gnames)=}, {gf.shape[0]=}")
    qi, gi = {n:i for i,n in enumerate(qnames)}, {n:i for i,n in enumerate(gnames)}
    rows = []
    with pairs_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("pair CSV needs at least two columns")
        for row in reader:
            if len(row) < 2: continue
            a,b = row[0],row[1]
            if a not in qi or b not in gi:
                raise KeyError(f"pair image missing from filename lists: {a!r}, {b!r}")
            sim = float(np.dot(qf[qi[a]], gf[gi[b]]))
            rows.append(row + [sim * 100.0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header + ["prediction"]); w.writerows(rows)
    vals = [r[-1] for r in rows]
    return {"pairs": len(rows), "min": min(vals) if vals else None, "max": max(vals) if vals else None, "mean": float(np.mean(vals)) if vals else None, "output": str(output)}

def self_test():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        names = ["a.jpg", "b.jpg", "c.jpg"]
        # Four descriptor branches. Query/gallery are identical, so diagonal cosine must be 1.
        for j,m in enumerate(MODELS):
            d = 8 + j
            x = rng.normal(size=(3,d)).astype(np.float32)
            p = root / "runs" / m; p.mkdir(parents=True)
            np.save(p/"query_f.npy", x); np.save(p/"gallery_f.npy", x.copy())
        (root/"query_filename.txt").write_text("\n".join(names)+"\n")
        (root/"gallery_filename.txt").write_text("\n".join(names)+"\n")
        (root/"pairs.csv").write_text("imageA,imageB\na.jpg,a.jpg\nb.jpg,b.jpg\nc.jpg,c.jpg\na.jpg,b.jpg\n")
        out = root/"submit.csv"
        rep = score_pairs(root/"runs", root/"pairs.csv", root/"query_filename.txt", root/"gallery_filename.txt", out)
        with out.open() as f:
            r=list(csv.DictReader(f))
        diag=[float(r[i]["prediction"]) for i in range(3)]
        assert all(abs(x-100.0)<1e-4 for x in diag), diag
        assert -100.0001 <= float(r[3]["prediction"]) <= 100.0001
        return {"status":"PASS", "diagonal_scores":diag, "summary":rep}

def main():
    ap=argparse.ArgumentParser(description="Correct four-branch feature fusion + pair scoring for Pet-ReID-IMAG.")
    ap.add_argument("--workspace-root", "--root", dest="workspace_root", type=Path, default=WORKSPACE_ROOT)
    ap.add_argument("--runs-root", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--query-names", type=Path)
    ap.add_argument("--gallery-names", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--model-dirs",
        nargs=4,
        default=list(MODELS),
        metavar=("M224", "M256", "M288", "M200"),
        help="four directories below ROOT/logs containing query_f.npy and gallery_f.npy",
    )
    ap.add_argument("--self-test", action="store_true")
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    workspace_root = args.workspace_root.expanduser().resolve()
    runs_root = (args.runs_root or workspace_root / "artifacts" / "runs" / "legacy").expanduser().resolve()
    data_root = (args.data_root or workspace_root / "data" / "processed" / "pet-reid-imag").expanduser().resolve()
    pairs = (args.pairs or data_root / "test" / "test_data.csv").expanduser().resolve()
    query_names = (args.query_names or runs_root / "s101_224" / "query_filename.txt").expanduser().resolve()
    gallery_names = (args.gallery_names or runs_root / "s101_224" / "gallery_filename.txt").expanduser().resolve()
    output = (args.output or runs_root / "fusion_submit" / "submit_fixed.csv").expanduser().resolve()
    rep=score_pairs(
        runs_root,
        pairs,
        query_names,
        gallery_names,
        output,
        args.model_dirs,
    )
    print(json.dumps(rep, indent=2))

if __name__ == "__main__": main()
