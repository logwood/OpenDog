"""Create an identity-disjoint, human-labeling queue without inventing labels.

The queue contains source-image records and ``unknown`` placeholders only. It
does not read the consumed blind split, infer labels from detections, or copy
images. Annotators must inspect the actual local face/nose crops produced by
the target graph and fill the strict provenance fields before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_local_visibility_data import SOURCES, read_json

WORKSPACE = Path(__file__).resolve().parents[3]
PARTS = ("face", "nose")
UNKNOWN = {"usable": None, "provenance": "unknown", "evidence": None}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def split_key(identity, seed):
    return hashlib.sha256(f"{seed}:{identity.casefold()}".encode()).hexdigest()


def load_candidates(workspace):
    manifests = {}
    for name, (role, relative) in SOURCES.items():
        if role != "train_pool":
            continue
        path = workspace / relative
        manifests[name] = (path, read_json(path))
    protected = set()
    protected_hashes = set()
    for name, (role, relative) in SOURCES.items():
        if role != "protected_development":
            continue
        payload = read_json(workspace / relative)
        protected |= {r["identity"].strip().casefold() for r in payload["records"]}
        protected_hashes |= {r["source_sha256"].lower() for r in payload["records"]}
    by_hash = {}
    for name, (path, payload) in manifests.items():
        source_hash = digest(path)
        for record in payload["records"]:
            identity = record["identity"].strip().casefold()
            image_hash = record["source_sha256"].lower()
            if identity in protected or image_hash in protected_hashes:
                continue
            old = by_hash.get(image_hash)
            if old and old["identity"].strip().casefold() != identity:
                raise ValueError("Conflicting identity labels for one source image")
            by_hash[image_hash] = {
                "identity": record["identity"],
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
                "source_manifest_sha256": source_hash,
                "source_manifest_path": str(path),
            }
    return list(by_hash.values())


def build_manifest(records, *, split, target_graph_sha256, source_paths):
    return {
        "schema_version": 1,
        "schema_name": "model_local_part_visibility",
        "split": split,
        "target_graph_sha256": target_graph_sha256,
        "source_manifests": [
            {"path": path, "sha256": digest(path)} for path in sorted(source_paths)
        ],
        "records": [
            {
                "identity": r["identity"],
                "source_path": r["source_path"],
                "source_sha256": r["source_sha256"],
                "source_manifest_sha256": r["source_manifest_sha256"],
                "local_visibility": {part: dict(UNKNOWN) for part in PARTS},
            }
            for r in records
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-graph-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-identities", type=int, default=0)
    args = parser.parse_args()
    if len(args.target_graph_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.target_graph_sha256):
        raise ValueError("target graph must be a lowercase SHA-256")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    candidates = load_candidates(WORKSPACE)
    identities = sorted({r["identity"].strip().casefold() for r in candidates}, key=lambda x: split_key(x, args.seed))
    if args.max_identities:
        identities = identities[: args.max_identities]
    selected = {identity for identity in identities}
    candidates = [r for r in candidates if r["identity"].strip().casefold() in selected]
    train_ids = {identity for identity in identities if int(split_key(identity, args.seed)[:2], 16) < 205}
    validation_ids = selected - train_ids
    if not train_ids or not validation_ids:
        raise ValueError("Identity split is empty; increase max-identities")
    train = [r for r in candidates if r["identity"].strip().casefold() in train_ids]
    validation = [r for r in candidates if r["identity"].strip().casefold() in validation_ids]
    source_paths = {r["source_manifest_path"] for r in candidates}
    output.mkdir(parents=True, exist_ok=False)
    queue = {
        "schema_version": 1,
        "status": "awaiting_human_model_crop_labels",
        "target_graph_sha256": args.target_graph_sha256,
        "seed": args.seed,
        "instructions": [
            "Run the target graph and inspect the actual face/nose crops, not the full source image.",
            "Use human provenance only after an annotator assesses each crop.",
            "Keep uncertain/occluded/out-of-semantic-view cases unknown; never call missing detections negative.",
            "Synthetic negatives require complete-part human extent evidence and exact pixel exclusion.",
        ],
        "candidate_records": len(candidates),
        "candidate_identities": len(selected),
        "train_records": len(train),
        "train_identities": len(train_ids),
        "validation_records": len(validation),
        "validation_identities": len(validation_ids),
        "source_manifests": sorted(source_paths),
        "protected_development_excluded": True,
        "blind_images_read": False,
    }
    (output / "queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "train.template.json").write_text(json.dumps(build_manifest(train, split="train", target_graph_sha256=args.target_graph_sha256, source_paths=source_paths), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "validation.template.json").write_text(json.dumps(build_manifest(validation, split="validation", target_graph_sha256=args.target_graph_sha256, source_paths=source_paths), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": queue["status"], "queue": str(output / "queue.json"), "train_records": len(train), "validation_records": len(validation)}), flush=True)


if __name__ == "__main__":
    main()
