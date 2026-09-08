"""Verify old 177/256 replay and the new *untrained* one-graph engineering path.

No visibility labels are generated and no optimizer is created. Synthetic RGB
probes exercise initialized-off and manually-enabled gates; those probe weights
are NOT learned candidates or accuracy evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import torch

from evaluate_frozen_pretrained_comparison import (
    ROOT, WORKSPACE, load_development_protocol, paired_comparison,
    retrieval_metrics, sha256_file, write_json,
)
from evaluate_local_path_candidate import raw_tensor, make_session, parity, RELEASE_HASH
from pet_id.unified_local_path import LocalPathUnifiedPetReIDExport, load_local_path_checkpoint
from pet_id.unified_local_visibility import (
    LocalVisibilityUnifiedPetReID, load_local_visibility_checkpoint,
    save_untrained_visibility_checkpoint,
)

OLD_CHECKPOINT_HASH = "32a03eb67b18858772a96e60a10167b9b9bc3923bfb032577e2199bf3107662e"


def log(**record):
    print(json.dumps(record), flush=True)


def export_probe(model, examples, output, mode):
    path = output / f"visibility_{mode}_engineering_probe.onnx"
    torch.onnx.export(
        LocalPathUnifiedPetReIDExport(model).eval(), (examples[0],), path,
        input_names=["rgb"], output_names=["embedding"],
        dynamic_axes={"rgb": {0: "batch", 2: "height", 3: "width"}, "embedding": {0: "batch"}},
        opset_version=20, dynamo=False, external_data=False,
    )
    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph)
    contract = {
        "inputs": [item.name for item in graph.graph.input],
        "outputs": [item.name for item in graph.graph.output],
        "external_initializers": sum(x.data_location == onnx.TensorProto.EXTERNAL for x in graph.graph.initializer),
        "nodes": len(graph.graph.node),
        "node_domains": sorted({x.domain for x in graph.graph.node}),
        "visibility_head_present": any("visibility_head" in node.name for node in graph.graph.node),
        "nose_encoder_present": any("nose_encoder" in node.name for node in graph.graph.node),
        "refiner_present": any("refiner" in node.name for node in graph.graph.node),
    }
    if contract["inputs"] != ["rgb"] or contract["outputs"] != ["embedding"] or contract["external_initializers"]:
        raise ValueError("Visibility probe is not a self-contained single RGB graph")
    if not all(contract[key] for key in ("visibility_head_present", "nose_encoder_present", "refiner_present")):
        raise ValueError("A required module disappeared from the exported graph")
    del graph
    gc.collect()
    with torch.inference_mode():
        references = [model(rgb).cpu().numpy() for rgb in examples]
    checks = []
    for provider in ("cpu", "cuda"):
        session = make_session(path, provider)
        for source, expected in zip(examples, references):
            actual = session.run(["embedding"], {"rgb": source.cpu().numpy()})[0]
            check = {"mode": mode, "provider": provider, "shape": list(source.shape), **parity(actual, expected)}
            checks.append(check)
            log(phase="onnx_parity", **check)
            if not check["passed"] or actual.shape != (source.shape[0], 512):
                raise ValueError("Visibility dynamic ONNX parity failed")
            np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-5)
        del session
        gc.collect()
    return {"path": str(path), "sha256": sha256_file(path), "contract": contract, "checks": checks, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(20260908)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Actual released-model replay requires the checked local CUDA GPU")
    device = torch.device("cuda")
    registry = WORKSPACE / "models/registry.json"
    registry_hash = sha256_file(registry)
    old_dir = WORKSPACE / "artifacts/runs/pretrained_comparison/step2_local_path_e2e_20260906"
    old_checkpoint = old_dir / "candidate.pth"
    if sha256_file(old_checkpoint) != OLD_CHECKPOINT_HASH:
        raise ValueError("Historical local-path checkpoint differs from the recorded candidate")
    release = WORKSPACE / "models/selected/unified_pet_reid_v3_v1/model_final.pth"
    if sha256_file(release) != RELEASE_HASH:
        raise ValueError("Production V3 source differs from the frozen reference")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running", "training_steps": 0, "new_visibility_accuracy_claim": False,
        "production_changed": False, "blind_images_read": False,
        "registry_sha256_before": registry_hash,
        "source_v3_sha256": RELEASE_HASH, "legacy_checkpoint_sha256": OLD_CHECKPOINT_HASH,
        "code_sha256": {name: sha256_file(ROOT / name) for name in (
            "pet_id/unified_local_path.py", "pet_id/unified_local_visibility.py",
            "pet_id/local_visibility_data.py", "tools/verify_local_visibility.py",
        )},
    }
    write_json(output / "verification.json", report)
    try:
        _, records, audit = load_development_protocol(
            WORKSPACE / "artifacts/runs/unified/v3/external_bifor_protocol_20260831/protocol_lock.json"
        )
        with np.load(old_dir / "features.npz", allow_pickle=False) as saved:
            if saved["source_sha256"].tolist() != [r["source_sha256"] for r in records]:
                raise ValueError("Historical feature order differs from development manifest")
            expected = saved["local_path_candidate"].copy()
        legacy, legacy_payload = load_local_path_checkpoint(old_checkpoint, device=device)
        log(phase="legacy_checkpoint_restored", config_migrated=legacy_payload["model_config"] != legacy.configuration())
        actual = []
        with torch.inference_mode():
            for index, record in enumerate(records):
                actual.append(legacy(raw_tensor(record, device))[0].cpu())
                if (index + 1) % 64 == 0 or index == 0:
                    log(phase="legacy_177_replay", images=index + 1, total=len(records))
        actual = torch.stack(actual)
        ids, paths = [r["identity"].casefold() for r in records], [r["source_path"] for r in records]
        metrics = retrieval_metrics(actual, ids, paths, include_queries=True)
        previous = retrieval_metrics(torch.from_numpy(expected), ids, paths, include_queries=True)
        paired = paired_comparison(metrics, previous)
        replay_parity = parity(actual.numpy(), expected)
        if metrics["top1_correct"] != 177 or paired["fixed_count"] or paired["harmed_count"] or not replay_parity["passed"]:
            raise ValueError("Legacy checkpoint failed to reproduce the original 177/256 result")
        report["legacy_replay"] = {
            "data_audit": audit, "metrics": metrics, "paired_to_saved": paired,
            "parity_to_saved": replay_parity, "passed": True,
        }
        write_json(output / "verification.json", report)
        log(phase="legacy_replay_passed", top1=metrics["top1_correct"], **replay_parity)
        model = LocalVisibilityUnifiedPetReID(legacy.model).to(device).eval()
        model.configure_visibility_training()
        report["trainable_parameters"] = {
            "visibility_head": sum(p.numel() for p in model.visibility_head.parameters() if p.requires_grad),
            "released_base": sum(p.numel() for p in model.model.parameters() if p.requires_grad),
        }
        examples = [torch.rand(shape, device=device) * 255 for shape in (
            (1, 3, 73, 121), (2, 3, 121, 73), (1, 3, 2, 3),
        )]
        with torch.inference_mode():
            checks = [model(rgb, return_aux=True) for rgb in examples]
        if not all(torch.equal(x["embedding"], x["face_descriptor"]) for x in checks):
            raise ValueError("Actual model failed exact initialized local-face fallback")
        report["initialized_exact_fallback"] = True
        checkpoint = output / "untrained_visibility_engineering_only.pth"
        save_untrained_visibility_checkpoint(model, checkpoint, source_v3_checkpoint=release)
        restored, _ = load_local_visibility_checkpoint(checkpoint, device=device)
        with torch.inference_mode():
            if not torch.equal(restored(examples[0]), checks[0]["embedding"]):
                raise ValueError("Actual visibility checkpoint roundtrip changed its output")
        report["new_checkpoint_roundtrip"] = {"passed": True, "sha256": sha256_file(checkpoint)}
        del restored, checks
        gc.collect()
        torch.cuda.empty_cache()
        report["target_graph_sha256"] = hashlib.sha256(json.dumps({
            "source_v3_sha256": RELEASE_HASH, "local_path_config": model.configuration()["local_path"],
            "local_path_code_sha256": report["code_sha256"]["pet_id/unified_local_path.py"],
        }, sort_keys=True).encode("utf-8")).hexdigest()
        report["onnx_probes"] = []
        for mode, bias in (("initialized_off", -2.0), ("manually_enabled_not_trained", 2.0)):
            with torch.no_grad():
                for classifier in model.visibility_head.classifiers:
                    classifier[-1].bias.fill_(bias)
            log(phase="export_start", mode=mode)
            report["onnx_probes"].append(export_probe(model, examples, output, mode))
            write_json(output / "verification.json", report)
        if sha256_file(registry) != registry_hash:
            raise ValueError("Production registry changed during verification")
        report.update(status="engineering_verified_training_data_blocked", elapsed_seconds=time.monotonic() - started,
                      registry_sha256_after=sha256_file(registry))
        write_json(output / "verification.json", report)
        log(status=report["status"], report=str(output / "verification.json"))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "verification.json", report)
        raise


if __name__ == "__main__":
    main()
