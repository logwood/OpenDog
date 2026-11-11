"""Offline validation: annotated video, clean best frames, JSONL and HTML review."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from .models import ASSETS, DogFaceModels, clip_box, ensure_models, file_hash
from .pipeline import CapturePipeline
from .runtime import peak_rss_mb

ROOT = Path(__file__).resolve().parents[3]


def save_image(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    data.tofile(path)


def draw_overlay(frame, record, target_size=960):
    h, w = frame.shape[:2]
    scale = min(1.0, target_size / max(h, w))
    out = cv2.resize(frame, (round(w * scale) // 2 * 2, round(h * scale) // 2 * 2))
    sx, sy = out.shape[1] / w, out.shape[0] / h
    if record.get("box"):
        x1, y1, x2, y2 = record["box"]
        cv2.rectangle(out, (round(x1 * sx), round(y1 * sy)), (round(x2 * sx), round(y2 * sy)), (50, 210, 240), 2)
        for i, (x, y) in enumerate(record["landmarks"]):
            color = (190, 120, 60) if i < 14 else (70, 240, 130) if i < 24 else (60, 190, 255)
            cv2.circle(out, (round(x * sx), round(y * sy)), 2, color, -1)
    state = record["tracking"]
    status = state["stable_state"].upper()
    cue = record["pose"]["yaw_cue"]
    lines = [f"POSE {status}  cue {cue:+.2f}  |  NEXT {state['target'].upper()}  {state['step']}/{state.get('step_count', 5)}",
             f"Face focus {record['quality']['laplacian']:.1f}  |  {'OK' if record['quality']['accepted'] else ', '.join(record['quality']['reasons'])}",
             f"Frame {record['frame_index']}  t={record['time_s']:.2f}s  analysis={record['compute_ms']:.1f}ms"]
    for k, text in enumerate(lines):
        y = 25 + 26 * k
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1)
    return out


def write_review(output, report):
    labels = {"front": "正面", "left": "正向脸部坐标：左", "right": "正向脸部坐标：右"}
    cards = []
    for pose, selected in report["selected"].items():
        q = selected["quality"]
        cards.append(f'<article><h3>{labels[pose]}</h3><img src="{selected["face_file"]}"><p>视频 {selected["time_s"]:.2f} 秒 · 第 {selected["frame_index"]} 帧</p><p>清晰度 {q["laplacian"]:.1f} · 质量分 {q["score"]:.2f}</p><a href="{selected["frame_file"]}">查看原始完整帧</a></article>')
    elapsed = report["performance"]["analysis_wall_seconds"]
    source = html.escape(report["input"])
    attribution = html.escape(json.dumps(report.get("source_attribution", {}), ensure_ascii=False, indent=2))
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>犬只转头与清晰帧验证</title><style>body{{font:16px/1.6 system-ui,"Microsoft Yahei",sans-serif;background:#f4f6f8;color:#1e293b;margin:0}}main{{max-width:1100px;margin:32px auto;padding:0 24px}}h1{{font-size:28px}}video{{width:100%;max-height:600px;background:#111;border-radius:12px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}}article,.stats{{background:white;padding:20px;border-radius:12px;margin:18px 0}}img{{width:100%;height:260px;object-fit:contain;background:#e9edf1}}.muted{{color:#64748b;overflow-wrap:anywhere}}a{{color:#0f766e}}pre{{white-space:pre-wrap;font-size:12px}}code{{background:#e2e8f0;padding:3px 6px}}</style>
<main><p class="muted">采集流程原型 · 仅使用视频帧</p><h1>转头判定与最佳帧</h1><p class="muted">{source}</p>
<div class="stats">处理 {report['frames_analyzed']} 帧 / {report['video_seconds']:.1f} 秒视频，分析用时 {elapsed:.1f} 秒。<br>计算耗时中位数 {report['performance']['compute_ms_p50']:.1f} ms，P95 {report['performance']['compute_ms_p95']:.1f} ms。<br>动作顺序：正面 → 左 → 正面 → 右 → 正面。完成状态：{'完成' if report['sequence_complete'] else '尚未完成'}。<br>识别主图：{'已选取正面帧' if report['recognition_input'] else '未找到合格正面帧，请补拍'}。</div>
<video controls preload="metadata" src="{report['preview_file']}"></video><div class="grid">{''.join(cards) or '<p>没有可用帧。</p>'}</div>
<p>左右以转正后的脸部坐标为准，转正方向改变会重置动作序列。质量分只用于同一段跟踪的候选比较；完整帧保存原始像素，脸部裁剪仅作整 90 度旋转，没有锐化或增强。</p>
<p><a href="result.json">完整结果和识别输入接口</a> · <a href="frames.jsonl">逐帧结果</a></p><details><summary>视频来源与授权</summary><pre>{attribution}</pre></details></main></html>'''
    (output / "review.html").write_text(body, encoding="utf-8")


def run(args):
    source = Path(args.video).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    views_mode = getattr(args, 'mode', 'challenge') == 'views'
    upstream_path = getattr(args, 'upstream_jsonl', None)
    models_dir = (Path(args.models) if args.device == 'trt' else ensure_models(args.models, download=args.download_models)) if not upstream_path else None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or ROOT / "artifacts/runs/dog_capture" / f"{stamp}-{source.stem}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve previous runs: {output}")
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video frame rate unavailable; transcode to a constant frame rate first")
    models = (DogFaceModels(models_dir, threads=args.threads, dog_threshold=args.dog_threshold,
                           landmark_period=args.landmark_period, device=args.device,
                           compact_confidence=views_mode and not getattr(args, 'full_heatmaps', False),
                           trt_directory=getattr(args, 'trt_engines', None), trt_cuda_graph=not getattr(args, 'no_cuda_graph', False))
              if models_dir is not None else None)
    upstream = None
    if views_mode:
        from .views import TargetViewPipeline
        from .view_io import ObservationReader, load_targets
        pipeline = TargetViewPipeline(models, targets=load_targets(getattr(args, 'target_config', None)),
                                      min_face_pixels=args.min_face_pixels, min_laplacian=args.min_laplacian)
        if upstream_path:
            upstream = ObservationReader(upstream_path)
    else:
        pipeline = CapturePipeline(models, args.hold, args.side_threshold, args.min_face_pixels, args.min_laplacian)
    states, rejects = Counter(), Counter()
    timings, stage_times = [], []
    writer = None
    frame_index, analysed, last, next_time = 0, 0, None, 0.0
    period = 1 / args.fps if args.fps > 0 else 0.0
    started = time.perf_counter()
    jsonl = (output / "frames.jsonl").open("w", encoding="utf-8")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            if args.max_seconds > 0 and timestamp >= args.max_seconds:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            if timestamp + 1e-6 >= next_time:
                last = (pipeline.process(frame, timestamp, frame_index, observation=upstream.get(frame_index))
                        if upstream else pipeline.process(frame, timestamp, frame_index))
                jsonl.write(json.dumps(last, ensure_ascii=False, allow_nan=False) + "\n")
                states[last['tracking']["stable_state"]] += 1
                rejects.update(last['quality']['reasons'])
                timings.append(last['compute_ms'])
                stage_times.append(last['timings_ms'])
                analysed += 1
                next_time = timestamp + period
                if analysed % 100 == 0:
                    print(f"{timestamp:.1f}s: {analysed} frames, median {np.median(timings):.1f} ms, best {list(pipeline.snapshot()['frames'])}", flush=True)
            if not args.no_preview:
                shown = draw_overlay(frame, last if last['frame_index'] == frame_index else {**last, 'box': None, 'landmarks': None})
                if writer is None:
                    writer = cv2.VideoWriter(str(output / "preview_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (shown.shape[1], shown.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Video encoder failed")
                writer.write(shown)
            frame_index += 1
    finally:
        capture.release()
        jsonl.close()
        if writer:
            writer.release()
        if upstream:
            upstream.close()
        if sys.exc_info()[0] is not None and models:
            models.close()
    if views_mode:
        try:
            pipeline.finalize()
        finally:
            if models:
                models.close()
    elif models:
        models.close()
    loop_seconds = time.perf_counter() - started
    if not analysed:
        raise ValueError("Video contains no decodable frames")
    snapshot = pipeline.snapshot()
    selected = {}
    for state, item in snapshot["frames"].items():
        record, original = item["record"], item["image"]
        frame_file, face_file = f"best_{state}_frame.png", f"best_{state}_face.png"
        save_image(output / frame_file, original)
        x1, y1, x2, y2 = clip_box(record["box"], original.shape)
        save_image(output / face_file, np.rot90(original[y1:y2, x1:x2], record.get('rotation_quarters', 0)))
        selected[state] = {"frame_index": record["frame_index"], "time_s": record["time_s"],
                           "quality": record["quality"], "pose": record["pose"],
                           'rotation_quarters': record.get('rotation_quarters', 0),
                           'selection_basis': record.get('selection_basis', 'confirmed_pose_dwell'),
                           "box_xyxy": record["box"], "frame_file": frame_file, "face_file": face_file}
    preview = "preview_raw.mp4"
    if writer and shutil.which("ffmpeg"):
        process = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(output / preview),
                                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                                  "-movflags", "+faststart", str(output / "preview.mp4")], capture_output=True, text=True)
        if process.returncode == 0:
            preview = "preview.mp4"
            (output / "preview_raw.mp4").unlink()
    attribution = {}
    source_manifest = ROOT / "assets/dog_capture/sources.json"
    if source_manifest.exists():
        attribution = json.loads(source_manifest.read_text(encoding="utf-8")).get(source.name, {})
    fixture_manifest = ROOT / 'assets/dog_capture/fixtures.json'
    if not attribution and fixture_manifest.exists():
        attribution = json.loads(fixture_manifest.read_text(encoding='utf8')).get(source.name, {})
    sequence_complete = snapshot['complete'] if not views_mode else False
    report = {"schema_version": 2 if views_mode else 1, "mode": 'views' if views_mode else 'challenge', "input": str(source), "input_sha256": file_hash(source),
              "output": str(output), "video_fps": fps, "video_seconds": frame_index / fps,
              "frames_decoded": frame_index, "frames_analyzed": analysed, "sampling_fps": args.fps,
              "directions": "nose left/right in upright normalized face coordinates; orientation change resets sequence",
              "selected_segment": snapshot["segment"], "sequence_complete": sequence_complete,
              "events": list(pipeline.events if views_mode else pipeline.tracker.events), "stable_frame_counts": dict(states), "rejected_frames_by_reason": dict(rejects),
              "selected": selected,
              "recognition_input": {"image_path": str(output / selected["front"]["frame_file"]),
                                    "face_path": str(output / selected["front"]["face_file"]),
                                    "box_xyxy": selected["front"]["box_xyxy"], "frame_index": selected["front"]["frame_index"],
                                    "sequence_complete": sequence_complete} if "front" in selected else None,
              "performance": {"analysis_wall_seconds": loop_seconds, "analysis_fps": analysed / loop_seconds,
                              "compute_ms_p50": float(np.median(timings)), "compute_ms_p95": float(np.percentile(timings, 95)),
                              "stage_mean_ms": {key: float(np.mean([t.get(key, 0) for t in stage_times])) for key in ("detector", "localizer", "landmarks", "tracking")},
                              "threads": args.threads, "device": models.device if models else 'upstream observations; zero additional inference', "includes_preview_encoding_in_analysis_wall": not args.no_preview,
                              "model_calls": dict(models.calls) if models else {}},
              "parameters": vars(args), "model_sha256": {name: sha for name, (_, sha) in ASSETS.items()} if models else {},
              "preview_file": preview if writer else "", "source_attribution": attribution,
              "notes": ["Prototype geometric yaw cue, not a calibrated 3-D angle or liveness score.",
                        "Best frame means highest quality score among stable, accepted, analysed frames of one track segment.",
                        "Original pixels are preserved; no sharpening, enhancement, or image synthesis."]}
    if views_mode:
        report.update(coverage_complete=snapshot['complete'], missing_views=[t.name for t in pipeline.targets if t.name not in selected],
                      view_configuration=pipeline.configuration(), candidate_verifications=pipeline.verifications,
                      candidate_verification_rejects=pipeline.verification_rejects)
        report['comparison_inputs'] = {name: {'frame_file': item['frame_file'], 'face_file': item['face_file'],
                                            'frame_index': item['frame_index'], 'box_xyxy': item['box_xyxy'],
                                            'pose': item['pose']} for name, item in selected.items()}
        for name, item in snapshot['frames'].items():
            selected[name]['verification'] = item['record']['verification']
            selected[name]['confirmed_at_s'] = item['record']['confirmed_at_s']
        report['notes'] = ['Unordered target-view selection, not an action/liveness challenge.',
                           'Each output passed pose-band, quality, temporal and same-frame landmark checks.',
                           'Quality is a ranking heuristic; no global clearest-frame or calibrated angle claim.',
                           'Upstream timing excludes the cost of producing supplied observations.',
                           'Original pixels preserved; no enhancement or sharpening.']
    report['performance']['peak_rss_mb'] = peak_rss_mb()
    if models and models.runtime_assets:
        report['runtime_assets'] = models.runtime_assets
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if views_mode:
        from .view_io import write_view_review
        write_view_review(output, report)
    else:
        write_review(output, report)
    print(json.dumps({"output": str(output), "performance": report["performance"], "selected": {s: v["frame_index"] for s, v in selected.items()}, "sequence_complete": report["sequence_complete"]}, ensure_ascii=False, indent=2))
    return report


def main(default_mode='challenge'):
    parser = argparse.ArgumentParser(description="Dog head-turn capture and original-frame selection")
    parser.add_argument("video")
    parser.add_argument('--mode', choices=['views', 'challenge'], default=default_mode)
    parser.add_argument('--upstream-jsonl', help='Same-frame FaceObservation JSONL; views mode only, loads zero models')
    parser.add_argument('--target-config', help='JSON list of view names and yaw-cue bands (not angles)')
    parser.add_argument('--full-heatmaps', action='store_true', help='Disable compact confidence output for A/B validation')
    parser.add_argument("--output")
    parser.add_argument("--models", default=str(ROOT / "models/pretrained/dog_face_landmarks_v1"))
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu', 'trt'], default='auto')
    parser.add_argument('--trt-engines', help='Directory built by prepare_dog_trt.py')
    parser.add_argument('--no-cuda-graph', action='store_true', help='TensorRT enqueue baseline without CUDA Graph')
    parser.add_argument("--fps", type=float, default=0, help="0 evaluates every frame; e.g. 15 reduces work")
    parser.add_argument("--hold", type=float, default=0.25)
    parser.add_argument("--side-threshold", type=float, default=0.25)
    parser.add_argument("--dog-threshold", type=float, default=0.35)
    parser.add_argument("--landmark-period", type=float, default=0.35, help="Neural refresh interval, 0 disables optical-flow reuse")
    parser.add_argument("--min-face-pixels", type=int, default=100)
    parser.add_argument("--min-laplacian", type=float, default=12)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.fps < 0 or args.max_seconds < 0 or args.landmark_period < 0 or not 0 < args.dog_threshold < 1:
        parser.error("Invalid runtime parameters")
    if args.upstream_jsonl and (args.mode != 'views' or args.mirror):
        parser.error('--upstream-jsonl requires views mode and already matching, unmirrored pixels')
    run(args)


if __name__ == "__main__":
    main()
