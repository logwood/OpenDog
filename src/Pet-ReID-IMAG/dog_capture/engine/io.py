"""Frame sources, the real-time latest-frame worker and run outputs.

Sources yield ``(frame, timestamp_s, index)`` with *capture* timestamps:
container PTS for files (variable frame rate is fine), a monotonic clock for
cameras and streams.  Cameras/streams reconnect with back-off instead of
ending the run on the first failed read.
"""
from __future__ import annotations

from datetime import datetime
import html
import json
import math
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def imread(path):
    data = np.fromfile(str(path), np.uint8)  # unicode-safe on Windows
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)  # applies EXIF orientation
    if image is None:
        raise ValueError(f'Cannot decode image {path}')
    return image


def imwrite(path, image):
    path = Path(path)
    ok, data = cv2.imencode(path.suffix or '.png', image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    temporary = path.with_name(path.name + '.part')
    data.tofile(str(temporary))
    temporary.replace(path)


class VideoSource:
    live = False

    def __init__(self, path, fps=None):
        self.path = str(path)
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise RuntimeError(f'Cannot open video {path}')
        declared = float(self.capture.get(cv2.CAP_PROP_FPS) or 0)
        self.fps = float(fps) if fps else (declared if math.isfinite(declared) and declared > 0 else 30.0)
        self.timestamp_source = 'container_pts'
        self.fallbacks = 0

    def __iter__(self):
        index, previous = 0, -math.inf
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            pts = float(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0
            nominal = index / self.fps
            # Some backends report 0 or a stale PTS; never emit a non-increasing time.
            if not math.isfinite(pts) or pts <= previous or (index > 0 and pts == 0):
                pts = max(nominal, previous + 1e-3)
                self.fallbacks += 1
            previous = pts
            yield frame, pts, index
            index += 1

    def describe(self):
        return {'type': 'video', 'path': self.path, 'fps': self.fps, 'timestamp_source': self.timestamp_source,
                'timestamp_fallbacks': self.fallbacks,
                'frame_count': int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}

    def close(self):
        self.capture.release()


class CameraSource:
    live = True

    def __init__(self, spec, max_reconnects=5, width=0, height=0):
        self.spec = int(spec) if str(spec).isdigit() else str(spec)
        self.max_reconnects = max_reconnects
        self.size = (width, height)
        self.reconnects = 0
        self.capture = None
        self._open()
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30) or 30.0

    def _open(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.spec)
        if self.size[0] and self.size[1]:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if not self.capture.isOpened():
            raise RuntimeError(f'Cannot open camera/stream {self.spec}')

    def __iter__(self):
        start = time.monotonic()
        index, failures = 0, 0
        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                failures += 1
                if failures <= 3:
                    time.sleep(0.05)
                    continue
                if self.reconnects >= self.max_reconnects:
                    break
                time.sleep(min(0.5 * 2 ** self.reconnects, 8))
                self.reconnects += 1
                try:
                    self._open()
                except RuntimeError:
                    continue
                failures = 0
                continue
            failures = 0
            yield frame, time.monotonic() - start, index
            index += 1

    def describe(self):
        return {'type': 'camera', 'source': str(self.spec), 'reconnects': self.reconnects,
                'timestamp_source': 'monotonic_clock_at_read'}

    def close(self):
        self.capture.release()


class ImageSource:
    """Photos (one file or a folder); each image is 1 s apart in synthetic time."""
    live = False

    def __init__(self, path):
        path = Path(path)
        self.files = ([path] if path.is_file() else
                      sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        if not self.files:
            raise FileNotFoundError(f'No images in {path}')
        self.path = str(path)
        self.fps = 1.0
        self.skipped = []

    def __iter__(self):
        for index, file in enumerate(self.files):
            try:
                frame = imread(file)
            except ValueError:
                self.skipped.append(str(file))
                continue
            yield frame, float(index), index

    def describe(self):
        return {'type': 'images', 'path': self.path, 'count': len(self.files), 'unreadable': self.skipped}

    def close(self):
        pass


def open_source(spec, fps=None):
    text = str(spec)
    if text.isdigit() or '://' in text:
        return CameraSource(text)
    path = Path(text)
    if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
        return ImageSource(path)
    if not path.is_file():
        raise FileNotFoundError(f'Input not found: {spec}')
    return VideoSource(path, fps)


class DetectionLog:
    """Upstream detections as JSONL, one line per analysed frame, increasing index:

    ``{"frame_index": 12, "detections": [{"box": [...], "score": .9, "keypoints": [[x, y], ...],
    "schema": "anyface5"}]}``.  The v0.x wrapper ``{"frame_index": 12, "observation":
    {bbox_xyxy, landmarks_xy, confidence, ...}}`` is accepted too.  Frames without a
    line get no detections.
    """

    def __init__(self, path):
        self.stream = open(path, encoding='utf8')
        self.current = None
        self.previous = -1

    def get(self, frame_index):
        while self.current is None or self.current['frame_index'] < frame_index:
            line = self.stream.readline()
            if not line:
                return []
            if not line.strip():
                continue
            value = json.loads(line)
            index = value.get('frame_index')
            if not isinstance(index, int) or index <= self.previous:
                raise ValueError('Detection log frame_index values must be strictly increasing integers')
            self.previous, self.current = index, value
        if self.current['frame_index'] != frame_index:
            return []
        if 'detections' in self.current:
            return list(self.current['detections'] or [])
        observation = self.current.get('observation')
        return [observation] if observation else []

    def close(self):
        self.stream.close()


class LatestFrameRunner:
    """One worker thread, capacity-one queue: slow inference never builds lag.

    ``submit`` replaces a frame still waiting (counted as dropped); results
    carry queue wait and capture-to-result latency.  Worker exceptions are
    re-raised in the caller on the next ``submit``/``poll``/``close``.
    """

    def __init__(self, engine, on_result=None, detections_for=None):
        self.engine = engine
        self.on_result = on_result
        self.detections_for = detections_for
        self.pending = queue.Queue(maxsize=1)
        self.latest = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.submitted = self.dropped = self.processed = 0
        self.thread = threading.Thread(target=self._run, name='dog-capture-worker', daemon=True)
        self.thread.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Capture worker failed') from self.error

    def submit(self, frame, timestamp, index):
        self._check()
        item = (frame, timestamp, index, time.perf_counter())
        try:
            self.pending.get_nowait()
            self.dropped += 1
        except queue.Empty:
            pass
        self.pending.put_nowait(item)
        self.submitted += 1

    def _run(self):
        try:
            while not self.stop.is_set() or not self.pending.empty():
                try:
                    frame, timestamp, index, submitted = self.pending.get(timeout=0.05)
                except queue.Empty:
                    continue
                started = time.perf_counter()
                detections = self.detections_for(index) if self.detections_for else None
                result = self.engine.process(frame, timestamp, index, detections=detections)
                if result is None:
                    continue
                result.timings_ms['queue_wait'] = (started - submitted) * 1000
                result.timings_ms['capture_to_result'] = (time.perf_counter() - submitted) * 1000
                self.processed += 1
                if self.on_result:
                    self.on_result(result, frame)
                with self.lock:
                    self.latest = result
        except BaseException as exc:  # surfaced in the caller thread
            self.error = exc
            self.stop.set()

    def poll(self):
        self._check()
        with self.lock:
            return self.latest

    def close(self, timeout=30):
        self.stop.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError('Capture worker did not stop')
        self._check()


def upright(frame, result):
    """The frame as the engine analysed it (rotation and mirror applied)."""
    if result.rotation_deg:
        frame = np.ascontiguousarray(np.rot90(frame, -(result.rotation_deg // 90)))
    if result.mirrored:
        frame = np.ascontiguousarray(frame[:, ::-1])
    return frame


def fit(image, size):
    """Letterbox ``image`` into ``size`` (w, h) without distortion."""
    w, h = size
    if (image.shape[1], image.shape[0]) == (w, h):
        return image
    ratio = min(w / image.shape[1], h / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))))
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def draw_overlay(frame, result, max_side=960):
    scale = min(1.0, max_side / max(frame.shape[:2]))
    out = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale))) if scale < 1 else frame.copy()
    if result is None:
        return out
    for face in result.faces:
        box = (face.detection.box * scale).astype(int)
        primary = face.track_id == result.primary_track
        ok = face.quality is not None and face.quality.passed
        color = (60, 200, 60) if ok else (0, 160, 255)
        cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), color, 3 if primary else 1)
        for x, y in face.detection.keypoints * scale:
            cv2.circle(out, (int(x), int(y)), 3, (255, 255, 0), -1)
        view = face.view_state.view if face.view_state else None
        label = f'#{face.track_id} {view or "-"}{"*" if face.view_state and face.view_state.stable else ""}'
        if face.quality is not None:
            label += f' q={face.quality.score:.2f}' if face.quality.passed else f' {",".join(face.quality.reasons[:2])}'
        cv2.putText(out, label, (box[0], max(18, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    status = f'{result.guidance["code"]}  captured: {",".join(result.captured_views) or "-"}'
    cv2.putText(out, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def choose_output_dir(output, source_spec, root):
    """Never overwrite: a non-empty directory gets a timestamped sub-directory."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if output:
        path = Path(output)
        if path.exists() and any(path.iterdir()):
            path = path / stamp
    else:
        name = Path(str(source_spec)).stem if not str(source_spec).isdigit() else f'camera{source_spec}'
        path = Path(root) / 'artifacts' / 'runs' / 'dog_capture_v1' / f'{stamp}-{name}'
    suffix = 1
    base = path
    while path.exists() and any(path.iterdir()):
        path = Path(f'{base}-{suffix}')
        suffix += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunWriter:
    def __init__(self, directory, log_frames=True, preview_fps=None, image_format='png'):
        self.directory = Path(directory)
        self.log = (self.directory / 'frames.jsonl').open('w', encoding='utf8') if log_frames else None
        self.preview = None
        self.preview_fps = preview_fps
        self.image_format = image_format
        self.lock = threading.Lock()

    def frame(self, result, frame):
        with self.lock:
            if self.log is not None:
                self.log.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
            if self.preview_fps:
                shown = draw_overlay(upright(frame, result), result)
                if self.preview is None:
                    self.preview_size = (shown.shape[1], shown.shape[0])
                    self.preview = cv2.VideoWriter(str(self.directory / 'preview.mp4'),
                                                   cv2.VideoWriter_fourcc(*'mp4v'), self.preview_fps,
                                                   self.preview_size)
                self.preview.write(fit(shown, self.preview_size))

    def save_bundle(self, bundle, prefix='views'):
        """Write aligned crop, raw face crop and full frame per view; returns paths."""
        out = {}
        if bundle is None:
            return out
        folder = self.directory / prefix
        folder.mkdir(exist_ok=True)
        ext = '.' + self.image_format
        for view in bundle.view_names:
            for rank, shot in enumerate(bundle.slots[view]):
                tag = view if rank == 0 else f'{view}_{rank + 1}'
                files = {'aligned': f'{prefix}/{tag}_aligned{ext}'}
                imwrite(self.directory / files['aligned'], shot.aligned)
                if shot.frame is not None:
                    files['frame'] = f'{prefix}/{tag}_frame{ext}'
                    imwrite(self.directory / files['frame'], shot.frame)
                    x1, y1, x2, y2 = shot.detection.box
                    m = 0.2 * max(x2 - x1, y2 - y1)
                    h, w = shot.frame.shape[:2]
                    crop = shot.frame[int(max(0, y1 - m)):int(min(h, y2 + m)), int(max(0, x1 - m)):int(min(w, x2 + m))]
                    if crop.size:
                        files['face'] = f'{prefix}/{tag}_face{ext}'
                        imwrite(self.directory / files['face'], crop)
                out.setdefault(view, []).append({**files, **shot.summary()})
        return out

    def close(self):
        with self.lock:
            if self.log is not None:
                self.log.close()
                self.log = None
            if self.preview is not None:
                self.preview.release()
                self.preview = None


def percentiles(values):
    if not values:
        return None
    values = np.asarray(values, np.float64)
    return {'p50': round(float(np.percentile(values, 50)), 3), 'p95': round(float(np.percentile(values, 95)), 3),
            'max': round(float(values.max()), 3)}


def write_review(directory, report):
    labels = {'front': '正面', 'left': '朝左', 'right': '朝右'}
    cards = []
    selection = report['selection']
    for view in report['views']:
        items = selection['views'].get(view)
        title = html.escape(labels.get(view, view))
        if not items:
            cards.append(f'<article><h2>{title}</h2><div class="empty">未采集到合格图像</div></article>')
            continue
        item = items[0]
        q = item['quality']
        factors = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v:.2f}</td></tr>' for k, v in q['factors'].items())
        links = ' · '.join(f'<a href="{html.escape(item[k])}">{n}</a>' for k, n in
                           (('aligned', '对齐图'), ('face', '原始裁剪'), ('frame', '原始整帧')) if k in item)
        cards.append(
            f'<article><h2>{title}</h2><img src="{html.escape(item["aligned"])}" alt="{title}">'
            f'<p>第 {item["frame_index"]} 帧 · {item["time_s"]:.2f} s · 质量 {q["score"]:.2f}（{q["level"]}）'
            f'<br>朝向线索 {item["pose"]["yaw"]:+.2f}（约 {item["pose"]["approx_deg"]:+.0f}°，仅供参考）</p>'
            f'<p>{links}</p><details><summary>质量分项</summary><table>{factors}</table></details></article>')
    stats = report['stats']
    frame_link = ' · <a href="frames.jsonl">frames.jsonl</a>' if report.get('frame_log') else ''
    from .guidance import MESSAGES
    guidance = ''.join(f'<li>{html.escape(MESSAGES.get(k, k).replace("：{view}", ""))} '
                       f'<span class="muted">({html.escape(k)})</span>：{v} 帧</li>'
                       for k, v in report['guidance_counts'].items())
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>犬脸优选抓拍</title>
<style>:root{{color-scheme:light dark;--bg:#f3f5f6;--card:#fff;--fg:#17313a;--muted:#5d7178}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11181b;--card:#1b2428;--fg:#e4ecee;--muted:#9fb1b6}}}}
body{{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg);margin:0}}
main{{max-width:1080px;margin:0 auto;padding:24px 16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin:16px 0}}
article,.stats{{background:var(--card);border-radius:12px;padding:16px}}img{{width:100%;aspect-ratio:1;object-fit:contain;background:#0002;border-radius:8px}}
.empty{{aspect-ratio:1;display:grid;place-items:center;color:var(--muted);background:#0001;border-radius:8px}}
.muted{{color:var(--muted);overflow-wrap:anywhere}}a{{color:#0a8a8c}}td{{padding:0 12px 0 0}}</style>
<main><p class="muted">dog_capture v1 · {html.escape(report["input"].get("path", report["input"].get("source", "")))}</p>
<h1>犬脸优选抓拍结果</h1>
<div class="stats">轨迹 #{selection["track_id"]} · 已采集 {len(selection["views"])}/{len(report["views"])} 个角度 ·
缺少：{html.escape("、".join(labels.get(v, v) for v in selection["missing_views"]) or "无")}<br>
处理 {stats["frames_processed"]} 帧（丢弃 {stats["frames_dropped"]}），{stats["processing_fps"]:.1f} 帧/秒 ·
检测器：{html.escape(report["models"]["detector"].get("kind", ""))} · 物种门控：{"开" if report["models"]["body_gate"] else "关"}</div>
<div class="grid">{"".join(cards)}</div>
<div class="stats"><b>引导提示统计</b><ul>{guidance}</ul>
<p class="muted">对齐图采用 PetFace 五点模板（224×224），可直接送入比对模型；原始整帧与裁剪保留输入像素。
朝向“约 X°”由单一口鼻长度模型换算，仅用于显示，判定使用无量纲几何线索。</p>
<p><a href="result.json">result.json</a>{frame_link}</p></div></main></html>'''
    (Path(directory) / 'review.html').write_text(body, encoding='utf8')
