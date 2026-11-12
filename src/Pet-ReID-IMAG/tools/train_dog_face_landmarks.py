"""Per-frame dog face landmark model on the canonical crop (the refiner's tracking network).

The capture engine finds a dog face once with the full-frame detector, then follows it on a
square crop where the face covers ``SHARE`` of the side (refine.py).  Running the full
YOLO-pose detector on that crop costs ~100 ms per frame in a phone browser (WASM) and, because
it fails on faces turned past ~45 deg, needs a four-way orientation search.  This network does
the crop job alone: 128 px input, YOLO11n backbone (layers 0-9, initialised from the dog-face
detector), a small FPN with one heatmap per keypoint (soft-argmax), and logits for "a dog
face is here" and "this keypoint is visible", the face box (centre and long side) that the
tracker uses to build the next frame's crop, and the face's roll (cos, sin) at any angle --
read from two extra heatmaps trained at every roll, the nose and the point between the eyes
(neither has a left/right twin, so both are well defined upside down).
Keypoints are trained within +-55 deg of upright (tracking keeps the crop upright from frame
to frame); the roll head covers the full circle, so a face found at an unknown angle needs
two runs (read the roll, re-crop upright) instead of a four-way search.

    python tools/train_dog_face_landmarks.py train --data <dogface6> --init dogface6_best.pt --out runs/lm
    python tools/train_dog_face_landmarks.py eval --data <dogface6> --weights runs/lm/best.pt
    python tools/train_dog_face_landmarks.py export --weights runs/lm/best.pt --output dog_face_landmarks.onnx

Data: the dogface6 YOLO-pose layout (images/{train,val}, labels/...: class cx cy w h then
6 x (x y v), v=0 unlabelled).  Output (ONNX, one tensor [1, 24]): keypoints in crop pixels
(x, y) * 6; face box centre x, y and long side in crop pixels; presence logit; visibility
logits * 6; cos, sin of the face's clockwise roll in the crop (refine.eye_line_deg).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(0)   # no OpenCV thread pool: it deadlocks DataLoader workers forked after its first use

SIZE = 128
SHARE = 0.35                       # face long side / crop side (refine.RefineConfig.share)
K = 6
FLIP = [1, 0, 2, 3, 5, 4]
NAMES = ['left_eye', 'right_eye', 'nose', 'chin', 'left_ear_base', 'right_ear_base']
KEYPOINT_MAX_ROLL = 55.0           # keypoint loss only on crops this close to upright
UPRIGHT_SHARE = 0.5                # samples drawn within +-50 deg of upright (the rest: any roll)


def face_tilt(kp):
    """Clockwise roll (deg) of a face from its labelled eyes, else ear bases, oriented so the nose
    is below the line (refine.eye_line_deg); None if neither pair is labelled."""
    for a, b in ((0, 1), (4, 5)):
        if kp[a, 2] > 0 and kp[b, 2] > 0 and kp[2, 2] > 0:
            axis = kp[b, :2] - kp[a, :2]
            mid = (kp[a, :2] + kp[b, :2]) / 2
            if float((kp[2, :2] - mid) @ np.array([-axis[1], axis[0]])) < 0:
                axis = -axis
            return math.degrees(math.atan2(axis[1], axis[0]))
    return None


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


# ---- data -----------------------------------------------------------------------------
def read_faces(data, split):
    """[(image path, [(box xyxy, keypoints [6, 3] px)])] for one split."""
    items = []
    for image in sorted(glob.glob(f'{data}/images/{split}/*.jpg')):
        label = image.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        try:
            rows = [list(map(float, line.split())) for line in open(label) if line.strip()]
        except FileNotFoundError:
            continue
        if rows:
            items.append((image, rows))
    return items


def to_pixels(rows, w, h):
    faces = []
    for r in rows:
        cx, cy, bw, bh = r[1] * w, r[2] * h, r[3] * w, r[4] * h
        kp = np.array(r[5:5 + 3 * K], np.float64).reshape(K, 3)
        kp[:, 0] *= w
        kp[:, 1] *= h
        faces.append((np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), kp))
    return faces


def crop_matrix(cx, cy, side, angle_deg, size=SIZE):
    """Frame -> size x size crop of a square of ``side`` centred at (cx, cy), turned clockwise."""
    s = size / side
    a = math.radians(angle_deg)
    c, n = math.cos(a), math.sin(a)
    o = (size - 1) / 2
    rot = np.array([[c, -n, o - c * o + n * o], [n, c, o - n * o - c * o], [0, 0, 1]])
    scale = np.array([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy], [0, 0, 1]])
    return rot @ scale


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


class CropSet:
    """Training samples made on the fly: a jittered, rotated canonical crop of one face, or a
    crop of something else (presence 0)."""

    def __init__(self, items, train=True, negatives=0.15, seed=0):
        self.samples = [(i, j) for i, (_, rows) in enumerate(items) for j in range(len(rows))]
        self.items, self.train, self.negatives = items, train, negatives
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rng = self.rng if not self.train else random
        i, j = self.samples[index]
        path, rows = self.items[i]
        image = cv2.imread(path)
        h, w = image.shape[:2]
        faces = to_pixels(rows, w, h)
        box, kp = faces[j]
        long_side = max(box[2] - box[0], box[3] - box[1])
        negative = self.train and rng.random() < self.negatives
        if negative:
            for _ in range(20):   # something of face size that is not a face
                side = long_side / SHARE * math.exp(rng.uniform(-0.4, 0.4))
                cx, cy = rng.uniform(0, w), rng.uniform(0, h)
                probe = [cx - side * SHARE / 2, cy - side * SHARE / 2, cx + side * SHARE / 2, cy + side * SHARE / 2]
                if all(iou(probe, f[0]) < 0.05 for f in faces):
                    break
            else:
                negative = False
        if not negative:
            jitter = 0.12 if self.train else 0.10
            cx = (box[0] + box[2]) / 2 + rng.uniform(-jitter, jitter) * long_side
            cy = (box[1] + box[3]) / 2 + rng.uniform(-jitter, jitter) * long_side
            side = long_side / SHARE * math.exp(rng.uniform(math.log(0.75), math.log(1.33)))
        own = None if negative else face_tilt(kp)
        if negative or rng.random() >= UPRIGHT_SHARE:
            angle = rng.uniform(-180, 180)
        else:   # near upright: what tracking feeds the model
            angle = -(own or 0.0) + rng.uniform(-50, 50)
        matrix = crop_matrix(cx, cy, side, angle)
        crop = cv2.warpAffine(image, matrix[:2], (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        points = np.zeros((K, 2), np.float32)
        mask = np.zeros(K, np.float32)          # coordinates labelled
        vis = np.zeros(K, np.float32)           # visibility target
        vis_mask = np.zeros(K, np.float32)
        face = np.zeros(3, np.float32)          # box centre x, y, long side (crop px); zero for negatives
        roll = np.zeros(3, np.float32)          # cos, sin of the face's roll in the crop, weight
        axis = np.zeros((2, 2), np.float32)     # nose, between the eyes (crop px), at any roll
        axis_mask = np.zeros(2, np.float32)
        if not negative:
            tilt = wrap(angle + (own or 0.0))   # the crop turns the content clockwise by angle
            roll[:] = (math.cos(math.radians(tilt)), math.sin(math.radians(tilt)), 1.0 if own is not None else 0.5)
            centre = matrix[:2, :2] @ [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] + matrix[:2, 2]
            face[:] = (*centre, long_side * SIZE / side)
            labelled = kp[:, 2] > 0
            p = (matrix[:2, :2] @ kp[:, :2].T).T + matrix[:2, 2]
            inside = (p[:, 0] >= 0) & (p[:, 0] < SIZE) & (p[:, 1] >= 0) & (p[:, 1] < SIZE)
            points[:] = p
            mask[:] = labelled & inside & (abs(tilt) <= KEYPOINT_MAX_ROLL)
            axis[0], axis_mask[0] = p[2], float(labelled[2] and inside[2])
            mid = (p[0] + p[1]) / 2
            axis[1] = mid
            axis_mask[1] = float(labelled[0] and labelled[1] and 0 <= mid[0] < SIZE and 0 <= mid[1] < SIZE)
            vis[:], vis_mask[:] = labelled, labelled
        if self.train:
            crop = self.augment(crop, points, mask, vis, vis_mask)
            if random.random() < 0.5:   # mirror: left/right swap
                crop = crop[:, ::-1].copy()
                points[:, 0] = SIZE - 1 - points[:, 0]
                points, mask, vis, vis_mask = points[FLIP], mask[FLIP], vis[FLIP], vis_mask[FLIP]
                if not negative:
                    face[0] = SIZE - 1 - face[0]
                    roll[1] = -roll[1]          # mirrored roll
                axis[:, 0] = SIZE - 1 - axis[:, 0]
        tensor = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1), np.float32) / 255.0
        return (tensor, points / SIZE, mask, np.float32(0 if negative else 1), vis, vis_mask, face / SIZE, roll,
                axis / SIZE, axis_mask)

    @staticmethod
    def augment(crop, points, mask, vis, vis_mask):
        crop = crop.astype(np.float32)
        crop = crop * random.uniform(0.6, 1.4) + random.uniform(-30, 30)          # contrast, brightness
        gray = crop.mean(axis=2, keepdims=True)
        crop = gray + (crop - gray) * random.uniform(0.5, 1.5)                    # saturation
        if random.random() < 0.3:
            crop = cv2.GaussianBlur(crop, (0, 0), random.uniform(0.5, 1.8))
        if random.random() < 0.3:
            crop = crop + np.random.normal(0, random.uniform(2, 8), crop.shape)
        if random.random() < 0.35:   # occluder (a hand, a toy): keypoints under it are not visible
            ow, oh = random.uniform(0.15, 0.45) * SIZE, random.uniform(0.15, 0.45) * SIZE
            ox, oy = random.uniform(0, SIZE - ow), random.uniform(0, SIZE - oh)
            colour = np.array([random.uniform(0, 255) for _ in range(3)], np.float32)
            crop[int(oy):int(oy + oh), int(ox):int(ox + ow)] = colour
            under = ((points[:, 0] >= ox) & (points[:, 0] <= ox + ow) & (points[:, 1] >= oy)
                     & (points[:, 1] <= oy + oh) & (mask > 0))
            vis[under], vis_mask[under] = 0, 1
            mask[under] = 0               # where it is, is unknowable: no coordinate loss
        if random.random() < 0.3:          # JPEG-like quality loss
            ok, enc = cv2.imencode('.jpg', np.clip(crop, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 80)])
            crop = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
        return np.clip(crop, 0, 255).astype(np.uint8)


# ---- model ----------------------------------------------------------------------------
def build_model(init=None):
    import torch
    import torch.nn as nn

    def conv(cin, cout, k=1):
        return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.SiLU())

    class Landmarker(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            with torch.no_grad():
                feats = self.features(torch.zeros(1, 3, SIZE, SIZE))
            c3, c4, c5 = (f.shape[1] for f in feats)
            self.lat5, self.lat4, self.lat3 = conv(c5, 64), conv(c4, 64), conv(c3, 64)
            self.fuse4, self.fuse3 = conv(64, 64, 3), conv(64, 64, 3)
            self.heat = nn.Conv2d(64, K + 2, 1)    # 6 keypoints (near upright) + nose, between eyes (any roll)
            self.cls = nn.Sequential(nn.Linear(c5, 128), nn.SiLU(), nn.Linear(128, 1 + K + 3))
            n = SIZE // 8
            grid = (torch.arange(n, dtype=torch.float32) + 0.5) / n
            self.register_buffer('gx', grid.repeat(n))            # cell centres, row-major
            self.register_buffer('gy', grid.repeat_interleave(n))

        def features(self, x):
            out = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in (4, 6, 9):
                    out.append(x)
            return out

        def forward(self, x):
            f3, f4, f5 = self.features(x)
            p5 = self.lat5(f5)
            p4 = self.fuse4(self.lat4(f4) + nn.functional.interpolate(p5, scale_factor=2, mode='nearest'))
            p3 = self.fuse3(self.lat3(f3) + nn.functional.interpolate(p4, scale_factor=2, mode='nearest'))
            heat = self.heat(p3).flatten(2)                        # [B, K, n*n] logits
            prob = heat.softmax(-1)
            xy = torch.stack([(prob * self.gx).sum(-1), (prob * self.gy).sum(-1)], -1)   # [B, K+2, 2] in 0..1
            out = self.cls(f5.mean((2, 3)))
            logits, raw = out[:, :1 + K], out[:, 1 + K:]
            face = torch.stack([0.5 + raw[:, 0], 0.5 + raw[:, 1], SHARE * torch.exp(raw[:, 2])], 1)   # crop units
            down = xy[:, K] - xy[:, K + 1]                              # between the eyes -> nose
            turn = torch.stack([down[:, 1], -down[:, 0]], 1)            # eye-line tilt = angle(down) - 90 deg
            turn = turn / turn.norm(dim=1, keepdim=True).clamp(min=1e-6)
            return xy, logits, face, heat, turn

    from ultralytics import YOLO
    layers = list(YOLO(init).model.model[:10]) if init else None
    if layers is None:
        raise ValueError('--init (a YOLO11 checkpoint) is required to build the backbone')
    return Landmarker(layers)


HEAT_SIGMA = 0.75                  # target heatmap spread, in cells of the stride-8 map


def heat_targets(points, n):
    """Gaussian distributions over the n x n cells around each keypoint (points in 0..1)."""
    import torch
    centres = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    gx, gy = centres.repeat(n), centres.repeat_interleave(n)
    dx = (gx[None, None] - points[..., :1]) * n
    dy = (gy[None, None] - points[..., 1:]) * n
    target = torch.exp(-(dx ** 2 + dy ** 2) / (2 * HEAT_SIGMA ** 2))
    return target / target.sum(-1, keepdim=True).clamp(min=1e-9)


def losses(xy, logits, face, heat, turn, points, mask, presence, vis, vis_mask, face_target, roll, axis, axis_mask):
    """Heatmap cross-entropy (dense, fast to learn) + soft-argmax coordinates (precise) + heads."""
    import torch.nn.functional as F
    import torch
    n = int(round(heat.shape[-1] ** 0.5))
    every, every_mask = torch.cat([points, axis], 1), torch.cat([mask, axis_mask], 1)
    ce = (-(heat_targets(every, n) * heat.log_softmax(-1)).sum(-1) * every_mask).sum() / every_mask.sum().clamp(min=1)
    xy, axis_xy = xy[:, :K], xy[:, K:]
    axis_l1 = (F.smooth_l1_loss(axis_xy, axis, reduction='none', beta=0.02).sum(-1) * axis_mask).sum() / axis_mask.sum().clamp(min=1)
    coord = (F.smooth_l1_loss(xy, points, reduction='none', beta=0.02).sum(-1) * mask).sum() / mask.sum().clamp(min=1)
    pres = F.binary_cross_entropy_with_logits(logits[:, 0], presence)
    v = (F.binary_cross_entropy_with_logits(logits[:, 1:], vis, reduction='none') * vis_mask).sum() / vis_mask.sum().clamp(min=1)
    b = (F.smooth_l1_loss(face, face_target, reduction='none', beta=0.02).sum(-1) * presence).sum() / presence.sum().clamp(min=1)
    w = roll[:, 2] * presence
    r = ((1 - (turn * roll[:, :2]).sum(1)) * w).sum() / w.sum().clamp(min=1)        # 1 - cos(error)
    # r (roll vs the eye line) is reported, not trained: on three-quarter faces the nose axis is
    # not perpendicular to the eye line (train labels: median 10 deg apart, > 55 deg for 3.7 %),
    # and the refiner's re-crop at the eye line removes that difference anyway.
    return (0.2 * ce + coord * 20 + 5 * axis_l1 + pres + 0.5 * v + 10 * b,
            (ce.item(), coord.item(), pres.item(), v.item(), b.item(), r.item()))


# ---- train / eval / export ------------------------------------------------------------
def loader(dataset, batch, shuffle, workers):
    import torch
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                       persistent_workers=workers > 0, worker_init_fn=_seed_worker)


def _seed_worker(worker_id):
    import torch
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % 2 ** 31
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2 ** 31)


def evaluate(model, items, faces=600):
    """Validation samples drawn like training (half near upright, half any roll): keypoint error
    over the eye distance where the crop is within KEYPOINT_MAX_ROLL of upright (good < 0.25,
    as tools/benchmark_refine.py), and the roll head's error on every face."""
    import torch
    dataset = CropSet(items, train=False)
    rng = np.random.default_rng(1)
    picks = rng.choice(len(dataset), size=min(faces, len(dataset)), replace=False)
    errs, pres, box_err, roll_err = [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(picks), 64):
            batch = [dataset[int(i)] for i in picks[start:start + 64]]
            x = torch.from_numpy(np.stack([b[0] for b in batch]))
            xy, logits, face, _, turn = model(x)
            xy = xy[:, :K].numpy() * SIZE
            for b, p, l, fb, t in zip(batch, xy, logits.numpy(), face.numpy(), turn.numpy()):
                pts, mask = b[1] * SIZE, b[2] > 0
                pres.append(1 / (1 + math.exp(-l[0])))
                box_err.append(abs(fb[2] / max(b[6][2], 1e-6) - 1))
                if b[7][2] == 1.0:
                    roll_err.append(abs(wrap(math.degrees(math.atan2(t[1], t[0]) - math.atan2(b[7][1], b[7][0])))))
                if not (mask[0] and mask[1]):
                    continue
                iod = np.linalg.norm(pts[0] - pts[1])
                if iod < 4:
                    continue
                errs.append(float(np.mean(np.linalg.norm(p[mask] - pts[mask], axis=1)) / iod))
    errs = np.array(errs)
    model.train()
    return {'faces': len(errs), 'good': float(np.mean(errs < 0.25)), 'median_err': float(np.median(errs)),
            'presence_mean': float(np.mean(pres)), 'box_size_err': float(np.median(box_err)),
            'roll_err_median': float(np.median(roll_err)), 'roll_within_30': float(np.mean(np.array(roll_err) < 30))}


def train(args):
    import torch
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_items, val_items = read_faces(args.data, 'train'), read_faces(args.data, 'val')
    model = build_model(args.init)
    params = [{'params': [p for n, p in model.named_parameters() if n.startswith('layers.')], 'lr': args.lr * 0.5},
              {'params': [p for n, p in model.named_parameters() if not n.startswith('layers.')], 'lr': args.lr}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    data = loader(CropSet(train_items), args.batch, True, args.workers)
    steps = args.epochs * len(data)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.5, args.lr], total_steps=steps, pct_start=0.05)
    best, log = -1.0, []
    for epoch in range(args.epochs):
        started, totals, n = time.time(), np.zeros(6), 0
        for x, *targets in data:
            loss, parts = losses(*model(x), *targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            totals += parts
            n += 1
        metrics = evaluate(model, val_items, args.val_faces)
        row = {'epoch': epoch + 1, 'loss': (totals / max(n, 1)).round(4).tolist(), 'seconds': round(time.time() - started),
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}}
        log.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), out / 'last.pt')
        if metrics['good'] > best:
            best = metrics['good']
            torch.save(model.state_dict(), out / 'best.pt')
        (out / 'log.json').write_text(json.dumps({'args': {k: v for k, v in vars(args).items() if k != 'func'}, 'log': log}, indent=1))


def load(args):
    import torch
    model = build_model(args.init)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    return model.eval()


def evaluate_cmd(args):
    print(json.dumps(evaluate(load(args), read_faces(args.data, 'val'), args.val_faces), indent=1))


def export(args):
    import torch
    model = load(args)

    class Packed(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            xy, logits, face, _, turn = self.m(x)
            return torch.cat([(xy[:, :K] * SIZE).flatten(1), face * SIZE, logits, turn], 1)

    target = Path(args.output)
    torch.onnx.export(Packed(model), torch.zeros(1, 3, SIZE, SIZE), str(target), opset_version=17,
                      input_names=['images'], output_names=['landmarks'], dynamo=False)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    meta = {'kind': 'crop_landmarks', 'schema': 'dogface6', 'input_size': SIZE, 'share': SHARE, 'kpt_names': NAMES,
            'outputs': 'x,y * 6, box cx, cy, long side (crop px), presence logit, visibility logit * 6, cos, sin roll',
            'keypoint_max_roll': KEYPOINT_MAX_ROLL, 'sha256': digest,
            'source_weights': str(args.weights),
            'license': 'YOLO11n backbone (Ultralytics, AGPL-3.0) trained on StanfordExtra/Stanford Dogs (research use)'}
    target.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'onnx': str(target), **meta}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--init', required=True, help='YOLO11 checkpoint for the backbone (the dog-face detector)')
    t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--batch', type=int, default=64)
    t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--workers', type=int, default=3)
    t.add_argument('--threads', type=int, default=4)
    t.add_argument('--val-faces', type=int, default=600)
    t.set_defaults(func=train)
    e = sub.add_parser('eval')
    e.add_argument('--data', required=True)
    e.add_argument('--init', required=True)
    e.add_argument('--weights', required=True)
    e.add_argument('--val-faces', type=int, default=600)
    e.set_defaults(func=evaluate_cmd)
    x = sub.add_parser('export')
    x.add_argument('--init', required=True, help='same checkpoint as for training (defines the backbone)')
    x.add_argument('--weights', required=True)
    x.add_argument('--output', required=True)
    x.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
