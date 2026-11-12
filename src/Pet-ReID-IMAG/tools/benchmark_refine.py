"""Benchmark: keypoints on the full image vs on a canonical crop, with faces turned by quarter turns.

    python tools/benchmark_refine.py --data <dogface6 dir with images/val, labels/val> --model dog_face_pose.onnx

Each sampled validation face is rotated by r = 0..3 quarter turns (a dog lying down), then
predicted (a) on the full image at 0 deg (the pre-refinement engine), (b) on the full image with
quarter-turn search, (c) on a square crop where the face covers SHARE of the side, with the
search.  The crop centre/size come from the ground-truth box with +-10 % shift and +-15 % scale
(a stand-in for the tracker).  Error = mean keypoint distance / ground-truth eye distance;
"good" < 0.25.  Results in docs/DOG_CAPTURE_USER_VIDEO_ANALYSIS_CN.md.
"""
import glob, json, sys
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dog_capture.engine.detectors import YoloPoseDetector

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--faces', type=int, default=500)
parser.add_argument('--shares', default='0.25,0.35,0.5')
parser.add_argument('--out', default='benchmark_refine.json')
parser.add_argument('--input-size', type=int, default=0, help='crop size for a dynamic-shape model (0 = the model size)')
ARGS = parser.parse_args()
N, SHARES = ARGS.faces, [float(s) for s in ARGS.shares.split(',')]
det = YoloPoseDetector(ARGS.model, score_threshold=0.15, **({'input_size': ARGS.input_size} if ARGS.input_size else {}))
ROT = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
FLIP = [1, 0, 2, 3, 5, 4]


def rot(img, k):
    return img if k % 4 == 0 else cv2.rotate(img, ROT[k % 4])


def rot_pts(p, k, w, h):
    """original -> rot(img, k) coordinates"""
    p = np.asarray(p, float).copy(); x, y = p[..., 0].copy(), p[..., 1].copy()
    k %= 4
    if k == 1: p[..., 0], p[..., 1] = h - 1 - y, x
    elif k == 2: p[..., 0], p[..., 1] = w - 1 - x, h - 1 - y
    elif k == 3: p[..., 0], p[..., 1] = y, w - 1 - x
    return p


def unrot(p, k, w, h):
    """rot(img, k) -> original coordinates (w x h original)"""
    p = np.asarray(p, float).copy(); x, y = p[..., 0].copy(), p[..., 1].copy()
    k %= 4
    if k == 1: p[..., 0], p[..., 1] = y, h - 1 - x
    elif k == 2: p[..., 0], p[..., 1] = w - 1 - x, h - 1 - y
    elif k == 3: p[..., 0], p[..., 1] = w - 1 - y, x
    return p


def square_crop(img, cx, cy, side, out=None):
    """crop a side x side square centred at (cx, cy) (pad 114), resized to out; returns crop and scale/offset"""
    out = out or det.size
    s = out / side
    M = np.float32([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]])
    crop = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return crop, s, (out / 2 - s * cx, out / 2 - s * cy)


def run(img, mirror=False):
    src = cv2.flip(img, 1) if mirror else img
    ds = det.detect(src)
    if not ds: return None
    # the face nearest the crop centre (the tracked one), weighted by score
    c = np.array(src.shape[1::-1]) / 2
    d = max(ds, key=lambda d: d.score / (1 + np.linalg.norm((np.array(d.box[:2]) + d.box[2:]) / 2 - c) / 80))
    kp = np.asarray(d.keypoints, float).copy()
    if mirror: kp[:, 0] = src.shape[1] - 1 - kp[:, 0]; kp = kp[FLIP]
    return float(d.score), kp


rng = np.random.default_rng(1)
files = sorted(glob.glob(f'{ARGS.data}/images/val/*.jpg')); rng.shuffle(files)
rows = []
for f in files:
    if len(rows) >= N: break
    lab = f.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    try: lines = [list(map(float, l.split())) for l in open(lab) if l.strip()]
    except FileNotFoundError: continue
    if not lines: continue
    img = cv2.imread(f); h, w = img.shape[:2]
    L = max(lines, key=lambda l: l[3] * l[4])
    gk = np.array(L[5:]).reshape(6, 3); gk[:, 0] *= w; gk[:, 1] *= h
    if not (gk[0, 2] > 0 and gk[1, 2] > 0 and gk[2, 2] > 0): continue
    iod = float(np.linalg.norm(gk[0, :2] - gk[1, :2]))
    if iod < 8: continue
    vis = gk[:, 2] > 0
    cx, cy, bw, bh = L[1] * w, L[2] * h, L[3] * w, L[4] * h
    r = len(rows) % 4
    img_r = rot(img, r); W, H = img_r.shape[1], img_r.shape[0]
    (ccx, ccy), = rot_pts([[cx, cy]], r, w, h)
    long_side = max(bw, bh)
    # tracker-like prediction: +-10% shift, +-15% scale
    jx, jy = rng.uniform(-.1, .1, 2) * long_side; js = np.exp(rng.uniform(np.log(.85), np.log(1.15)))
    item = {'r': r, 'share': long_side / max(w, h), 'iod': iod}
    # full image: oracle orientation and max-score orientation
    full = []
    for k in range(4):
        res = run(rot(img_r, k))
        if res is None: full.append(None); continue
        kp = unrot(unrot(res[1], k, W, H), r, w, h)
        # only count it if it is this face (nose within the GT box)
        hit = abs(kp[2, 0] - cx) < bw and abs(kp[2, 1] - cy) < bh
        err = float(np.mean(np.linalg.norm(kp[vis] - gk[vis, :2], axis=1)) / iod)
        full.append({'k': k, 'score': res[0], 'err': err if hit else 9.0, 'correct': (r + k) % 4 == 0})
    item['full'] = full
    for share in SHARES:
        side = long_side * js / share
        crop, s, (ox, oy) = square_crop(img_r, ccx + jx, ccy + jy, side)
        cands = []
        for k in range(4):
            res = run(rot(crop, k))
            if res is None: cands.append(None); continue
            kp = unrot(res[1], k, det.size, det.size)
            kp = (kp - [ox, oy]) / s                      # crop -> rotated image
            kp = unrot(kp, r, w, h)                        # rotated image -> original
            err = float(np.mean(np.linalg.norm(kp[vis] - gk[vis, :2], axis=1)) / iod)
            cands.append({'k': k, 'score': res[0], 'err': err, 'correct': (r + k) % 4 == 0})
        item[f'crop{share}'] = cands
    rows.append(item)
json.dump(rows, open(ARGS.out, 'w'))


def summarise(name, pick):
    cells = []
    for lo, hi in ((0, .15), (.15, .35), (.35, .5), (.5, 1.01), (0, 1.01)):
        sel = [r for r in rows if lo <= r['share'] < hi]
        e = np.array([9.0 if pick(r) is None else pick(r)['err'] for r in sel])
        ok = sum(1 for r in sel if pick(r) is not None and pick(r)['correct'])
        cells.append(f"{lo}-{hi}: n{len(sel)} good {np.mean(e < 0.25):.0%} orient {ok / max(len(sel), 1):.0%}")
    print(f'{name:30s}', ' | '.join(cells))


best = lambda cands: max([c for c in cands if c], key=lambda c: c['score'], default=None)
oracle = lambda cands: next((c for c in cands if c and c['correct']), None)
summarise('full image, 0 deg only', lambda r: r['full'][0])
summarise('full image, quarter-turn search', lambda r: best(r['full']))
for share in SHARES:
    summarise(f'crop {share}, quarter-turn search', lambda r, s=share: best(r[f'crop{s}']))
    summarise(f'crop {share}, true orientation', lambda r, s=share: oracle(r[f'crop{s}']))
