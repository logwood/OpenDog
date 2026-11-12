/**
 * Pixel operations with OpenCV semantics, in plain TypeScript (no canvas), so
 * the engine runs identically in browsers, Web Workers, Node and WebViews.
 * Frames are RGBA (ImageData layout).
 */
export interface Frame {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array; // RGBA, row-major
}

export function makeFrame(width: number, height: number): Frame {
  return { width, height, data: new Uint8ClampedArray(width * height * 4) };
}

/** cv2.COLOR_BGR2GRAY fixed-point formula (identical integer rounding). */
export function toGray(frame: Frame): Uint8Array {
  const { width, height, data } = frame;
  const out = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = (data[j + 2] * 1868 + data[j + 1] * 9617 + data[j] * 4899 + 8192) >> 14;
  }
  return out;
}

export function fromBGR(bgr: Uint8Array, width: number, height: number): Frame {
  const frame = makeFrame(width, height);
  for (let i = 0, j = 0; i < width * height; i++, j += 3) {
    frame.data[i * 4] = bgr[j + 2];
    frame.data[i * 4 + 1] = bgr[j + 1];
    frame.data[i * 4 + 2] = bgr[j];
    frame.data[i * 4 + 3] = 255;
  }
  return frame;
}

export function crop(frame: Frame, x1: number, y1: number, x2: number, y2: number): Frame {
  x1 = Math.max(0, Math.floor(x1)); y1 = Math.max(0, Math.floor(y1));
  x2 = Math.min(frame.width, Math.ceil(x2)); y2 = Math.min(frame.height, Math.ceil(y2));
  const out = makeFrame(Math.max(0, x2 - x1), Math.max(0, y2 - y1));
  for (let y = 0; y < out.height; y++) {
    const start = ((y + y1) * frame.width + x1) * 4;
    out.data.set(frame.data.subarray(start, start + out.width * 4), y * out.width * 4);
  }
  return out;
}

/** Rotate clockwise by 0/90/180/270 degrees (np.rot90(frame, -k)). */
export function rotate(frame: Frame, degrees: number): Frame {
  const k = (((degrees / 90) % 4) + 4) % 4;
  if (k === 0) return frame;
  const { width: w, height: h } = frame;
  const out = makeFrame(k % 2 ? h : w, k % 2 ? w : h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let nx: number, ny: number;
      if (k === 1) { nx = h - 1 - y; ny = x; } else if (k === 2) { nx = w - 1 - x; ny = h - 1 - y; } else { nx = y; ny = w - 1 - x; }
      const s = (y * w + x) * 4, d = (ny * out.width + nx) * 4;
      out.data[d] = frame.data[s]; out.data[d + 1] = frame.data[s + 1];
      out.data[d + 2] = frame.data[s + 2]; out.data[d + 3] = frame.data[s + 3];
    }
  }
  return out;
}

export function mirror(frame: Frame): Frame {
  const out = makeFrame(frame.width, frame.height);
  for (let y = 0; y < frame.height; y++) {
    for (let x = 0; x < frame.width; x++) {
      const s = (y * frame.width + x) * 4, d = (y * frame.width + (frame.width - 1 - x)) * 4;
      for (let c = 0; c < 4; c++) out.data[d + c] = frame.data[s + c];
    }
  }
  return out;
}

/** Separable taps in flat typed arrays: output o reads index[first[o] .. first[o] + count[o]). */
interface Taps { first: Int32Array; count: Int32Array; index: Int32Array; weight: Float64Array }

function packTaps(lists: [number[], number[]][]): Taps {
  const total = lists.reduce((n, [i]) => n + i.length, 0);
  const taps: Taps = { first: new Int32Array(lists.length), count: new Int32Array(lists.length),
    index: new Int32Array(total), weight: new Float64Array(total) };
  let k = 0;
  lists.forEach(([index, weight], o) => {
    taps.first[o] = k; taps.count[o] = index.length;
    for (let t = 0; t < index.length; t++, k++) { taps.index[k] = index[t]; taps.weight[k] = weight[t]; }
  });
  return taps;
}

/** Separable area (box-coverage) weights, as cv2.INTER_AREA for downscaling. */
function areaWeights(src: number, dst: number): Taps {
  const scale = src / dst;
  const lists: [number[], number[]][] = [];
  for (let o = 0; o < dst; o++) {
    const start = o * scale, end = start + scale;
    const index: number[] = [], weight: number[] = [];
    for (let i = Math.floor(start); i < Math.min(src, Math.ceil(end)); i++) {
      const w = Math.min(end, i + 1) - Math.max(start, i);
      if (w > 1e-9) { index.push(i); weight.push(w / scale); }
    }
    lists.push([index, weight]);
  }
  return packTaps(lists);
}

/** Bilinear weights with half-pixel centres, as cv2.INTER_LINEAR. */
function linearWeights(src: number, dst: number): Taps {
  const scale = src / dst;
  const lists: [number[], number[]][] = [];
  for (let o = 0; o < dst; o++) {
    let s = (o + 0.5) * scale - 0.5;
    if (s < 0) s = 0;
    let i = Math.floor(s);
    let f = s - i;
    if (i >= src - 1) { i = src - 1; f = 0; }
    lists.push(f > 0 ? [[i, i + 1], [1 - f, f]] : [[i], [1]]);
  }
  return packTaps(lists);
}

// Camera frames keep their size, so the separable taps and the scratch buffer are reused.
const tapCache = new Map<string, Taps>();
function taps(src: number, dst: number): Taps {
  const key = `${src}:${dst}`;
  let t = tapCache.get(key);
  if (!t) {
    t = dst < src ? areaWeights(src, dst) : linearWeights(src, dst);
    if (tapCache.size > 32) tapCache.clear();
    tapCache.set(key, t);
  }
  return t;
}
let scratch = new Float32Array(0);

/** cv2.resize: INTER_AREA when shrinking, INTER_LINEAR otherwise (per axis). Returns RGBA. */
export function resize(frame: Frame, outW: number, outH: number): Frame {
  const wx = taps(frame.width, outW), wy = taps(frame.height, outH);
  if (scratch.length < outW * frame.height * 4) scratch = new Float32Array(outW * frame.height * 4);
  const tmp = scratch;
  const data = frame.data;
  for (let y = 0; y < frame.height; y++) {
    const row = y * frame.width * 4;
    for (let x = 0; x < outW; x++) {
      const first = wx.first[x], last = first + wx.count[x];
      let r = 0, g = 0, b = 0;
      for (let t = first; t < last; t++) {
        const p = row + wx.index[t] * 4, w = wx.weight[t];
        r += data[p] * w; g += data[p + 1] * w; b += data[p + 2] * w;
      }
      const q = (y * outW + x) * 4;
      tmp[q] = r; tmp[q + 1] = g; tmp[q + 2] = b;   // alpha is not carried: frames are opaque
    }
  }
  const out = makeFrame(outW, outH);
  const od = out.data;
  for (let y = 0; y < outH; y++) {
    const first = wy.first[y], last = first + wy.count[y];
    for (let x = 0; x < outW; x++) {
      const q = (y * outW + x) * 4;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let t = first; t < last; t++) v += tmp[(wy.index[t] * outW + x) * 4 + c] * wy.weight[t];
        od[q + c] = Math.round(v);
      }
      od[q + 3] = 255;
    }
  }
  return out;
}

/** cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT=border) with a forward 2x3 matrix. */
export function warpAffine(frame: Frame, m: number[][], outW: number, outH: number, border = 0): Frame {
  const [[a, b, c], [d, e, f]] = m;
  const det = a * e - b * d;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  const ic = -(ia * c + ib * f), iff = -(id * c + ie * f);
  const out = makeFrame(outW, outH);
  const { width: w, height: h, data } = frame;
  const od = out.data;
  for (let y = 0; y < outH; y++) {
    const by = ib * y, ey = ie * y;   // same evaluation order as ia*x + ib*y + ic (bit-identical)
    for (let x = 0; x < outW; x++) {
      const sx = ia * x + by + ic, sy = id * x + ey + iff;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const fx = sx - x0, fy = sy - y0;
      const q = (y * outW + x) * 4;
      if (x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h) {   // all four taps inside: no border tests
        const p00 = (y0 * w + x0) * 4, p10 = p00 + w * 4;
        const gx = 1 - fx, gy = 1 - fy;
        // values are >= 0, so floor(v + 0.5) equals Math.round(v)
        od[q] = ((gy * (gx * data[p00] + fx * data[p00 + 4]) + fy * (gx * data[p10] + fx * data[p10 + 4])) + 0.5) | 0;
        od[q + 1] = ((gy * (gx * data[p00 + 1] + fx * data[p00 + 5]) + fy * (gx * data[p10 + 1] + fx * data[p10 + 5])) + 0.5) | 0;
        od[q + 2] = ((gy * (gx * data[p00 + 2] + fx * data[p00 + 6]) + fy * (gx * data[p10 + 2] + fx * data[p10 + 6])) + 0.5) | 0;
      } else if (x0 + 1 < 0 || y0 + 1 < 0 || x0 >= w || y0 >= h) {   // entirely outside
        od[q] = od[q + 1] = od[q + 2] = Math.round(border);
      } else {
        const inside = (xx: number, yy: number) => xx >= 0 && yy >= 0 && xx < w && yy < h;
        for (let ch = 0; ch < 3; ch++) {
          const s00 = inside(x0, y0) ? data[(y0 * w + x0) * 4 + ch] : border;
          const s01 = inside(x0 + 1, y0) ? data[(y0 * w + x0 + 1) * 4 + ch] : border;
          const s10 = inside(x0, y0 + 1) ? data[((y0 + 1) * w + x0) * 4 + ch] : border;
          const s11 = inside(x0 + 1, y0 + 1) ? data[((y0 + 1) * w + x0 + 1) * 4 + ch] : border;
          od[q + ch] = Math.round((1 - fy) * ((1 - fx) * s00 + fx * s01) + fy * ((1 - fx) * s10 + fx * s11));
        }
      }
      od[q + 3] = 255;
    }
  }
  return out;
}

/** Letterbox into a square canvas: returns the canvas, ratio and padding. */
export function letterbox(frame: Frame, size: number, center = true, pad = 114) {
  // already the network size (a refinement crop): resizing at scale 1 is the identity
  if (frame.width === size && frame.height === size) return { canvas: frame, ratio: 1, pad: [0, 0] as [number, number] };
  const ratio = Math.min(size / frame.height, size / frame.width);
  const nw = Math.max(1, Math.round(frame.width * ratio)), nh = Math.max(1, Math.round(frame.height * ratio));
  const resized = resize(frame, nw, nh);
  const canvas = makeFrame(size, size);
  canvas.data.fill(pad);
  const px = center ? Math.floor((size - nw) / 2) : 0, py = center ? Math.floor((size - nh) / 2) : 0;
  for (let y = 0; y < nh; y++) {
    canvas.data.set(resized.data.subarray(y * nw * 4, (y + 1) * nw * 4), ((y + py) * size + px) * 4);
  }
  return { canvas, ratio, pad: [px, py] as [number, number] };
}
