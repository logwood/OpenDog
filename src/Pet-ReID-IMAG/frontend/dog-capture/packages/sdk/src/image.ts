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

/** Separable area (box-coverage) weights, as cv2.INTER_AREA for downscaling. */
function areaWeights(src: number, dst: number) {
  const scale = src / dst;
  const taps: { index: number[]; weight: number[] }[] = [];
  for (let o = 0; o < dst; o++) {
    const start = o * scale, end = start + scale;
    const index: number[] = [], weight: number[] = [];
    for (let i = Math.floor(start); i < Math.min(src, Math.ceil(end)); i++) {
      const w = Math.min(end, i + 1) - Math.max(start, i);
      if (w > 1e-9) { index.push(i); weight.push(w / scale); }
    }
    taps.push({ index, weight });
  }
  return taps;
}

/** Bilinear weights with half-pixel centres, as cv2.INTER_LINEAR. */
function linearWeights(src: number, dst: number) {
  const scale = src / dst;
  const taps: { index: number[]; weight: number[] }[] = [];
  for (let o = 0; o < dst; o++) {
    let s = (o + 0.5) * scale - 0.5;
    if (s < 0) s = 0;
    let i = Math.floor(s);
    let f = s - i;
    if (i >= src - 1) { i = src - 1; f = 0; }
    taps.push(f > 0 ? { index: [i, i + 1], weight: [1 - f, f] } : { index: [i], weight: [1] });
  }
  return taps;
}

/** cv2.resize: INTER_AREA when shrinking, INTER_LINEAR otherwise (per axis). Returns RGBA. */
export function resize(frame: Frame, outW: number, outH: number): Frame {
  const wx = outW < frame.width ? areaWeights(frame.width, outW) : linearWeights(frame.width, outW);
  const wy = outH < frame.height ? areaWeights(frame.height, outH) : linearWeights(frame.height, outH);
  const tmp = new Float32Array(outW * frame.height * 4);
  for (let y = 0; y < frame.height; y++) {
    const row = y * frame.width * 4;
    for (let x = 0; x < outW; x++) {
      const { index, weight } = wx[x];
      let r = 0, g = 0, b = 0, a = 0;
      for (let t = 0; t < index.length; t++) {
        const p = row + index[t] * 4, w = weight[t];
        r += frame.data[p] * w; g += frame.data[p + 1] * w; b += frame.data[p + 2] * w; a += frame.data[p + 3] * w;
      }
      const q = (y * outW + x) * 4;
      tmp[q] = r; tmp[q + 1] = g; tmp[q + 2] = b; tmp[q + 3] = a;
    }
  }
  const out = makeFrame(outW, outH);
  for (let y = 0; y < outH; y++) {
    const { index, weight } = wy[y];
    for (let x = 0; x < outW; x++) {
      for (let c = 0; c < 4; c++) {
        let v = 0;
        for (let t = 0; t < index.length; t++) v += tmp[(index[t] * outW + x) * 4 + c] * weight[t];
        out.data[(y * outW + x) * 4 + c] = Math.round(v);
      }
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
  const sample = (x: number, y: number, ch: number) =>
    (x < 0 || y < 0 || x >= w || y >= h) ? border : data[(y * w + x) * 4 + ch];
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const sx = ia * x + ib * y + ic, sy = id * x + ie * y + iff;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const fx = sx - x0, fy = sy - y0;
      const q = (y * outW + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const v = (1 - fy) * ((1 - fx) * sample(x0, y0, ch) + fx * sample(x0 + 1, y0, ch))
          + fy * ((1 - fx) * sample(x0, y0 + 1, ch) + fx * sample(x0 + 1, y0 + 1, ch));
        out.data[q + ch] = Math.round(v);
      }
      out.data[q + 3] = 255;
    }
  }
  return out;
}

/** Letterbox into a square canvas: returns the canvas, ratio and padding. */
export function letterbox(frame: Frame, size: number, center = true, pad = 114) {
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
