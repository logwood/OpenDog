/**
 * Opt-in: real ONNX models through onnxruntime-web (WASM) vs the Python detectors.
 *   DOG_CAPTURE_REAL=<dir> with dog_face_pose.onnx, yolox_nano.onnx, frame.bgr and ref.json
 *   (ref.json = {w, h, faces: [Python Detection.to_dict()], bodies: [boxes, scores]}).
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { createOrtRunner, fromBGR, YoloPoseDetector, YoloxBodyGate } from '../src';

const dir = process.env.DOG_CAPTURE_REAL;
const ready = !!dir && existsSync(resolve(dir, 'ref.json'));

describe.skipIf(!ready)('real models on onnxruntime-web', () => {
  it('matches the Python face detector and species gate on a real dog photo', async () => {
    const ort = await import('onnxruntime-web');
    ort.env.wasm.numThreads = 1;
    const ref = JSON.parse(readFileSync(resolve(dir!, 'ref.json'), 'utf8'));
    const frame = fromBGR(new Uint8Array(readFileSync(resolve(dir!, 'frame.bgr'))), ref.w, ref.h);
    const faceRunner = await createOrtRunner(ort, new Uint8Array(readFileSync(resolve(dir!, 'dog_face_pose.onnx'))), ['wasm']);
    const faces = await new YoloPoseDetector(faceRunner, { kpt: [6, 3] }).detect(frame);
    expect(faces.length).toBe(ref.faces.length);
    faces[0].box.forEach((v, i) => expect(Math.abs(v - ref.faces[0].box[i])).toBeLessThan(2.0));
    faces[0].keypoints.forEach(([x, y], i) => {
      expect(Math.abs(x - ref.faces[0].keypoints[i][0])).toBeLessThan(2.0);
      expect(Math.abs(y - ref.faces[0].keypoints[i][1])).toBeLessThan(2.0);
    });
    expect(Math.abs(faces[0].score - ref.faces[0].score)).toBeLessThan(0.02);
    const gateRunner = await createOrtRunner(ort, new Uint8Array(readFileSync(resolve(dir!, 'yolox_nano.onnx'))), ['wasm']);
    const { boxes } = await new YoloxBodyGate(gateRunner).detect(frame);
    expect(boxes.length).toBe(ref.bodies[0].length);
    boxes[0].forEach((v, i) => expect(Math.abs(v - ref.bodies[0][0][i])).toBeLessThan(3.0));
  }, 60_000);
});
