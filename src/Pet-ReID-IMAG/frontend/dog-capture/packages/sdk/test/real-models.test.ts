/**
 * Opt-in: real ONNX models through onnxruntime-web (WASM) vs the Python detectors.
 *   DOG_CAPTURE_REAL=<dir> with dog_face_pose.onnx, yolox_nano.onnx, frame.bgr and ref.json
 *   (ref.json = {w, h, faces: [Python Detection.to_dict()], bodies: [boxes, scores]}).
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { createOrtRunner, CropLandmarker, defaultConfig, eyeContrast, FaceRefiner, fromBGR, getSchema, YoloPoseDetector, YoloxBodyGate, type Detection } from '../src';

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

const refineReady = !!dir && existsSync(resolve(dir, 'refine_ref.json'));

describe.skipIf(!refineReady)('refinement with the real face model (tools/export_refine_parity.py)', () => {
  it('turns faces upright, verifies them and measures eye evidence like the Python engine', async () => {
    const ort = await import('onnxruntime-web');
    ort.env.wasm.numThreads = 1;
    const ref = JSON.parse(readFileSync(resolve(dir!, 'refine_ref.json'), 'utf8'));
    const runner = await createOrtRunner(ort, new Uint8Array(readFileSync(resolve(dir!, 'dog_face_pose.onnx'))), ['wasm']);
    runner.inputSize = ref.model_input;
    const detector = new YoloPoseDetector(runner, { kpt: [6, 3] });
    let landmarker = null;
    if (ref.landmarks) {   // the crop landmarker tracks; faces[].precise = the detector on its upright crop
      const lmRunner = await createOrtRunner(ort, new Uint8Array(readFileSync(resolve(dir!, ref.landmarks.file))), ['wasm']);
      landmarker = new CropLandmarker(lmRunner, { share: ref.landmarks.share, keypointMaxRoll: ref.landmarks.keypoint_max_roll });
    }
    const refiner = new FaceRefiner(detector, { ...defaultConfig().refine, crop_size: ref.crop_size ?? 0 }, landmarker);
    expect(refiner.size).toBe(ref.landmarks?.input_size || ref.crop_size || ref.model_input);
    for (const item of ref.frames) {
      const frame = fromBGR(new Uint8Array(readFileSync(resolve(dir!, item.file))), item.w, item.h);
      // With a landmarker, start from the reference's full-frame boxes: the 128 px tracker turns a 0.7 px
      // box difference (the detector's own, tested above) into a few degrees; this checks the port.
      const found = ref.landmarks
        ? item.full.map((f: any) => ({ box: f.box, score: f.score, keypoints: f.keypoints, schema: getSchema(f.schema),
          kpScores: f.kp_scores ?? null, source: f.source }))
        : await detector.detect(frame);
      const refined = (await refiner.refine(frame, found, 0, true)).filter((d) => d.extra?.refined);
      const label = `frame ${item.frame_index}`;
      expect(refined.length, label).toBe(item.faces.length);
      for (const [i, e] of item.faces.entries()) {
        const d = refined[i];
        expect(d.extra!.upright_k, label).toBe(e.upright_k);
        expect(Math.abs(Number(d.extra!.upright_angle) - e.upright_angle), label).toBeLessThan(3);
        expect(Math.abs(d.score - e.score), label).toBeLessThan(0.05);
        const iod = Math.hypot(e.keypoints[0][0] - e.keypoints[1][0], e.keypoints[0][1] - e.keypoints[1][1]);
        // the 128 px tracker re-crops once more: cv2's fixed-point warp vs this float port adds a little
        const kpTol = ref.landmarks ? 0.06 : 0.04;
        d.keypoints.forEach(([x, y], k) => expect(Math.hypot(x - e.keypoints[k][0], y - e.keypoints[k][1]) / iod, label).toBeLessThan(kpTol));
        // same implementation on the reference keypoints; on our own keypoints, a point on an edge can shift it more
        // (cv2 warps in 5-bit fixed point, this port in float: the brightness-range term moves a little)
        eyeContrast(frame, { ...d, keypoints: e.keypoints }).forEach((v, k) => expect(Math.abs(v - e.eye_contrast[k]), label).toBeLessThan(0.05));
        eyeContrast(frame, d).forEach((v, k) => expect(Math.abs(v - e.eye_contrast[k]), label).toBeLessThan(0.12));
        let checked = d;
        if (landmarker) {
          // stage by stage from the reference's own inputs (chained, small differences compound):
          // the precise pass on Python's tracked face, then verify on Python's precise face
          const asDet = (x: any): Detection => ({ box: x.box, score: x.score, keypoints: x.keypoints, schema: getSchema(x.schema),
            kpScores: x.kp_scores ?? null, source: x.source, extra: { upright_angle: x.upright_angle, region: x.region, refined: true } });
          const sharp = await refiner.precise(frame, asDet(e));
          expect(sharp === null, label).toBe(e.precise === null);
          if (sharp && e.precise) {
            const piod = Math.hypot(e.precise.keypoints[0][0] - e.precise.keypoints[1][0], e.precise.keypoints[0][1] - e.precise.keypoints[1][1]);
            sharp.keypoints.forEach(([x, y], k) => expect(Math.hypot(x - e.precise.keypoints[k][0], y - e.precise.keypoints[k][1]) / piod, `${label} precise`).toBeLessThan(0.04));
            expect(Math.abs(Number(sharp.extra!.upright_angle) - e.precise.upright_angle), label).toBeLessThan(3);
          }
          checked = e.precise ? asDet(e.precise) : asDet(e);
        }
        const v = await refiner.verify(frame, checked);
        // a re-detection scored within the detector's tolerance (0.02, above) of its 0.35 threshold may
        // land on either side of it
        const flip = (v.score === 0) !== (e.verify.score === 0) && Math.abs(Math.max(v.score, e.verify.score) - 0.35) < 0.03;
        if (flip) continue;
        expect(Math.abs(v.score - e.verify.score), label).toBeLessThan(0.05);
        v.eyes.forEach((x, k) => expect(Math.abs(x - e.verify.eyes[k]), label).toBeLessThan(0.04));
        // Python writes an infinite distance (face not found again) as null
        for (const key of ['mirror', 'agreement'] as const) {
          const want = e.verify[key] ?? Infinity;
          if (Number.isFinite(want)) expect(Math.abs(v[key] - want), `${label} ${key}`).toBeLessThan(0.05);
          else expect(v[key], `${label} ${key}`).toBe(Infinity);
        }
      }
    }
  }, 120_000);
});
