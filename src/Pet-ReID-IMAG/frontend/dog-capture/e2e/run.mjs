// End-to-end: Chromium fake camera (real dog footage) -> H5 -> worker (ORT-WASM, real models)
// -> upload -> server re-check (real models) -> token -> commit into the stand-in gallery.
//   E2E_MEDIA=<dir with front.y4m, sides.y4m> DOG_CAPTURE_MODELS=<models> PYTHON=<python> node e2e/run.mjs <out-dir>
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:net';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// CommonJS resolution honours NODE_PATH, so a globally installed Playwright works too.
const { chromium } = createRequire(import.meta.url)('playwright');

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(process.argv[2] ?? join(here, 'results'));
mkdirSync(out, { recursive: true });
const reservation = createServer();
await new Promise((resolve) => reservation.listen(0, '127.0.0.1', resolve));
const port = reservation.address().port;
await new Promise((resolve) => reservation.close(resolve));
const base = `http://127.0.0.1:${port}`;
const media = process.env.E2E_MEDIA;
if (!media) throw new Error('Set E2E_MEDIA to the generated Y4M fixture directory');

const server = spawn(process.env.PYTHON ?? 'python', [join(here, 'server.py'), '--port', String(port)],
  { stdio: ['ignore', 'inherit', 'inherit'], env: process.env, windowsHide: true });
let startupError = null;
server.on('error', (error) => { startupError = error; });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const scenarios = [
  { name: 'identify_front', video: 'front.y4m', body: { purpose: 'identify', policy: 'identify_quick', client_commit: true } },
  { name: 'enroll_sides', video: 'sides.y4m', body: { purpose: 'enroll', policy: 'sides_only', pet_id: 'e2e-dog', client_commit: true } },
];
const summary = [];
const browsers = new Set();
try {
  let ready = false;
  for (let i = 0; i < 200; i++) {
    if (startupError || server.exitCode !== null) throw startupError ?? new Error(`Server exited: ${server.exitCode}`);
    try { if ((await fetch(`${base}/capture/`)).ok) { ready = true; break; } } catch { /* starting */ }
    await sleep(300);
  }
  if (!ready) throw new Error('Capture server failed to start in 60 seconds (build H5 first)');
  for (const s of scenarios) {
    const browser = await chromium.launch({ channel: process.env.E2E_BROWSER || undefined,
      args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      `--use-file-for-fake-video-capture=${join(media, s.video)}`] });
    browsers.add(browser);
    const page = await browser.newPage({ viewport: { width: 412, height: 860 }, deviceScaleFactor: 1 });
    const consoleErrors = [];
    page.on('response', (r) => { if (r.status() >= 400) consoleErrors.push(`HTTP ${r.status()} ${r.url()}`); });
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    // What went over the wire for the heavy assets (runtime and models), and with which encoding.
    const transfers = [];
    page.on('requestfinished', async (request) => {
      const url = request.url();
      if (!/\.(wasm|onnx)$/.test(new URL(url).pathname)) return;
      const response = await request.response();
      const sizes = await request.sizes().catch(() => null);
      transfers.push({ path: new URL(url).pathname, status: response?.status(),
        encoding: response?.headers()['content-encoding'] ?? null, type: response?.headers()['content-type'] ?? null,
        wire_bytes: sizes ? sizes.responseBodySize : null });
    });
    const session = await (await fetch(`${base}/v1/capture/sessions`, { method: 'POST',
      headers: { 'content-type': 'application/json' }, body: JSON.stringify(s.body) })).json();
    const started = Date.now();
    await page.goto(base + session.h5_path);
    const isolated = await page.evaluate(() => self.crossOriginIsolated);
    await page.getByRole('button', { name: '开始采集' }).waitFor();
    await page.waitForFunction(() => !document.querySelector('button.primary')?.disabled, null, { timeout: 120_000 });
    const loadedMs = Date.now() - started;
    await page.screenshot({ path: join(out, `${s.name}-1-intro.png`) });
    await page.getByRole('button', { name: '开始采集' }).click();
    // Poll without waiting on elements so the timeline reflects what the user saw.
    const t0 = Date.now();
    const timeline = [];
    let shot = false;
    let captureMs = null;
    while (Date.now() - t0 < 90_000) {
      const ui = await page.evaluate(() => ({
        prompt: document.querySelector('.prompt')?.textContent?.trim() ?? null,
        fps: document.querySelector('.fps')?.textContent?.trim() ?? null,
        review: !!document.querySelector('.review'),
      }));
      const last = timeline.at(-1);
      if (!last || last.prompt !== ui.prompt) timeline.push({ t_ms: Date.now() - t0, prompt: ui.prompt, fps: ui.fps });
      if (!shot && Date.now() - t0 > 1200 && !ui.review) {
        await page.screenshot({ path: join(out, `${s.name}-2-camera.png`) });
        shot = true;
      }
      if (ui.review) { captureMs = Date.now() - t0; break; }
      await sleep(150);
    }
    if (captureMs === null) throw new Error(`${s.name}: no review screen within 90 s`);
    await page.getByText('确认照片').waitFor({ timeout: 30_000 });
    await page.screenshot({ path: join(out, `${s.name}-3-review.png`), fullPage: true });
    const thumbs = await page.locator('.thumbs img').count();
    if (process.env.E2E_RETAKE === '1' && s.name === 'identify_front') {
      await page.getByRole('button', { name: '重拍正面', exact: true }).click();
      await page.locator('.stage').waitFor({ timeout: 10_000 });
      await page.locator('.review').waitFor({ timeout: 60_000 });
      await page.screenshot({ path: join(out, `${s.name}-3-retaken.png`), fullPage: true });
    }
    await page.getByRole('button', { name: '提交' }).click();
    await page.locator('.result h1').waitFor({ timeout: 60_000 });
    const verdict = await page.locator('.result h1').textContent();
    await page.screenshot({ path: join(out, `${s.name}-4-result.png`), fullPage: true });
    let committed = null;
    const commitButton = page.locator('.result button.primary');
    if (verdict === '采集成功' && await commitButton.count()) {
      await commitButton.click();
      await page.locator('.result .outcome').waitFor({ timeout: 30_000 });
      committed = JSON.parse(await page.locator('pre.json').textContent());
      committed.outcome = await page.locator('.result .outcome').textContent();
      await page.screenshot({ path: join(out, `${s.name}-5-committed.png`), fullPage: true });
    }
    const status = await (await fetch(`${base}/v1/capture/sessions/${session.session_id}`)).json();
    summary.push({ scenario: s.name, cross_origin_isolated: isolated, model_load_ms: loadedMs, capture_ms: captureMs,
      prompt_timeline: timeline, thumbnails: thumbs, verdict, server_status: status.result?.status,
      server_shots: status.result?.shots?.map((x) => ({ id: x.id, accepted: x.accepted, reasons: x.reasons, server: x.server })),
      recognition: status.result?.recognition, state: status.state, committed: committed?.action ?? null,
      gallery: committed?.gallery ?? null, outcome: committed?.outcome ?? null,
      signals: status.signals_recorded, transfers, console_errors: consoleErrors });
    await browser.close();
    browsers.delete(browser);
    if (verdict !== '采集成功' || status.state !== 'committed' || consoleErrors.length) {
      throw new Error(`${s.name}: verification/commit/browser check failed (see summary.json)`);
    }
    if (committed?.gallery?.predicted_pet_id && !committed.outcome.includes(committed.gallery.predicted_pet_id)) {
      throw new Error('UI outcome does not reflect the real recognition response');
    }
  }
  summary.push({ gallery_calls: await (await fetch(`${base}/e2e/calls`)).json() });
} finally {
  for (const browser of browsers) await browser.close();
  if (process.platform === 'win32' && server.pid) spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
  else server.kill();
  writeFileSync(join(out, 'summary.json'), JSON.stringify(summary, null, 2));
}
writeFileSync(join(out, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
