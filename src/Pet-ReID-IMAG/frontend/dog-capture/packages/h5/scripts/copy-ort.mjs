// Copy the onnxruntime-web WASM runtimes into public/ort/ so the H5 works offline / on a LAN
// (no CDN). WebGPU uses the *.jsep.* variant; plain WASM the other one.
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
// the package does not export package.json: walk up from its main entry to the dist folder
let dist = dirname(require.resolve('onnxruntime-web'));
while (!existsSync(join(dist, 'ort-wasm-simd-threaded.wasm')) && dirname(dist) !== dist) dist = dirname(dist);
const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'ort');
mkdirSync(out, { recursive: true });
// WASM runtime (always) and the WebGPU build's asyncify runtime (fetched only on devices with a
// hardware GPU adapter, see sdk/src/worker.ts).
const files = readdirSync(dist).filter((f) => /^ort-wasm-simd-threaded(\.asyncify)?\.(wasm|mjs)$/.test(f));
if (files.length !== 4) throw new Error(`ORT WASM/WebGPU runtime files missing in ${dist}`);
for (const f of files) copyFileSync(join(dist, f), join(out, f));
console.log(`copied ${files.length} onnxruntime-web runtime files to public/ort/`);
