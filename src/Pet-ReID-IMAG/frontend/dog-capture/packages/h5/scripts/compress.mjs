// Write .br / .gz next to every compressible build file (like nginx gzip_static / brotli_static).
// mount_capture() serves them when the browser accepts the encoding: the 14.8 MB ORT
// runtime goes over the wire as ~2.8 MB (br) / 3.8 MB (gzip).
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { brotliCompressSync, constants, gzipSync } from 'node:zlib';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
const exts = /\.(wasm|m?js|css|html|json|svg)$/;
let before = 0, after = 0;
function walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) { walk(path); continue; }
    if (!exts.test(name)) continue;
    const data = readFileSync(path);
    if (data.length < 1024) continue;
    const br = brotliCompressSync(data, { params: {
      [constants.BROTLI_PARAM_QUALITY]: 9, [constants.BROTLI_PARAM_SIZE_HINT]: data.length } });
    writeFileSync(`${path}.br`, br);
    writeFileSync(`${path}.gz`, gzipSync(data, { level: 9 }));
    before += data.length; after += br.length;
  }
}
walk(root);
console.log(`compressed ${(before / 2 ** 20).toFixed(1)} MB -> ${(after / 2 ** 20).toFixed(1)} MB (br)`);
