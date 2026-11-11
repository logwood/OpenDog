import { defaultClientConditions, defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// Served by the Pet ReID service at /capture/ (tools/serve_pet_api.py --enable-capture).
// Cross-origin isolation enables multi-threaded WASM inference (SharedArrayBuffer).
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  base: '/capture/',
  plugins: [vue()],
  // Load the onnxruntime WASM runtime from /capture/ort/ at run time instead of bundling it twice.
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions] },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 2048 },
  server: {
    headers: isolation,
    proxy: { '/v1': { target: process.env.DOG_CAPTURE_API ?? 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  preview: { headers: isolation },
});
