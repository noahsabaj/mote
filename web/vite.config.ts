import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { mockBackend } from './mock/index';

// The production app is served by the Python backend at `/` (same origin, port 7860).
// In `npm run dev` there is no Python process, so `mockBackend()` answers /api, /v1 and
// /ws/generate itself with clearly-labelled fake data. It is a dev-only plugin: it has no
// `apply: 'build'` hook and nothing from `mock/` is reachable from `src/`.
export default defineConfig({
  plugins: [svelte(), mockBackend()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022'
  },
  server: {
    // PORT lets a second dev server (a parallel session, a preview harness) pick its own.
    port: Number(process.env.PORT) || 5173,
    strictPort: false
  }
});
