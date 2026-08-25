import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-vision'],
  },
  resolve: {
    alias: {
      '@mediapipe/tasks-vision': path.resolve(__dirname, 'src/shims/mediapipe-tasks-vision.ts'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8765',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:8765',
        changeOrigin: true,
      },
    },
  },
});
