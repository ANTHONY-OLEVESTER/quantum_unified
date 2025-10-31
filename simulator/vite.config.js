import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: '.',
  base: '/quantum_unified/', // for GitHub Pages under repo name
  server: {
    fs: {
      // allow importing one level up to reuse app.jsx placed at repo root
      allow: [
        resolve(__dirname),
        resolve(__dirname, '..')
      ]
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
})

