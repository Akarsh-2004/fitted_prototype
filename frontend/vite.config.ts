import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5188,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8011',
        changeOrigin: true,
        secure: false,
      },
      '/data': {
        target: 'http://127.0.0.1:8011',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
