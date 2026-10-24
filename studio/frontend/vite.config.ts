// Vite config: React plugin; the dev server proxies /api to the FastAPI backend.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/", // the backend mounts dist/ at /
  server: {
    port: 5173,
    // run_studio.py default port. The proxy keeps Host "localhost:5173" and the
    // browser sends Origin http://localhost:5173: both pass the backend's
    // loopback-only Host / Origin checks, so no changeOrigin is needed.
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
