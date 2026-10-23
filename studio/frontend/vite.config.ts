// Vite config: React plugin; the dev server proxies /api to the FastAPI backend.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/", // the backend mounts dist/ at /
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" }, // run_studio.py default port
  },
});
