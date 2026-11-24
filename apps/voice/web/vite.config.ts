import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The bot serves the built page at /live/ from the ov_voice package, so the
// build goes there and every asset URL is relative. `npm run dev` proxies the
// bot's routes to a bot on localhost:7860.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  build: { outDir: "../src/ov_voice/web", emptyOutDir: true },
  server: {
    proxy: Object.fromEntries(
      ["/start", "/sessions", "/model"].map((path) => [path, "http://localhost:7860"]),
    ),
  },
});
