import fs from "fs";
import path from "path";
import { createServer } from "./index";
import * as express from "express";
import { applyLocalMigrations, ensureInitialAdmin } from "./services/database";
import { resumeIncompleteAnalysisTasks } from "./services/analysis-runner";

const API_BASE = "/api";
const host = process.env.HOST || "127.0.0.1";

applyLocalMigrations();
await ensureInitialAdmin();
const resumedTaskCount = await resumeIncompleteAnalysisTasks();
const app = createServer();
const port = Number(process.env.PORT) || 8080;

// In production, serve the built SPA files at the root path
const __dirname = import.meta.dirname;

// In the staged desktop application the server lives in app/node-server and the SPA in app/renderer.
function resolveSpaDistPath() {
  const rendererCandidate = process.env.AIS_RENDERER_ROOT;
  const spaCandidate = path.join(__dirname, "../spa");
  const distCandidate = path.join(__dirname, "..");
  const candidates = [rendererCandidate, spaCandidate, distCandidate].filter((candidate): candidate is string => Boolean(candidate));
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, "index.html"))) return candidate;
  }
  throw new Error(`Cannot find SPA index.html. Tried: ${candidates.join(", ")}`);
}

const spaDistPath = resolveSpaDistPath();
const spaIndexHtml = path.join(spaDistPath, "index.html");

// Prevent accidentally exposing server bundle when spaDistPath falls back to `dist`.
app.use((req, res, next) => {
  if (req.path === "/server" || req.path.startsWith("/server/")) {
    return res.status(404).end();
  }
  return next();
});

// Serve static SPA assets from the root path
app.use(express.static(spaDistPath, { index: false }));

// SPA fallback for React Router (everything except API routes)
app.get(["/", "/*splat"], (req, res, next) => {
  if (req.path.startsWith(API_BASE) || req.path.startsWith("/health")) return next();
  res.sendFile(spaIndexHtml);
});

const server = app.listen(port, host, () => {
  if (resumedTaskCount) console.log(`Resumed ${resumedTaskCount} incomplete analysis task(s).`);
  console.log(`AIS local service listening at http://${host}:${port}`);
});

let shuttingDown = false;
function shutdown(signal: string) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`Received ${signal}, closing local service.`);
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 8_000).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
