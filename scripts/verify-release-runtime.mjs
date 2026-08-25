import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { copyFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import net from "node:net";

const root = process.cwd();
const releaseRoot = path.join(root, ".release");
const appRoot = path.join(releaseRoot, "app");
const python = path.join(releaseRoot, "algorithm-runtime", "python", "python.exe");
const core = path.join(releaseRoot, "algorithm-runtime", "AIS_core_algo");
const annotation = path.join(releaseRoot, "annotation-runtime", "annotation-platform");
const nodeEntry = path.join(appRoot, "node-server", "server.mjs");
const electronExecutable = path.join(root, "node_modules", "electron", "dist", "electron.exe");
const required = [
  electronExecutable,
  python,
  core,
  annotation,
  nodeEntry,
  path.join(appRoot, "renderer", "index.html"),
  path.join(releaseRoot, "manifest.json"),
  path.join(releaseRoot, "release-report.json"),
];
for (const item of required) if (!existsSync(item)) throw new Error(`Release validation input is missing: ${item}`);

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8", ...options });
  if (result.status !== 0) throw new Error(`${command} failed:\n${result.stdout}\n${result.stderr}`);
}

function reserveLoopbackPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close((error) => {
        if (error || !address || typeof address === "string") reject(error || new Error("Unable to allocate local port."));
        else resolve(address.port);
      });
    });
  });
}

function wait(url, child, name, headers) {
  return new Promise((resolve, reject) => {
    let count = 0;
    let stderr = "";
    child.stderr?.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    const timer = setInterval(async () => {
      if (child.exitCode !== null) {
        clearInterval(timer);
        reject(new Error(`${name} exited before becoming healthy.\n${stderr}`));
        return;
      }
      try {
        if ((await fetch(url, { headers })).ok) {
          clearInterval(timer);
          resolve();
          return;
        }
      } catch {
        // Service is still starting.
      }
      if (++count >= 120) {
        clearInterval(timer);
        reject(new Error(`${name} did not become healthy.`));
      }
    }, 500);
  });
}

function stop(child) {
  if (!child?.pid || child.exitCode !== null) return;
  child.kill("SIGTERM");
  spawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore", windowsHide: true });
}

const data = mkdtempSync(path.join(os.tmpdir(), "ais-release-smoke-"));
const [nodePort, algorithmPort, annotationPort] = await Promise.all([
  reserveLoopbackPort(),
  reserveLoopbackPort(),
  reserveLoopbackPort(),
]);
const serviceToken = crypto.randomBytes(32).toString("base64url");
const env = {
  ...process.env,
  AIS_DATA_ROOT: path.join(data, "data"),
  AIS_RESULTS_ROOT: path.join(data, "results"),
  AIS_DATA_HOME: data,
  AIS_CORE_ROOT: core,
  AIS_INITIAL_ADMIN_FILE: path.join(data, "initial-admin.json"),
  AIS_MIGRATION_FILE: path.join(releaseRoot, "migrations", "20260820063846_initial_local_schema", "migration.sql"),
  AIS_RENDERER_ROOT: path.join(appRoot, "renderer"),
  AIS_ALGORITHM_URL: `http://127.0.0.1:${algorithmPort}`,
  ANNOTATION_BASE_URL: `http://127.0.0.1:${annotationPort}`,
  AIS_NODE_BASE_URL: `http://127.0.0.1:${nodePort}`,
  AIS_SERVICE_TOKEN: serviceToken,
  ANNOTATION_TOKEN_SECRET: crypto.randomBytes(32).toString("base64url"),
  PYTHONPATH: [core, annotation].join(path.delimiter),
  PORT: String(nodePort),
  HOST: "127.0.0.1",
  ELECTRON_RUN_AS_NODE: "1",
};

run(python, ["-c", "import uvicorn; import prediction.api; import backend.main"], {
  cwd: core,
  env,
});
const initialAdmin = path.join(releaseRoot, "deployment", "initial-admin.json");
if (!existsSync(initialAdmin)) throw new Error("Release deployment initial-admin.json is missing.");
await copyFile(initialAdmin, env.AIS_INITIAL_ADMIN_FILE);

const node = spawn(electronExecutable, [nodeEntry], { cwd: appRoot, env, stdio: "pipe", windowsHide: true });
const algorithm = spawn(python, ["-m", "uvicorn", "prediction.api:app", "--host", "127.0.0.1", "--port", String(algorithmPort)], { cwd: core, env, stdio: "pipe", windowsHide: true });
const annotationApi = spawn(python, ["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(annotationPort)], { cwd: annotation, env, stdio: "pipe", windowsHide: true });

try {
  await Promise.all([
    wait(`${env.AIS_NODE_BASE_URL}/api/ping`, node, "Node service"),
    wait(`${env.AIS_ALGORITHM_URL}/health`, algorithm, "Algorithm service", { "x-ais-service-token": serviceToken }),
    wait(`${env.ANNOTATION_BASE_URL}/api/health`, annotationApi, "Annotation service"),
  ]);
  const ping = await fetch(`${env.AIS_NODE_BASE_URL}/api/ping`);
  if (!ping.ok) throw new Error("Node /api/ping failed.");
  const algorithmUnauthorized = await fetch(`${env.AIS_ALGORITHM_URL}/api/predict`, { method: "POST" });
  if (algorithmUnauthorized.status !== 401) throw new Error("Algorithm API did not reject an unauthenticated local request.");
  if (!existsSync(path.join(data, "ais.db"))) throw new Error("SQLite initialization failed.");
  JSON.parse(readFileSync(path.join(releaseRoot, "manifest.json"), "utf8"));
  JSON.parse(readFileSync(path.join(releaseRoot, "release-report.json"), "utf8"));
  console.log("Release runtime health validation passed.");
} finally {
  stop(node);
  stop(algorithm);
  stop(annotationApi);
  rmSync(data, { recursive: true, force: true });
}
