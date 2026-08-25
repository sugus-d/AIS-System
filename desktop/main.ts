import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { ChildProcess, spawn, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import {
  appendFileSync,
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import net from "node:net";
import path from "node:path";

const isDev = !app.isPackaged;
const services: ChildProcess[] = [];
const resourceRoot = () => (isDev ? process.cwd() : process.resourcesPath);
const appRoot = () => (isDev ? process.cwd() : app.getAppPath());
const dataRoot = () => path.join(app.getPath("userData"), "ais-data");
const python = () => isDev
  ? path.join(resourceRoot(), "runtime", "python", "python.exe")
  : path.join(resourceRoot(), "algorithm-runtime", "python", "python.exe");
const MAX_LOG_BYTES = Number(process.env.AIS_LOG_MAX_BYTES || 10 * 1024 * 1024);
const LOG_ARCHIVES = 3;
let isShuttingDown = false;

function desktopLog(message: string) {
  const directory = path.join(dataRoot(), "logs");
  const current = path.join(directory, "desktop.log");
  mkdirSync(directory, { recursive: true });
  if (existsSync(current) && statSync(current).size >= MAX_LOG_BYTES) {
    const oldest = `${current}.${LOG_ARCHIVES}`;
    if (existsSync(oldest)) unlinkSync(oldest);
    for (let index = LOG_ARCHIVES - 1; index >= 1; index -= 1) {
      const source = `${current}.${index}`;
      if (existsSync(source)) renameSync(source, `${current}.${index + 1}`);
    }
    renameSync(current, `${current}.1`);
  }
  appendFileSync(current, `${new Date().toISOString()} ${message}\n`);
}

function applyPendingRestore() {
  const pending = path.join(dataRoot(), "restore.pending.json");
  if (!existsSync(pending)) return;
  const { stage } = JSON.parse(readFileSync(pending, "utf8"));
  if (typeof stage !== "string" || !existsSync(path.join(stage, "manifest.json")) || !existsSync(path.join(stage, "ais.db"))) {
    throw new Error("The pending local restore is invalid.");
  }
  const token = `${Date.now()}-${process.pid}`;
  const restoreDb = path.join(dataRoot(), `ais.db.restore-${token}`);
  const priorDb = path.join(dataRoot(), `ais.db.previous-${token}`);
  const swapped: Array<{ target: string; previous: string }> = [];
  try {
    copyFileSync(path.join(stage, "ais.db"), restoreDb);
    if (existsSync(path.join(dataRoot(), "ais.db"))) renameSync(path.join(dataRoot(), "ais.db"), priorDb);
    renameSync(restoreDb, path.join(dataRoot(), "ais.db"));
    for (const name of ["data", "results"]) {
      const target = path.join(dataRoot(), name);
      const previous = path.join(dataRoot(), `${name}.previous-${token}`);
      const source = path.join(stage, name);
      if (existsSync(target)) renameSync(target, previous);
      swapped.push({ target, previous });
      if (existsSync(source)) cpSync(source, target, { recursive: true, force: false });
      else mkdirSync(target, { recursive: true });
    }
    if (existsSync(priorDb)) unlinkSync(priorDb);
    for (const item of swapped) if (existsSync(item.previous)) rmSync(item.previous, { recursive: true, force: true });
    renameSync(pending, `${pending}.consumed`);
  } catch (error) {
    if (existsSync(restoreDb)) unlinkSync(restoreDb);
    for (const item of swapped.reverse()) {
      if (existsSync(item.target)) rmSync(item.target, { recursive: true, force: true });
      if (existsSync(item.previous)) renameSync(item.previous, item.target);
    }
    if (existsSync(priorDb)) {
      if (existsSync(path.join(dataRoot(), "ais.db"))) unlinkSync(path.join(dataRoot(), "ais.db"));
      renameSync(priorDb, path.join(dataRoot(), "ais.db"));
    }
    throw error;
  }
}

function reserveLoopbackPort() {
  return new Promise<number>((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close((error) => {
        if (error || !address || typeof address === "string") reject(error || new Error("Unable to reserve a local port."));
        else resolve(address.port);
      });
    });
  });
}

function trackService(child: ChildProcess, name: string) {
  services.push(child);
  child.stdout?.on("data", (data) => desktopLog(`[${name}] ${data.toString().trim()}`));
  child.stderr?.on("data", (data) => desktopLog(`[${name}] ${data.toString().trim()}`));
  child.on("error", (error) => desktopLog(`[${name} spawn] ${error.message}`));
  child.on("exit", (code, signal) => desktopLog(`[${name} exit] code=${code} signal=${signal}`));
  return child;
}

function startPython(cwd: string, args: string[], env: Record<string, string>, name: string) {
  return trackService(spawn(python(), args, { cwd, env: { ...process.env, ...env }, windowsHide: true }), name);
}

function startNode(entry: string, env: Record<string, string>) {
  // app.getAppPath() resolves to resources/app.asar when packaged. Windows cannot
  // use an ASAR virtual path as a child-process working directory.
  const cwd = isDev ? process.cwd() : path.dirname(process.execPath);
  return trackService(
    spawn(process.execPath, [entry], {
      cwd,
      env: { ...process.env, ...env, ELECTRON_RUN_AS_NODE: "1" },
      windowsHide: true,
    }),
    "node",
  );
}

function ensureInitialAdminFile(deploymentRoot: string) {
  const target = path.join(dataRoot(), "initial-admin.json");
  const consumed = `${target}.consumed`;
  if (!existsSync(target) && !existsSync(consumed) && !existsSync(path.join(dataRoot(), "ais.db"))) {
    const source = path.join(deploymentRoot, "initial-admin.json");
    if (!existsSync(source)) throw new Error("Initial administrator configuration is missing from this installation.");
    copyFileSync(source, target);
  }
  return target;
}

async function bootServices() {
  const resources = resourceRoot();
  const core = isDev ? path.join(resources, "AIS_core_algo") : path.join(resources, "algorithm-runtime", "AIS_core_algo");
  const annotation = isDev ? path.join(resources, "annotation-platform") : path.join(resources, "annotation-runtime", "annotation-platform");
  const deployment = isDev ? path.join(resources, "deployment") : path.join(resources, "deployment");
  const migrations = isDev ? path.join(resources, "prisma", "migrations") : path.join(resources, "migrations");
  if (!existsSync(python()) || !existsSync(core) || !existsSync(annotation) || !existsSync(deployment) || !existsSync(migrations)) {
    throw new Error("AIS runtime is incomplete. Reinstall the signed setup package.");
  }

  mkdirSync(dataRoot(), { recursive: true });
  desktopLog("Starting local services.");
  applyPendingRestore();
  const [nodePort, algorithmPort, annotationPort] = await Promise.all([
    reserveLoopbackPort(),
    reserveLoopbackPort(),
    reserveLoopbackPort(),
  ]);
  const serviceToken = crypto.randomBytes(32).toString("base64url");
  const nodeBaseUrl = `http://127.0.0.1:${nodePort}`;
  const shared = {
    AIS_DATA_ROOT: path.join(dataRoot(), "data"),
    AIS_RESULTS_ROOT: path.join(dataRoot(), "results"),
    AIS_DATA_HOME: dataRoot(),
    AIS_CORE_ROOT: core,
    AIS_ALGORITHM_URL: `http://127.0.0.1:${algorithmPort}`,
    AIS_SERVICE_TOKEN: serviceToken,
    AIS_INITIAL_ADMIN_FILE: ensureInitialAdminFile(deployment),
    AIS_MIGRATION_FILE: path.join(migrations, "20260820063846_initial_local_schema", "migration.sql"),
    ANNOTATION_TOKEN_SECRET: crypto.randomBytes(32).toString("base64url"),
    ANNOTATION_BASE_URL: `http://127.0.0.1:${annotationPort}`,
    AIS_NODE_BASE_URL: nodeBaseUrl,
    AIS_RENDERER_ROOT: isDev ? path.join(resources, "dist") : path.join(appRoot(), "renderer"),
    PYTHONPATH: [core, annotation].join(path.delimiter),
  };

  startNode(path.join(appRoot(), "node-server", "server.mjs"), { ...shared, PORT: String(nodePort), HOST: "127.0.0.1" });
  startPython(core, ["-m", "uvicorn", "prediction.api:app", "--host", "127.0.0.1", "--port", String(algorithmPort)], shared, "algorithm");
  startPython(annotation, ["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(annotationPort)], shared, "annotation");
  return { nodeBaseUrl, algorithmPort, annotationPort, serviceToken };
}

async function waitForService(url: string, name: string, headers?: HeadersInit) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try {
      if ((await fetch(url, { headers })).ok) return;
    } catch {
      // Service is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`${name} did not become healthy within 60 seconds.`);
}

async function terminateServiceTree(service: ChildProcess) {
  if (!service.pid || service.exitCode !== null) return;
  service.kill("SIGTERM");
  await Promise.race([
    new Promise<void>((resolve) => service.once("exit", () => resolve())),
    new Promise<void>((resolve) => setTimeout(resolve, 5_000)),
  ]);
  if (service.exitCode === null) {
    spawnSync("taskkill", ["/pid", String(service.pid), "/t", "/f"], { stdio: "ignore", windowsHide: true });
  }
}

async function stopServices() {
  await Promise.allSettled(services.splice(0).map(terminateServiceTree));
}

async function exportDiagnostics() {
  const result = await dialog.showOpenDialog({ title: "Choose a diagnostics destination", properties: ["openDirectory", "createDirectory"] });
  if (result.canceled || !result.filePaths[0]) return { cancelled: true };
  const destination = path.join(result.filePaths[0], `AIS-diagnostics-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  mkdirSync(destination, { recursive: true });
  const logs = path.join(dataRoot(), "logs");
  if (existsSync(logs)) cpSync(logs, path.join(destination, "logs"), { recursive: true });
  const manifest = path.join(resourceRoot(), "manifest.json");
  if (existsSync(manifest)) copyFileSync(manifest, path.join(destination, "manifest.json"));
  writeFileSync(path.join(destination, "runtime.json"), JSON.stringify({ exportedAt: new Date().toISOString(), appVersion: app.getVersion(), platform: process.platform }, null, 2));
  return { cancelled: false, path: destination };
}

async function createWindow(urls: Awaited<ReturnType<typeof bootServices>>) {
  const serviceHeader = { "x-ais-service-token": urls.serviceToken };
  await Promise.all([
    waitForService(`${urls.nodeBaseUrl}/api/ping`, "Business API"),
    waitForService(`http://127.0.0.1:${urls.algorithmPort}/health`, "AIS algorithm API", serviceHeader),
    waitForService(`http://127.0.0.1:${urls.annotationPort}/api/health`, "Annotation API"),
  ]);
  const window = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    webPreferences: {
      preload: path.join(appRoot(), "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });
  await window.loadURL(isDev ? "http://127.0.0.1:8080" : urls.nodeBaseUrl);
}

app.whenReady().then(async () => {
  try {
    const urls = await bootServices();
    await createWindow(urls);
  } catch (error) {
    desktopLog(`Startup failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    await stopServices();
    await dialog.showMessageBox({ type: "error", title: "AIS startup failed", message: error instanceof Error ? error.message : "Unknown error" });
    app.quit();
  }
});

app.on("before-quit", (event) => {
  if (isShuttingDown) return;
  event.preventDefault();
  isShuttingDown = true;
  void stopServices().finally(() => app.quit());
});

ipcMain.handle("app:version", (event) => {
  if (!event.senderFrame.url.startsWith("http://127.0.0.1") && !isDev) throw new Error("Unauthorized IPC sender.");
  return app.getVersion();
});
ipcMain.handle("diagnostics:export", async (event) => {
  if (!event.senderFrame.url.startsWith("http://127.0.0.1") && !isDev) throw new Error("Unauthorized IPC sender.");
  return exportDiagnostics();
});
