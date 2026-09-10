import { app, BrowserWindow, dialog, ipcMain, Menu } from "electron";
import { ChildProcess, spawn, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import {
  appendFileSync,
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import net from "node:net";
import os from "node:os";
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

// 单实例运行：重复启动会抢占新端口，而 localStorage（记住密码、登录态）是按 origin 隔离的，
// 端口一变这些数据就"看不见"了。第二次启动改为聚焦已有窗口。
const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
}
app.on("second-instance", () => {
  const existing = BrowserWindow.getAllWindows().find((window) => !window.isDestroyed());
  if (!existing) return;
  if (existing.isMinimized()) existing.restore();
  existing.show();
  existing.focus();
});

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

// 固定 node 端口：localStorage 按 origin（协议+主机+端口）隔离，若端口每次重启随机，
// 记住密码 / 登录态会在重启后全部丢失。优先固定端口，被占用时才回退随机端口。
const NODE_PORT = 7266;

function reserveLoopbackPort(preferred?: number) {
  return new Promise<number>((resolve, reject) => {
    const tryListen = (port: number, onFree: (port: number) => void, onBusy?: () => void) => {
      const server = net.createServer();
      server.once("error", (err) => {
        server.close();
        if ((err as NodeJS.ErrnoException)?.code === "EADDRINUSE" && onBusy) onBusy();
        else reject(err);
      });
      server.listen(port, "127.0.0.1", () => {
        const address = server.address();
        server.close((error) => {
          if (error || !address || typeof address === "string") reject(error || new Error("Unable to reserve a local port."));
          else onFree(address.port);
        });
      });
    };
    if (preferred) tryListen(preferred, resolve, () => tryListen(0, resolve));
    else tryListen(0, resolve);
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
    reserveLoopbackPort(NODE_PORT),
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
  Menu.setApplicationMenu(null); // 去掉窗口顶部系统菜单栏
  const serviceHeader = { "x-ais-service-token": urls.serviceToken };
  // 仅等业务 API（node 服务，约 0.3s）即显示窗口；算法/标注服务在后台继续启动，避免启动白屏等待
  await waitForService(`${urls.nodeBaseUrl}/api/ping`, "Business API");
  void Promise.all([
    waitForService(`http://127.0.0.1:${urls.algorithmPort}/health`, "AIS algorithm API", serviceHeader),
    waitForService(`http://127.0.0.1:${urls.annotationPort}/api/health`, "Annotation API"),
  ]).catch((error) => desktopLog(`Background service health check failed: ${error instanceof Error ? error.message : String(error)}`));
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

const SPLASH_HTML = `<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;height:100%;}
  body{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;font-family:'Microsoft YaHei','PingFang SC',sans-serif;color:#2563eb;background:#f8fafc;}
  .spin{width:42px;height:42px;border:4px solid #dbeafe;border-top-color:#2563eb;border-radius:50%;animation:ais-spin .8s linear infinite;}
  .text{font-size:15px;letter-spacing:1px;}
  @keyframes ais-spin{to{transform:rotate(360deg)}}
</style></head><body><div class="spin"></div><div class="text">AIS 筛查系统正在启动…</div></body></html>`;

app.whenReady().then(async () => {
  if (!gotSingleInstanceLock) return;
  const splash = new BrowserWindow({
    width: 420, height: 260, frame: false, resizable: false, movable: true, center: true,
    alwaysOnTop: true, show: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  splash.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(SPLASH_HTML)}`);
  splash.once("ready-to-show", () => splash.show());
  try {
    const urls = await bootServices();
    await createWindow(urls);
  } catch (error) {
    desktopLog(`Startup failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    await stopServices();
    await dialog.showMessageBox({ type: "error", title: "AIS startup failed", message: error instanceof Error ? error.message : "Unknown error" });
    app.quit();
  } finally {
    if (!splash.isDestroyed()) splash.close();
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
// 「记住密码」改为存到用户数据目录，避免因本地服务端口变化（origin 改变）而丢失
const rememberedLoginPath = () => path.join(app.getPath("userData"), "remembered-login.json");
const assertLocalSender = (event: { senderFrame?: { url?: string } | null }) => {
  if (!event.senderFrame?.url?.startsWith("http://127.0.0.1") && !isDev) throw new Error("Unauthorized IPC sender.");
};
ipcMain.handle("login:remember:get", (event) => {
  assertLocalSender(event);
  try {
    const saved = JSON.parse(readFileSync(rememberedLoginPath(), "utf8"));
    return saved && typeof saved.username === "string" ? { username: saved.username, password: typeof saved.password === "string" ? saved.password : "" } : null;
  } catch {
    return null;
  }
});
ipcMain.handle("login:remember:set", (event, value: { username?: unknown; password?: unknown }) => {
  assertLocalSender(event);
  try {
    const username = String(value?.username ?? "");
    if (!username) return false;
    writeFileSync(rememberedLoginPath(), JSON.stringify({ username, password: String(value?.password ?? "") }), "utf8");
    return true;
  } catch {
    return false;
  }
});
ipcMain.handle("login:remember:clear", (event) => {
  assertLocalSender(event);
  try {
    rmSync(rememberedLoginPath(), { force: true });
    return true;
  } catch {
    return false;
  }
});
ipcMain.handle("diagnostics:export", async (event) => {
  if (!event.senderFrame.url.startsWith("http://127.0.0.1") && !isDev) throw new Error("Unauthorized IPC sender.");
  return exportDiagnostics();
});
ipcMain.handle("report:save-pdf", async (event, html: string, suggestedName: string, locale?: string) => {
  if (!event.senderFrame.url.startsWith("http://127.0.0.1") && !isDev) throw new Error("Unauthorized IPC sender.");
  const parent = BrowserWindow.fromWebContents(event.sender);
  const printWin = new BrowserWindow({ show: false, width: 900, height: 1200 });
  // 用临时 HTML 文件加载（data: URL 有大小限制，报告含多张 base64 图片时会导致加载失败）
  const tmpDir = mkdtempSync(path.join(os.tmpdir(), "ais-pdf-"));
  const tmpHtml = path.join(tmpDir, "report.html");
  try {
    writeFileSync(tmpHtml, String(html || ""), "utf8");
    await printWin.loadFile(tmpHtml);
    // 等待报告图片全部加载完成（最多 ~4s），避免 PDF 中出现空白图
    for (let attempt = 0; attempt < 40; attempt += 1) {
      const done = await printWin.webContents
        .executeJavaScript(`Array.from(document.images).every((img) => img.complete)`)
        .catch(() => true);
      if (done) break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    const pdf = await printWin.webContents.printToPDF({
      printBackground: true,
      pageSize: "A4",
      margins: { marginType: "default" },
    });
    const isTw = String(locale || "").toLowerCase().startsWith("zh-tw");
    const fallbackName = isTw ? "AIS報告" : "AIS报告";
    const defaultPath = /\.pdf$/i.test(String(suggestedName || "")) ? suggestedName : `${String(suggestedName || fallbackName)}.pdf`;
    const options = {
      title: isTw ? "另存為 PDF" : "另存为 PDF",
      defaultPath,
      filters: [{ name: isTw ? "PDF 文件" : "PDF 文档", extensions: ["pdf"] }],
    };
    const { canceled, filePath } = parent
      ? await dialog.showSaveDialog(parent, options)
      : await dialog.showSaveDialog(options);
    if (canceled || !filePath) return { ok: false, canceled: true };
    writeFileSync(filePath, pdf);
    desktopLog(`Report PDF saved: ${filePath}`);
    return { ok: true, filePath };
  } finally {
    printWin.destroy();
    rmSync(tmpDir, { recursive: true, force: true });
  }
});
