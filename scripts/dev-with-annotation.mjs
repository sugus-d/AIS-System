import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const annotationRoot = path.join(root, "annotation-platform");
const annotationFrontend = path.join(annotationRoot, "frontend");
const coreAlgoRoot = process.env.AIS_CORE_ROOT || path.join(root, "AIS_core_algo");
const algorithmPort = Number(process.env.AIS_ALGORITHM_PORT) || 8000;
const annotationPort = Number(process.env.ANNOTATION_PORT) || 5174;
const isWindows = process.platform === "win32";
const pnpmCommand = isWindows ? "pnpm.cmd" : "pnpm";
const npmCommand = isWindows ? "npm.cmd" : "npm";
const uvCommand = isWindows ? "uv.exe" : "uv";
const children = [];
let shuttingDown = false;

if (!existsSync(annotationRoot) || !existsSync(annotationFrontend) || !existsSync(coreAlgoRoot)) {
  console.error("[AIS] 内置标注模块目录不完整，请检查 annotation-platform 和 annotation-core。");
  process.exit(1);
}

function portInUse(port, host = "127.0.0.1") {
  return new Promise((resolve) => {
    const socket = net.createConnection({ port, host });
    socket.once("connect", () => { socket.destroy(); resolve(true); });
    socket.once("error", () => resolve(false));
  });
}

function killChild(child) {
  if (!child || child.killed) return;
  if (isWindows && child.pid) {
    spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore", windowsHide: true });
  } else {
    child.kill("SIGTERM");
  }
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  children.forEach(killChild);
  setTimeout(() => process.exit(code), 300);
}

function start(name, command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || root,
    env: { ...process.env, ...(options.env || {}) },
    stdio: ["inherit", "pipe", "pipe"],
    windowsHide: true,
    shell: isWindows,
  });
  child.stdout.on("data", (chunk) => process.stdout.write(`[${name}] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[${name}] ${chunk}`));
  child.on("exit", (code, signal) => {
    if (code && !shuttingDown) {
      console.error(`[${name}] 服务意外退出（code=${code}，signal=${signal || "-"}）。`);
      shutdown(code);
    }
  });
  children.push(child);
}

async function startIfNeeded(name, port, command, args, options = {}) {
  if (await portInUse(port)) {
    console.log(`[${name}] 端口 ${port} 已被占用，复用现有服务。`);
    return;
  }
  start(name, command, args, options);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
process.on("exit", () => {
  if (!shuttingDown) children.forEach(killChild);
});

await startIfNeeded("app", 8080, pnpmCommand, ["exec", "vite", "--host", "::", "--port", "8080"]);
await startIfNeeded("algorithm-api", algorithmPort, uvCommand, ["run", "--project", coreAlgoRoot, "python", "-m", "uvicorn", "prediction.api:app", "--host", "0.0.0.0", "--port", String(algorithmPort)], { cwd: coreAlgoRoot });
await startIfNeeded("annotation-api", 8765, uvCommand, ["run", "--project", annotationRoot, "python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8765"], {
  cwd: annotationRoot,
  env: {
    PYTHONPATH: [annotationRoot, coreAlgoRoot].join(path.delimiter),
    ANNOTATION_TOKEN_SECRET: process.env.ANNOTATION_TOKEN_SECRET || "ais-annotation-mock-secret",
    AIS_DATA_ROOT: process.env.AIS_DATA_ROOT || path.join(coreAlgoRoot, "data"),
    AIS_RESULTS_ROOT: process.env.AIS_RESULTS_ROOT || path.join(coreAlgoRoot, "results"),
  },
});
await startIfNeeded("annotation-ui", annotationPort, npmCommand, ["run", "dev", "--", "--host", "0.0.0.0", "--port", String(annotationPort)], {
  cwd: annotationFrontend,
});

console.log("\nAIS 系统已启动（包含内置标注模块）");
console.log("  访问地址: http://localhost:8080");
console.log("  标注功能: 请从审核中报告的背部图像中进入");
console.log("  按 Ctrl+C 停止 AIS 系统及其内置服务。\n");
