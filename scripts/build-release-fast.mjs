import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import { spawnSync } from "node:child_process";

// 快速构建：只重建 client/server/desktop 代码并更新 .release/app，
// 复用已有的 node_modules 与 Python 运行时（algorithm-runtime / annotation-runtime）。
// 适用于"只改了前端/后端/Electron 主进程代码"的 bug 修复场景。
// 注意：改过 package.json 依赖、或改过 runtime/requirements-runtime.lock 时，
// 必须改用完整构建 `pnpm build:win`。

const root = process.cwd();
const releaseRoot = path.join(root, ".release");
const appRoot = path.join(releaseRoot, "app");
const packageJson = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"));

function run(command, args) {
  const executable = process.platform === "win32" && command === "pnpm"
    ? path.join(process.env.APPDATA || "", "npm", "node_modules", "pnpm", "bin", "pnpm.cjs")
    : command;
  const commandArgs = executable.endsWith("pnpm.cjs") ? [executable, ...args] : args;
  const result = spawnSync(executable.endsWith("pnpm.cjs") ? process.execPath : executable, commandArgs, {
    cwd: root,
    stdio: "inherit",
    shell: false,
  });
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed.`);
}

function copy(source, target, filter) {
  if (!existsSync(source)) throw new Error(`Required release input is missing: ${source}`);
  cpSync(source, target, { recursive: true, dereference: true, filter });
}

function replaceDir(source, target, filter) {
  rmSync(target, { recursive: true, force: true });
  mkdirSync(path.dirname(target), { recursive: true });
  copy(source, target, filter);
}

function sourceFilter(source) {
  const normalized = source.replace(/\\/g, "/");
  return !/(^|\/)(\.git|\.venv|__pycache__|tests?|docs?|data|results|outputs|node_modules)(\/|$)/.test(normalized)
    && !/\.(pyc|pyo|ipynb|map)$/.test(normalized);
}

function sha256(file) {
  return crypto.createHash("sha256").update(readFileSync(file)).digest("hex");
}

function relativeFiles(directory, base = directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) return relativeFiles(absolute, base);
    return [{ path: path.relative(base, absolute).replace(/\\/g, "/"), bytes: statSync(absolute).size, sha256: sha256(absolute) }];
  });
}

// 前置校验：必须已有可复用的暂存（首次先跑一次完整构建）
if (!existsSync(path.join(appRoot, "node_modules"))) {
  throw new Error("缺少 .release/app/node_modules。请先执行一次完整 `pnpm build:win`（或 `pnpm release:stage`），再使用快速构建。");
}
if (!existsSync(path.join(releaseRoot, "algorithm-runtime", "python", "python.exe"))) {
  throw new Error("缺少 .release/algorithm-runtime/python。请先执行一次完整 `pnpm build:win`（或 `pnpm release:stage`），再使用快速构建。");
}

// 1. 只重建代码部分（跳过 Python 运行时与 node_modules 部署）
run("pnpm", ["prisma", "generate"]);
run("pnpm", ["build:client"]);
run("pnpm", ["build:server"]);
run("pnpm", ["build:desktop"]);
run("pnpm", ["build:preload"]);

// 2. 更新 app 内的代码产物（复用 node_modules）
copy(path.join(root, "dist", "desktop", "main.cjs"), path.join(appRoot, "main.cjs"));
copy(path.join(root, "dist", "desktop", "preload.cjs"), path.join(appRoot, "preload.cjs"));
copy(path.join(root, "dist", "server", "node-build.mjs"), path.join(appRoot, "node-server", "server.mjs"));
replaceDir(path.join(root, "dist"), path.join(appRoot, "renderer"), (source) => {
  const normalized = source.replace(/\\/g, "/");
  return !normalized.endsWith("/server/node-build.mjs")
    && !normalized.endsWith("/server/node-build.mjs.map")
    && !normalized.includes("/desktop/")
    && !normalized.endsWith(".map");
});

// 3. 同步算法/标注源码、迁移、部署资源（复用 Python 运行时）
replaceDir(path.join(root, "AIS_core_algo"), path.join(releaseRoot, "algorithm-runtime", "AIS_core_algo"), sourceFilter);
replaceDir(path.join(root, "annotation-platform"), path.join(releaseRoot, "annotation-runtime", "annotation-platform"), sourceFilter);
replaceDir(path.join(root, "prisma", "migrations"), path.join(releaseRoot, "migrations"));
replaceDir(path.join(root, "deployment"), path.join(releaseRoot, "deployment"), (source) => !source.endsWith(".consumed"));

// 4. 重写版本与清单
writeFileSync(path.join(appRoot, "package.json"), JSON.stringify({
  name: "ais-local-screening-runtime",
  version: packageJson.version,
  private: true,
  main: "main.cjs",
  type: "commonjs",
}, null, 2));

const manifest = {
  desktop: packageJson.version,
  nodeService: packageJson.version,
  algorithm: "2.0.0",
  annotation: "1.0.0",
  models: "v1.0.0",
  platform: "win32-x64",
};
writeFileSync(path.join(releaseRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

const report = {
  generatedAt: new Date().toISOString(),
  manifest,
  files: relativeFiles(releaseRoot).filter((item) => !item.path.endsWith("release-report.json")),
};
writeFileSync(path.join(releaseRoot, "release-report.json"), `${JSON.stringify(report, null, 2)}\n`);

console.log(`Updated release staging (fast): ${releaseRoot}`);
console.log("已跳过 Python 运行时与 node_modules 部署。若改了依赖，请改用 `pnpm build:win` 完整构建。");
