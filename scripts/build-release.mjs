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

function requireReleaseInputs() {
  const required = [
    "dist/desktop/main.cjs",
    "dist/desktop/preload.cjs",
    "dist/server/node-build.mjs",
    "dist/index.html",
    "prisma/migrations",
    "deployment/initial-admin.json",
    "runtime/requirements-runtime.lock",
  ];
  for (const relative of required) {
    if (!existsSync(path.join(root, relative))) throw new Error(`Required release input is missing: ${relative}`);
  }
}

rmSync(releaseRoot, { recursive: true, force: true });
mkdirSync(appRoot, { recursive: true });
run("pnpm", ["prisma", "generate"]);
run("pnpm", ["build:client"]);
run("pnpm", ["build:server"]);
run("pnpm", ["build:desktop"]);
run("pnpm", ["build:preload"]);
requireReleaseInputs();

copy(path.join(root, "dist", "desktop", "main.cjs"), path.join(appRoot, "main.cjs"));
copy(path.join(root, "dist", "desktop", "preload.cjs"), path.join(appRoot, "preload.cjs"));
copy(path.join(root, "dist", "server", "node-build.mjs"), path.join(appRoot, "node-server", "server.mjs"));
copy(path.join(root, "dist"), path.join(appRoot, "renderer"), (source) => {
  const normalized = source.replace(/\\/g, "/");
  return !normalized.endsWith("/server/node-build.mjs")
    && !normalized.endsWith("/server/node-build.mjs.map")
    && !normalized.includes("/desktop/")
    && !normalized.endsWith(".map");
});

const deployRoot = path.join(releaseRoot, "node-deploy");
run("pnpm", [
  "--config.node-linker=hoisted",
  "--filter",
  ".",
  "deploy",
  "--prod",
  "--legacy",
  deployRoot,
]);
copy(path.join(deployRoot, "node_modules"), path.join(appRoot, "node_modules"));
const nativeSqliteBinding = path.join(
  root,
  "node_modules",
  "better-sqlite3",
  "build",
  "Release",
  "better_sqlite3.node",
);
if (!existsSync(nativeSqliteBinding)) {
  throw new Error(`Windows x64 better-sqlite3 binding is missing: ${nativeSqliteBinding}`);
}
copy(
  nativeSqliteBinding,
  path.join(appRoot, "node_modules", "better-sqlite3", "build", "Release", "better_sqlite3.node"),
);
rmSync(deployRoot, { recursive: true, force: true });

copy(path.join(root, "AIS_core_algo"), path.join(releaseRoot, "algorithm-runtime", "AIS_core_algo"), sourceFilter);
copy(path.join(root, "annotation-platform"), path.join(releaseRoot, "annotation-runtime", "annotation-platform"), sourceFilter);
copy(path.join(root, "prisma", "migrations"), path.join(releaseRoot, "migrations"));
copy(path.join(root, "deployment"), path.join(releaseRoot, "deployment"), (source) => !source.endsWith(".consumed"));
run("node", ["scripts/build-python-runtime.mjs"]);

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
console.log(`Assembled release staging directory: ${releaseRoot}`);
