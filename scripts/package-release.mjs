// package-release.mjs
// electron-builder 包装器：
//   1) 确保 .release/app 内 better-sqlite3 是按 Electron ABI 编译的（不是则就地重建）；
//   2) 打包前把该 binding 临时替换到根目录 pnpm store（electron-builder 的模块收集会回退到根目录），
//   3) 运行 electron-builder，finally 恢复 store；
//   4) 打包后对 win-unpacked 内的 binding 做 Electron ABI 断言，失败即终止。
// 用法（与 electron-builder CLI 参数一致）：
//   node scripts/package-release.mjs --config electron-builder.release.yml --win nsis
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const root = process.cwd();
const releaseRoot = path.join(root, ".release");
const appModuleDir = path.join(releaseRoot, "app", "node_modules", "better-sqlite3");
const appBinding = path.join(appModuleDir, "build", "Release", "better_sqlite3.node");
const rootModuleLink = path.join(root, "node_modules", "better-sqlite3");
const electron = path.join(root, "node_modules", "electron", "dist", "electron.exe");
const args = process.argv.slice(2);

const RELATIVE_PACKAGED_BINDING = path.join(
  "resources", "app.asar.unpacked", "node_modules", "better-sqlite3", "build", "Release", "better_sqlite3.node",
);

function error(message) {
  console.error(`[package-release] ${message}`);
}

/** 用 Electron 内建 Node 实际加载某个 better-sqlite3 模块目录，验证 ABI 兼容。 */
function loadsUnderElectron(moduleDir) {
  if (!existsSync(electron)) throw new Error(`electron 可执行文件缺失: ${electron}`);
  const loader = path.join(releaseRoot, ".abi-load-check.cjs");
  mkdirSync(releaseRoot, { recursive: true });
  writeFileSync(
    loader,
    'try{const m=require(process.argv[2]);const C=typeof m==="function"?m:m.Database;const db=new C(":memory:");db.close();}catch(e){console.error("ABI LOAD FAIL: "+e.message.split("\\n")[0]);process.exit(1);}\n',
  );
  try {
    const result = spawnSync(electron, [loader, moduleDir], {
      env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" },
      encoding: "utf8",
      windowsHide: true,
      timeout: 30000,
    });
    return result.status === 0;
  } finally {
    rmSync(loader, { force: true });
  }
}

function electronVersion() {
  return JSON.parse(readFileSync(path.join(root, "node_modules", "electron", "package.json"), "utf8")).version;
}

/** 在 .pnpm store 中定位 @electron/rebuild 的 CLI（它是 electron-builder 的传递依赖，顶层不可 resolve）。 */
function findElectronRebuildCli() {
  const pnpmDir = path.join(root, "node_modules", ".pnpm");
  if (!existsSync(pnpmDir)) return null;
  const dir = readdirSync(pnpmDir).find((d) => d.startsWith("@electron+rebuild@"));
  if (!dir) return null;
  const pkgDir = path.join(pnpmDir, dir, "node_modules", "@electron", "rebuild");
  const pkgPath = path.join(pkgDir, "package.json");
  if (!existsSync(pkgPath)) return null;
  const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
  const bin = typeof pkg.bin === "string" ? pkg.bin : pkg.bin?.["electron-rebuild"];
  const cli = bin ? path.join(pkgDir, bin) : "";
  return cli && existsSync(cli) ? cli : null;
}

/** 在 .release/app 内就地重建 better-sqlite3（cwd 限定在 .release/app，绝不动 pnpm store）。 */
function rebuildStagingBinding() {
  const cli = findElectronRebuildCli();
  if (!cli) throw new Error("无法在 .pnpm store 中定位 @electron/rebuild CLI");
  const version = electronVersion();
  console.log(`[package-release] 重建 .release/app 内 better-sqlite3（Electron ${version}）...`);
  const result = spawnSync(process.execPath, [cli, "-f", "-w", "better-sqlite3", "-v", version], {
    cwd: appModuleDir,
    stdio: "inherit",
    windowsHide: true,
  });
  if (result.status !== 0) throw new Error(`@electron/rebuild 失败，无法为 Electron 编译 better-sqlite3（exit ${result.status}）。`);
}

/** 从 CLI 参数推断 electron-builder 输出目录。 */
function resolveOutputDir() {
  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];
    if (token.startsWith("--config.directories.output=")) return token.split("=").slice(1).join("=");
    if (token === "--config.directories.output" && args[i + 1]) return args[i + 1];
  }
  return "release";
}

let exitCode = 0;
const storeBinding = (() => {
  if (!existsSync(rootModuleLink)) return null;
  const resolved = realpathSync(rootModuleLink);
  return path.join(resolved, "build", "Release", "better_sqlite3.node");
})();
let backup = null;

try {
  // 前置：暂存必须存在
  if (!existsSync(appBinding)) {
    throw new Error(".release/app 缺少 better-sqlite3 binding。请先执行 `pnpm release:stage` 或 `pnpm build:win:quick`（首次全量构建用 `pnpm build:win`）。");
  }
  if (!storeBinding || !existsSync(storeBinding)) {
    throw new Error(`无法定位根目录 store binding（${storeBinding || rootModuleLink}）。`);
  }

  // 1) 暂存 binding 若不是 Electron ABI，就地重建
  if (!loadsUnderElectron(appModuleDir)) {
    rebuildStagingBinding();
    if (!loadsUnderElectron(appModuleDir)) throw new Error("重建后 binding 仍无法在 Electron 下加载。");
    console.log("[package-release] 暂存 binding 已重建为 Electron ABI。");
  } else {
    console.log("[package-release] 暂存 binding 已是 Electron ABI，跳过重建。");
  }

  // 2) 临时替换 store binding（electron-builder 会从根目录收集模块）
  backup = `${storeBinding}.package-backup`;
  copyFileSync(storeBinding, backup);
  try {
    copyFileSync(appBinding, storeBinding);
  } catch (cause) {
    throw new Error(`无法写入根目录 store binding：${cause.message}。请先关闭占用 node_modules\\.pnpm 的进程（如 dev 服务 / 正在运行的 App）。`);
  }
  console.log("[package-release] 已临时替换 store binding（打包后自动恢复）。");

  // 3) 运行 electron-builder（参数原样透传）
  const command = `pnpm exec electron-builder ${args.join(" ")}`;
  const result = spawnSync("cmd.exe", ["/d", "/s", "/c", command], {
    cwd: root,
    stdio: "inherit",
    windowsHide: true,
  });
  if (result.status !== 0) {
    exitCode = result.status ?? 1;
    error(`electron-builder 失败（exit ${exitCode}）。`);
  } else {
    // 4) 打包后 ABI 断言：必须经 app.asar 路径加载（与真实运行一致）。
    //    不能直接 require app.asar.unpacked 里的模块目录——better-sqlite3 依赖的
    //    bindings 等包仍在 asar 内，直接加载会因模块解析不到而误报。
    //    存在性检查用 app.asar.unpacked 的真实文件（普通 Node 看不到 asar 内部路径）。
    const packagedModuleDir = path.join(
      root, resolveOutputDir(), "win-unpacked", "resources", "app.asar", "node_modules", "better-sqlite3",
    );
    const packagedBindingReal = path.join(
      root, resolveOutputDir(), "win-unpacked", "resources", "app.asar.unpacked", "node_modules",
      "better-sqlite3", "build", "Release", "better_sqlite3.node",
    );
    if (!existsSync(packagedBindingReal)) {
      exitCode = 1;
      error(`打包产物缺失: ${packagedBindingReal}`);
    } else if (!loadsUnderElectron(packagedModuleDir)) {
      exitCode = 1;
      error("ABI 断言失败：打包产物内的 better-sqlite3 无法在 Electron 下加载，请勿分发该安装包。");
    } else {
      console.log("[package-release] ABI 断言通过：打包产物内的 better-sqlite3 可在 Electron 下加载。");
    }
  }
} catch (cause) {
  exitCode = 1;
  error(cause instanceof Error ? cause.message : String(cause));
} finally {
  if (backup && existsSync(backup)) {
    try {
      copyFileSync(backup, storeBinding);
      rmSync(backup, { force: true });
      console.log("[package-release] store binding 已恢复。");
    } catch (cause) {
      error(`恢复 store binding 失败：${cause.message}。备份保留在 ${backup}，请手动恢复。`);
      exitCode = 1;
    }
  }
}
process.exit(exitCode);
