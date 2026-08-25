import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";

const root = process.cwd();
const releaseRoot = path.join(root, ".release");
const runtimeRoot = path.join(releaseRoot, "algorithm-runtime");
const targetPython = path.join(runtimeRoot, "python", "python.exe");
const lockFile = path.join(root, "runtime", "requirements-runtime.lock");
const sourcePythonHome = process.env.AIS_RUNTIME_PYTHON_HOME;

function run(command, args, options = {}) {
  const executable = process.platform === "win32" && command === "uv" ? "uv.exe" : command;
  const result = spawnSync(executable, args, {
    cwd: options.cwd ?? root,
    stdio: "inherit",
    shell: options.shell ?? false,
    env: options.env ?? process.env,
  });
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed.`);
}

if (!sourcePythonHome) {
  throw new Error("AIS_RUNTIME_PYTHON_HOME must point to a clean, pinned CPython 3.11 runtime directory.");
}
if (!existsSync(path.join(sourcePythonHome, "python.exe"))) {
  throw new Error(`AIS_RUNTIME_PYTHON_HOME is not a CPython runtime: ${sourcePythonHome}`);
}
if (!existsSync(lockFile)) {
  throw new Error(`Python runtime lock is missing: ${lockFile}. Run pnpm runtime:lock on the release build runner.`);
}

const targetPythonHome = path.join(runtimeRoot, "python");
rmSync(targetPythonHome, { recursive: true, force: true });
mkdirSync(runtimeRoot, { recursive: true });
cpSync(sourcePythonHome, targetPythonHome, { recursive: true, dereference: true });
rmSync(path.join(targetPythonHome, "Lib", "EXTERNALLY-MANAGED"), { force: true });
run("uv", ["pip", "sync", "--no-managed-python", "--python", targetPython, "--strict", lockFile]);
const stagedAlgorithm = path.join(runtimeRoot, "AIS_core_algo");
const stagedAnnotation = path.join(releaseRoot, "annotation-runtime", "annotation-platform");
run(targetPython, ["-c", "import uvicorn; import prediction.api; import backend.main"], {
  cwd: stagedAlgorithm,
  shell: false,
  env: {
    ...process.env,
    ANNOTATION_TOKEN_SECRET: crypto.randomBytes(32).toString("base64url"),
    PYTHONPATH: [stagedAlgorithm, stagedAnnotation, process.env.PYTHONPATH]
      .filter(Boolean)
      .join(path.delimiter),
  },
});
console.log(`Built standalone Python runtime at ${runtimeRoot}`);
