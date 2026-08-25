import path from "node:path";
import { spawnSync } from "node:child_process";

const root = process.cwd();
const input = path.join(root, "runtime", "requirements-runtime.in");
const output = path.join(root, "runtime", "requirements-runtime.lock");
const executable = process.platform === "win32" ? "uv.exe" : "uv";
const result = spawnSync(executable, ["pip", "compile", "--generate-hashes", "--python-version", "3.11", "--output-file", output, input], {
  cwd: root,
  stdio: "inherit",
  shell: false,
});
if (result.status !== 0) process.exit(result.status || 1);
