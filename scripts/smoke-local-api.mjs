import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";

const root = process.cwd();
const data = path.join(root, ".ais-runtime", "api-smoke");
rmSync(data, { recursive: true, force: true }); mkdirSync(data, { recursive: true });
const initial = path.join(data, "initial-admin.json"); copyFileSync(path.join(root, "deployment", "initial-admin.json"), initial);
const env = { ...process.env, AIS_DATA_HOME: data, AIS_INITIAL_ADMIN_FILE: initial, AIS_MIGRATION_FILE: path.join(root, "prisma", "migrations", "20260820063846_initial_local_schema", "migration.sql"), PORT: "18180" };
let stderr = "";
const service = spawn(process.execPath, [path.join(root, "dist", "server", "node-build.mjs")], { env, windowsHide: true, stdio: "pipe" });
service.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
try {
  for (let attempt = 0; attempt < 30; attempt += 1) { try { if ((await fetch("http://127.0.0.1:18180/api/ping")).ok) break; } catch { /* startup */ } await new Promise((resolve) => setTimeout(resolve, 300)); if (attempt === 29) throw new Error(`Local API did not become healthy. ${stderr}`); }
  const response = await fetch("http://127.0.0.1:18180/api/auth/login", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ username: "local_admin", password: "AIS-Local-Initial-2026!Secure" }) }); const body = await response.json();
  if (!response.ok || !body.success || !body.data?.token || !existsSync(path.join(data, "ais.db")) || existsSync(initial)) throw new Error("SQLite initialization or administrator login failed.");
  console.log("Local API initialization and administrator login verified.");
} finally { service.kill(); }
