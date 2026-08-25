import { copyFile, cp, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { createHash, randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import { localPaths } from "./database";

const exec = promisify(execFile);
const backupRoot = path.join(localPaths.dataRoot, "backups");
const runPowerShell = async (script: string) => { await exec("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], { windowsHide: true, maxBuffer: 1024 * 1024 }); };
const quote = (value: string) => `'${value.replace(/'/g, "''")}'`;
const backupName = (name: string) => path.basename(name).replace(/[^a-zA-Z0-9._-]/g, "_");
type BackupFile = { path: string; sizeBytes: number; sha256: string };
type Manifest = { format: "ais-local-backup"; version: 2; createdAt: string; database: "ais.db"; directories: string[]; files: BackupFile[] };

function normalizeRelative(value: string) {
  const normalized = path.posix.normalize(value.replaceAll("\\", "/"));
  if (!normalized || normalized === "." || normalized.startsWith("../") || path.posix.isAbsolute(normalized)) throw new Error("Backup manifest contains an unsafe file path.");
  return normalized;
}
async function digest(file: string) { return createHash("sha256").update(await readFile(file)).digest("hex"); }
async function collectFiles(root: string, directory = ""): Promise<BackupFile[]> {
  const current = path.join(root, directory); const entries = await readdir(current, { withFileTypes: true }); const files: BackupFile[] = [];
  for (const entry of entries) {
    const relative = directory ? path.posix.join(directory.replaceAll("\\", "/"), entry.name) : entry.name;
    if (entry.isDirectory()) files.push(...await collectFiles(root, relative));
    else if (entry.isFile()) { const absolute = path.join(root, relative); files.push({ path: normalizeRelative(relative), sizeBytes: (await stat(absolute)).size, sha256: await digest(absolute) }); }
  }
  return files.sort((a, b) => a.path.localeCompare(b.path));
}

export async function createBackup() {
  await mkdir(backupRoot, { recursive: true }); const id = `${new Date().toISOString().replace(/[:.]/g, "-")}-${randomUUID().slice(0, 8)}`; const stage = path.join(backupRoot, `stage-${id}`); const archive = path.join(backupRoot, `AIS-local-${id}.zip`); await mkdir(stage, { recursive: true });
  if (!existsSync(localPaths.databasePath)) throw new Error("The local database has not been initialized.");
  await copyFile(localPaths.databasePath, path.join(stage, "ais.db"));
  for (const directory of ["data", "results"]) { const source = path.join(localPaths.dataRoot, directory); if (existsSync(source)) await cp(source, path.join(stage, directory), { recursive: true }); }
  const manifest: Manifest = { format: "ais-local-backup", version: 2, createdAt: new Date().toISOString(), database: "ais.db", directories: ["data", "results"], files: await collectFiles(stage) };
  await writeFile(path.join(stage, "manifest.json"), JSON.stringify(manifest, null, 2));
  await runPowerShell(`Compress-Archive -Path ${quote(path.join(stage, "*"))} -DestinationPath ${quote(archive)} -Force`);
  return { name: path.basename(archive), createdAt: manifest.createdAt };
}
export async function listBackups() { await mkdir(backupRoot, { recursive: true }); return Promise.all((await readdir(backupRoot)).filter((name) => name.endsWith(".zip")).map(async (name) => ({ name, sizeBytes: (await stat(path.join(backupRoot, name))).size }))); }
async function validateBackup(stage: string) {
  const manifestPath = path.join(stage, "manifest.json"); if (!existsSync(manifestPath)) throw new Error("Backup manifest is missing.");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as Manifest;
  if (manifest.format !== "ais-local-backup" || manifest.version !== 2 || manifest.database !== "ais.db" || !Array.isArray(manifest.files) || !manifest.files.length) throw new Error("Backup archive is not compatible with this application.");
  const actual = (await collectFiles(stage)).filter((file) => file.path !== "manifest.json"); const expected = new Map(manifest.files.map((file) => [normalizeRelative(file.path), file]));
  if (expected.size !== manifest.files.length || actual.length !== manifest.files.length) throw new Error("Backup archive file manifest does not match its contents.");
  for (const file of actual) { const reference = expected.get(file.path); if (!reference || reference.sizeBytes !== file.sizeBytes || reference.sha256 !== file.sha256) throw new Error(`Backup integrity verification failed for ${file.path}.`); }
  return manifest;
}
export async function scheduleRestore(name: string) {
  const archive = path.join(backupRoot, backupName(name)); if (!archive.endsWith(".zip") || !existsSync(archive)) throw new Error("Backup archive was not found.");
  const stage = path.join(backupRoot, `restore-${randomUUID()}`); await mkdir(stage, { recursive: true }); await runPowerShell(`Expand-Archive -LiteralPath ${quote(archive)} -DestinationPath ${quote(stage)} -Force`);
  await validateBackup(stage);
  await writeFile(path.join(localPaths.dataRoot, "restore.pending.json"), JSON.stringify({ stage, requestedAt: new Date().toISOString() })); return { restartRequired: true };
}
