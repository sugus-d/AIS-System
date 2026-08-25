import Database from "better-sqlite3";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const databasePath = path.join(root, ".ais-runtime", "migration-verification.db");
mkdirSync(path.dirname(databasePath), { recursive: true });
const database = new Database(databasePath);
try {
  const hasTable = (name) => Boolean(database.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?").get(name));
  for (const name of ["20260820063846_initial_local_schema", "20260820153000_reviews_annotations"]) {
    const migration = path.join(root, "prisma", "migrations", name, "migration.sql");
    const expected = name.endsWith("annotations") ? "ReportReview" : "User";
    if (!hasTable(expected)) database.exec(readFileSync(migration, "utf8"));
  }
  const required = ["User", "Case", "ScanFile", "AnalysisTask", "Report", "ReportReview", "AnnotationSession", "AuditLog"];
  const missing = required.filter((table) => !hasTable(table));
  if (missing.length) throw new Error(`Missing SQLite tables: ${missing.join(", ")}`);
  console.log("SQLite migrations verified.");
} finally { database.close(); }
