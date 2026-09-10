import { existsSync, mkdirSync, readFileSync, renameSync } from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import bcrypt from "bcryptjs";
import Sqlite from "better-sqlite3";
import { PrismaBetterSqlite3 } from "@prisma/adapter-better-sqlite3";
import { PrismaClient } from "../generated/prisma/client";

const dataRoot = process.env.AIS_DATA_HOME || path.join(process.cwd(), "data");
const databasePath = path.join(dataRoot, "ais.db");
mkdirSync(dataRoot, { recursive: true });
export function applyLocalMigrations() {
  const migrationPath = process.env.AIS_MIGRATION_FILE;
  const connection = new Sqlite(databasePath);
  try {
    const exists = (name: string) => connection.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?").get(name);
    const hasColumn = (table: string, column: string) => { try { connection.prepare(`SELECT "${column}" FROM "${table}" LIMIT 1`).get(); return true; } catch { return false; } };
    if (!exists("User")) {
      if (!migrationPath || !existsSync(migrationPath)) throw new Error("Local database migration file is missing.");
      connection.exec(readFileSync(migrationPath, "utf8"));
    }
    if (!exists("ReportReview")) {
      if (!migrationPath) throw new Error("Local database migration file is missing.");
      const reviewMigration = path.join(path.dirname(path.dirname(migrationPath)), "20260820153000_reviews_annotations", "migration.sql");
      if (!existsSync(reviewMigration)) throw new Error("Review and annotation migration file is missing.");
      connection.exec(readFileSync(reviewMigration, "utf8"));
    }
    if (!exists("Feedback")) {
      if (!migrationPath) throw new Error("Local database migration file is missing.");
      const feedbackMigration = path.join(path.dirname(path.dirname(migrationPath)), "20260820180000_feedback", "migration.sql");
      if (!existsSync(feedbackMigration)) throw new Error("Feedback migration file is missing.");
      connection.exec(readFileSync(feedbackMigration, "utf8"));
    }
    if (!exists("Institution")) {
      if (!migrationPath) throw new Error("Local database migration file is missing.");
      const institutionMigration = path.join(path.dirname(path.dirname(migrationPath)), "20260820200000_institutions", "migration.sql");
      if (!existsSync(institutionMigration)) throw new Error("Institution migration file is missing.");
      connection.exec(readFileSync(institutionMigration, "utf8"));
    }
    if (!hasColumn("Case", "idNumber")) {
      if (!migrationPath) throw new Error("Local database migration file is missing.");
      const contactMigration = path.join(path.dirname(path.dirname(migrationPath)), "20260826120000_case_contact", "migration.sql");
      if (!existsSync(contactMigration)) throw new Error("Case contact migration file is missing.");
      connection.exec(readFileSync(contactMigration, "utf8"));
    }
    if (!hasColumn("ScanFile", "department")) {
      if (!migrationPath) throw new Error("Local database migration file is missing.");
      const identityMigration = path.join(path.dirname(path.dirname(migrationPath)), "20260828120000_scanfile_identity", "migration.sql");
      if (!existsSync(identityMigration)) throw new Error("ScanFile identity migration file is missing.");
      connection.exec(readFileSync(identityMigration, "utf8"));
    }
    if (exists("ReportXray")) {
      // 报告↔X光关联已废弃：X 光是受检者级附加数据，不绑定报告，回滚该表
      if (!migrationPath) throw new Error("Local database migration file is missing.");
      const dropXrayMigration = path.join(path.dirname(path.dirname(migrationPath)), "20260827130000_drop_report_xray", "migration.sql");
      if (!existsSync(dropXrayMigration)) throw new Error("Report xray rollback migration file is missing.");
      connection.exec(readFileSync(dropXrayMigration, "utf8"));
    }
    if (exists("Report")) {
      // 一次性数据清理：同一文件只保留最新一份报告（重新分析已改为原地更新），并清理历史重复版本的级联残留
      connection.exec(`DELETE FROM "Report" WHERE EXISTS (SELECT 1 FROM "Report" AS newer WHERE newer."caseId" = "Report"."caseId" AND newer."fileId" = "Report"."fileId" AND newer.version > "Report".version); DELETE FROM "ReportReview" WHERE "reportId" NOT IN (SELECT id FROM "Report"); DELETE FROM "AnnotationSession" WHERE "reportId" NOT IN (SELECT id FROM "Report");`);
    }
  } finally { connection.close(); }
}
const adapter = new PrismaBetterSqlite3({ url: databasePath });
export const db = new PrismaClient({ adapter });
export const localPaths = { dataRoot, databasePath, scans: path.join(dataRoot, "data", "mesh"), results: path.join(dataRoot, "results"), logs: path.join(dataRoot, "logs") };

type InitialAdmin = { username: string; password: string; displayName?: string; department?: string };
export async function ensureInitialAdmin() {
  if (await db.user.count()) return;
  const configPath = process.env.AIS_INITIAL_ADMIN_FILE || path.join(process.cwd(), "deployment", "initial-admin.json");
  if (!existsSync(configPath)) throw new Error(`未找到首次部署管理员配置：${configPath}`);
  const initial = JSON.parse(readFileSync(configPath, "utf8")) as InitialAdmin;
  if (!initial.username || !initial.password || initial.password.length < 6) throw new Error("首次部署管理员配置无效，密码至少需要 6 位。");
  // 首次运行只创建系统管理员账号：系统管理员不归属任何机构，机构在后续创建账号时按需创建
  await db.user.create({ data: { username: initial.username, passwordHash: await bcrypt.hash(initial.password, 12), displayName: initial.displayName || initial.username, department: initial.department, institutionId: null, role: "system_admin" } });
  renameSync(configPath, `${configPath}.consumed`);
}
export function createToken(userId: string) { return Buffer.from(`${userId}:${Date.now()}:${crypto.randomUUID()}`).toString("base64url"); }
export function parseToken(token?: string) { if (!token) return null; try { return Buffer.from(token, "base64url").toString().split(":")[0] || null; } catch { return null; } }
export async function audit(userId: string | undefined, action: string, entity: string, entityId?: string, payload?: unknown) { await db.auditLog.create({ data: { userId, action, entity, entityId, payload: payload ? JSON.stringify(payload) : undefined } }); }
