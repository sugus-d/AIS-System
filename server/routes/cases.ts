import { unlink } from "node:fs/promises";
import { Router } from "express";
import { db, audit } from "../services/database";
import { canAccessCase, requireRoles } from "../middleware/access";

const router = Router();
const present = (item: any) => ({ ...item, height: item.heightCm, weight: item.weightKg, fileCount: item._count?.files ?? 0, reportCount: item._count?.reports ?? 0, latestReportTime: item.reports?.[0]?.createdAt ?? null, latestReport: item.reports?.[0] ? { id: item.reports[0].id, cobbAngle: item.reports[0].cobbAngle, severity: item.reports[0].severity, reportStatus: item.reports[0].review?.status ?? item.reports[0].annotationStatus ?? null } : null });
const computeStatus = (item: any, inFlight?: Set<string>) => { if (inFlight?.has(item.id)) return "analyzing"; const latest = Array.isArray(item.reports) ? item.reports[0] : undefined; if (latest) return latest.review?.status || "under_review"; if ((item._count?.files ?? 0) > 0) return "pending_analysis"; return "pending_upload"; };

// 档案归属机构：请求显式指定 → 创建者本人所属机构 → 系统内唯一机构（系统管理员建档时自动归入）
const resolveCaseInstitutionId = async (user: any, body: any): Promise<string> => {
  const requested = typeof body?.institutionId === "string" ? body.institutionId : "";
  if (requested) {
    const found = await db.institution.findUnique({ where: { id: requested }, select: { id: true } });
    if (found) return found.id;
  }
  if (user.institutionId) return user.institutionId;
  const institutions = await db.institution.findMany({ select: { id: true }, orderBy: { createdAt: "asc" } });
  if (institutions.length === 1) return institutions[0].id;
  throw new Error(institutions.length === 0 ? "系统尚未创建机构，请先在「机构管理」中创建机构。" : "请选择档案所属机构。");
};

router.get("/", async (req: any, res) => {
  const page = Math.max(1, Number(req.query.page || 1)); const pageSize = Math.min(Math.max(1, Number(req.query.pageSize || 20)), 100);
  const where: any = req.query.keyword ? { OR: [{ caseNumber: { contains: String(req.query.keyword) } }, { name: { contains: String(req.query.keyword) } }] } : {};
  const [rows, inFlightRows] = await Promise.all([
    db.case.findMany({ where, include: { _count: { select: { files: true, reports: true } }, reports: { include: { review: true }, orderBy: { createdAt: "desc" }, take: 1 } }, orderBy: { updatedAt: "desc" } }),
    db.analysisTask.findMany({ where: { status: { in: ["pending", "running"] } }, select: { caseId: true } }),
  ]);
  const inFlight = new Set(inFlightRows.map((task) => task.caseId));
  let list = rows.filter((item) => canAccessCase(req.user, item));
  if (typeof req.query.status === "string" && req.query.status && req.query.status !== "all") list = list.filter((item) => computeStatus(item, inFlight) === req.query.status);
  res.json({ success: true, data: { list: list.slice((page - 1) * pageSize, page * pageSize).map((item) => ({ ...present(item), status: computeStatus(item, inFlight) })), total: list.length, page, pageSize } });
});
router.get("/stats/summary", async (req: any, res) => { const rows = (await db.case.findMany({ include: { files: true, reports: true } })).filter((item) => canAccessCase(req.user, item)); res.json({ success: true, data: { total: rows.length, files: rows.reduce((n, item) => n + item.files.length, 0), reports: rows.reduce((n, item) => n + item.reports.length, 0) } }); });
router.get("/:id", async (req: any, res) => { const item = await db.case.findUnique({ where: { id: req.params.id }, include: { files: { orderBy: { createdAt: "desc" }, include: { tasks: { orderBy: { createdAt: "desc" }, take: 1 } } }, reports: { orderBy: { version: "desc" } } } }); if (!item || !canAccessCase(req.user, item)) return res.status(404).json({ success: false, message: "Case not found." }); return res.json({ success: true, data: present(item) }); });

async function createCase(user: any, body: any) {
  const institutionId = await resolveCaseInstitutionId(user, body);
  if (!body?.name || !body?.gender || !body?.birthDate || !body?.height || !body?.weight) throw new Error("姓名、性别、出生日期、身高和体重为必填项。");
  const existingNumbers = await db.case.findMany({ select: { caseNumber: true } });
  let maxNumber = 0;
  for (const row of existingNumbers) { const match = /^CASE(\d+)$/.exec(row.caseNumber); if (match) maxNumber = Math.max(maxNumber, Number(match[1])); }
  return db.case.create({ data: { caseNumber: `CASE${String(maxNumber + 1).padStart(6, "0")}`, name: body.name, gender: body.gender, birthDate: new Date(body.birthDate), heightCm: Number(body.height), weightKg: Number(body.weight), medicalHistory: body.medicalHistory, idNumber: body.idNumber, phone: body.phone, department: body.department, doctor: body.doctor, ownerId: user.id, institutionId } });
}
router.post("/", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { try { const item = await createCase(req.user, req.body); await audit(req.user.id, "create", "Case", item.id); res.status(201).json({ success: true, data: present(item) }); } catch (error) { res.status(400).json({ success: false, message: error instanceof Error ? error.message : "Cannot create case." }); } });
router.put("/:id", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const existing = await db.case.findUnique({ where: { id: req.params.id } }); if (!existing || !canAccessCase(req.user, existing)) return res.status(404).json({ success: false, message: "Case not found." }); const body = req.body || {}; const item = await db.case.update({ where: { id: existing.id }, data: { name: body.name, gender: body.gender, birthDate: body.birthDate ? new Date(body.birthDate) : undefined, heightCm: body.height ? Number(body.height) : undefined, weightKg: body.weight ? Number(body.weight) : undefined, medicalHistory: body.medicalHistory, idNumber: body.idNumber, phone: body.phone, department: body.department, doctor: body.doctor } }); await audit(req.user.id, "update", "Case", item.id); res.json({ success: true, data: present(item) }); });
async function cascadeDeleteCase(tx: any, caseId: string) {
  await tx.reportReview.deleteMany({ where: { report: { caseId } } });
  await tx.annotationSession.deleteMany({ where: { report: { caseId } } });
  await tx.report.deleteMany({ where: { caseId } });
  await tx.analysisTask.deleteMany({ where: { caseId } });
  await tx.scanFile.deleteMany({ where: { caseId } });
  await tx.case.delete({ where: { id: caseId } });
}
router.delete("/:id", async (req: any, res) => { const item = await db.case.findUnique({ where: { id: req.params.id }, include: { files: true } }); if (!item || !canAccessCase(req.user, item)) return res.status(404).json({ success: false, message: "Case not found." }); await db.$transaction(async (tx) => { await cascadeDeleteCase(tx, item.id); }); for (const file of item.files) await unlink(file.storedPath).catch(() => undefined); await audit(req.user.id, "delete", "Case", item.id); res.json({ success: true }); });
router.post("/batch", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const created: any[] = []; const skipped: any[] = []; for (const body of Array.isArray(req.body?.cases) ? req.body.cases : []) { try { created.push(await createCase(req.user, body)); } catch (error) { skipped.push({ reason: error instanceof Error ? error.message : "Invalid case." }); } } res.status(201).json({ success: true, data: { cases: created.map(present), skipped } }); });
router.post("/batch-delete", async (req: any, res) => { const ids = Array.isArray(req.body?.ids) ? req.body.ids.filter((id: unknown): id is string => typeof id === "string") : []; const rows = await db.case.findMany({ where: { id: { in: ids } }, include: { files: true } }); const allowed = rows.filter((item) => canAccessCase(req.user, item)); await db.$transaction(async (tx) => { for (const item of allowed) await cascadeDeleteCase(tx, item.id); }); for (const item of allowed) for (const file of item.files) await unlink(file.storedPath).catch(() => undefined); res.json({ success: true, data: { count: allowed.length } }); });
export default router;
