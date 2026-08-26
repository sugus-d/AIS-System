import { Router } from "express";
import { db, audit } from "../services/database";
import { canAccessCase, requireRoles } from "../middleware/access";

const router = Router();
const present = (item: any) => ({ ...item, height: item.heightCm, weight: item.weightKg, fileCount: item._count?.files ?? 0, reportCount: item._count?.reports ?? 0 });
const institutionFor = (user: any, body: any) => user.role === "system_admin" && typeof body?.institutionId === "string" ? body.institutionId : user.institutionId;

router.get("/", async (req: any, res) => {
  const page = Math.max(1, Number(req.query.page || 1)); const pageSize = Math.min(Math.max(1, Number(req.query.pageSize || 20)), 100);
  const where: any = req.query.keyword ? { OR: [{ caseNumber: { contains: String(req.query.keyword) } }, { name: { contains: String(req.query.keyword) } }] } : {};
  const rows = await db.case.findMany({ where, include: { _count: { select: { files: true, reports: true } } }, orderBy: { updatedAt: "desc" } });
  const list = rows.filter((item) => canAccessCase(req.user, item));
  res.json({ success: true, data: { list: list.slice((page - 1) * pageSize, page * pageSize).map(present), total: list.length, page, pageSize } });
});
router.get("/stats/summary", async (req: any, res) => { const rows = (await db.case.findMany({ include: { files: true, reports: true } })).filter((item) => canAccessCase(req.user, item)); res.json({ success: true, data: { total: rows.length, files: rows.reduce((n, item) => n + item.files.length, 0), reports: rows.reduce((n, item) => n + item.reports.length, 0) } }); });
router.get("/:id", async (req: any, res) => { const item = await db.case.findUnique({ where: { id: req.params.id }, include: { files: { orderBy: { createdAt: "desc" } }, reports: { orderBy: { version: "desc" } } } }); if (!item || !canAccessCase(req.user, item)) return res.status(404).json({ success: false, message: "Case not found." }); return res.json({ success: true, data: present(item) }); });

async function createCase(user: any, body: any) {
  const institutionId = institutionFor(user, body); if (!institutionId) throw new Error("User institution is not configured.");
  if (!body?.name || !body?.gender || !body?.birthDate || !body?.height || !body?.weight) throw new Error("姓名、性别、出生日期、身高和体重为必填项。");
  const count = await db.case.count();
  return db.case.create({ data: { caseNumber: `CASE${String(count + 1).padStart(6, "0")}`, name: body.name, gender: body.gender, birthDate: new Date(body.birthDate), heightCm: Number(body.height), weightKg: Number(body.weight), medicalHistory: body.medicalHistory, idNumber: body.idNumber, phone: body.phone, department: body.department, doctor: body.doctor, ownerId: user.id, institutionId } });
}
router.post("/", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { try { const item = await createCase(req.user, req.body); await audit(req.user.id, "create", "Case", item.id); res.status(201).json({ success: true, data: present(item) }); } catch (error) { res.status(400).json({ success: false, message: error instanceof Error ? error.message : "Cannot create case." }); } });
router.put("/:id", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const existing = await db.case.findUnique({ where: { id: req.params.id } }); if (!existing || !canAccessCase(req.user, existing)) return res.status(404).json({ success: false, message: "Case not found." }); const body = req.body || {}; const item = await db.case.update({ where: { id: existing.id }, data: { name: body.name, gender: body.gender, birthDate: body.birthDate ? new Date(body.birthDate) : undefined, heightCm: body.height ? Number(body.height) : undefined, weightKg: body.weight ? Number(body.weight) : undefined, medicalHistory: body.medicalHistory, idNumber: body.idNumber, phone: body.phone, department: body.department, doctor: body.doctor } }); await audit(req.user.id, "update", "Case", item.id); res.json({ success: true, data: present(item) }); });
router.delete("/:id", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const item = await db.case.findUnique({ where: { id: req.params.id } }); if (!item || !canAccessCase(req.user, item)) return res.status(404).json({ success: false, message: "Case not found." }); await db.case.delete({ where: { id: item.id } }); await audit(req.user.id, "delete", "Case", item.id); res.json({ success: true }); });
router.post("/batch", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const created: any[] = []; const skipped: any[] = []; for (const body of Array.isArray(req.body?.cases) ? req.body.cases : []) { try { created.push(await createCase(req.user, body)); } catch (error) { skipped.push({ reason: error instanceof Error ? error.message : "Invalid case." }); } } res.status(201).json({ success: true, data: { cases: created.map(present), skipped } }); });
router.post("/batch-delete", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const ids = Array.isArray(req.body?.ids) ? req.body.ids.filter((id: unknown): id is string => typeof id === "string") : []; const rows = await db.case.findMany({ where: { id: { in: ids } } }); const allowed = rows.filter((item) => canAccessCase(req.user, item)).map((item) => item.id); await db.case.deleteMany({ where: { id: { in: allowed } } }); res.json({ success: true, data: { count: allowed.length } }); });
export default router;
