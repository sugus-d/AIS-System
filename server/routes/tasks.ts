import { randomUUID } from "node:crypto";
import { Router } from "express";
import { db, audit } from "../services/database";
import { enqueueAnalysisTask, taskView } from "../services/analysis-runner";
import { canAccessCase, requireRoles } from "../middleware/access";

const router = Router();
async function visibleTask(user: any, id: string) { const task = await db.analysisTask.findUnique({ where: { id }, include: { file: { include: { case: true } }, submittedBy: true } }); return task && canAccessCase(user, task.file.case) ? task : null; }
router.get("/", async (req: any, res) => {
  const tasks = await db.analysisTask.findMany({ include: { file: { include: { case: true } }, submittedBy: true }, orderBy: { createdAt: "desc" } });
  let list = tasks.filter((task) => canAccessCase(req.user, task.file.case));
  for (const key of ["type", "status"] as const) if (typeof req.query[key] === "string") list = list.filter((task) => task[key] === req.query[key]);
  if (typeof req.query.keyword === "string") list = list.filter((task) => task.id.includes(req.query.keyword) || task.caseId.includes(req.query.keyword));
  const page = Math.max(1, Number(req.query.page || 1)); const pageSize = Math.min(100, Math.max(1, Number(req.query.pageSize || 20)));
  res.json({ success: true, data: { list: list.slice((page - 1) * pageSize, page * pageSize).map((task) => ({ ...taskView(task), submitter: task.submittedBy.username })), total: list.length, page, pageSize } });
});
router.get("/stats", async (req: any, res) => { const tasks = (await db.analysisTask.findMany({ include: { file: { include: { case: true } } } })).filter((task) => canAccessCase(req.user, task.file.case)); const data = Object.fromEntries(["pending", "running", "success", "failed", "cancelled"].map((status) => [status, tasks.filter((task) => task.status === status).length])); res.json({ success: true, data: { total: tasks.length, ...data } }); });
router.get("/:id", async (req: any, res) => { const task = await visibleTask(req.user, req.params.id); if (!task) return res.status(404).json({ success: false, message: "Task not found." }); res.json({ success: true, data: { ...taskView(task), submitter: task.submittedBy.username, subtasks: [] } }); });
router.post("/:id/cancel", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const task = await visibleTask(req.user, req.params.id); if (!task) return res.status(404).json({ success: false, message: "Task not found." }); if (!['pending', 'running'].includes(task.status)) return res.status(400).json({ success: false, message: "Task cannot be cancelled." }); await db.analysisTask.update({ where: { id: task.id }, data: { status: "cancelled", finishedAt: new Date() } }); await audit(req.user.id, "cancel", "AnalysisTask", task.id); res.json({ success: true }); });
async function retry(user: any, source: any) { const task = await db.analysisTask.create({ data: { id: randomUUID(), type: source.type, caseId: source.caseId, fileId: source.fileId, submittedById: user.id } }); await audit(user.id, "retry", "AnalysisTask", task.id, { sourceTaskId: source.id }); void enqueueAnalysisTask(task.id); return taskView(task); }
router.post("/:id/retry", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const task = await visibleTask(req.user, req.params.id); if (!task) return res.status(404).json({ success: false, message: "Task not found." }); if (!['failed', 'cancelled'].includes(task.status)) return res.status(400).json({ success: false, message: "Task cannot be retried." }); res.status(202).json({ success: true, data: await retry(req.user, task) }); });
router.post("/batchCancel", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const ids = Array.isArray(req.body?.taskIds) ? req.body.taskIds : []; let count = 0; for (const id of ids) { const task = typeof id === "string" ? await visibleTask(req.user, id) : null; if (task && ['pending', 'running'].includes(task.status)) { await db.analysisTask.update({ where: { id: task.id }, data: { status: "cancelled", finishedAt: new Date() } }); count += 1; } } res.json({ success: true, data: { count } }); });
router.post("/batchRetry", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => { const ids = Array.isArray(req.body?.taskIds) ? req.body.taskIds : []; const tasks: any[] = []; for (const id of ids) { const source = typeof id === "string" ? await visibleTask(req.user, id) : null; if (source && ['failed', 'cancelled'].includes(source.status)) tasks.push(await retry(req.user, source)); } res.status(202).json({ success: true, data: { tasks, count: tasks.length } }); });
export default router;
