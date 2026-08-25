import { randomUUID } from "node:crypto";
import { Router } from "express";
import { db, audit } from "../services/database";
import { enqueueAnalysisTask, enqueueAnalysisTasksSequentially, taskView } from "../services/analysis-runner";
import { canAccessCase, requireRoles } from "../middleware/access";

const router = Router();
async function queue(user: any, caseId: string, fileId: string, type = "single_analysis") {
  const task = await db.analysisTask.create({ data: { id: randomUUID(), type, caseId, fileId, submittedById: user.id } });
  await audit(user.id, "queue_analysis", "AnalysisTask", task.id, { caseId, fileId, type });
  return taskView(task);
}
router.post("/single", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
  const caseId = typeof req.body?.caseId === "string" ? req.body.caseId : "";
  const subject = await db.case.findUnique({ where: { id: caseId }, include: { files: { orderBy: { createdAt: "desc" } } } });
  if (!subject || !canAccessCase(req.user, subject)) return res.status(404).json({ success: false, message: "Case not found." });
  const file = req.body?.fileId ? subject.files.find((item) => item.id === req.body.fileId) : subject.files[0];
  if (!file) return res.status(400).json({ success: false, message: "Upload a PLY scan before analysis." });
  const task = await queue(req.user, subject.id, file.id); void enqueueAnalysisTask(task.id); res.status(202).json({ success: true, data: task });
});
router.post("/batch", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
  const caseIds = Array.isArray(req.body?.caseIds) ? req.body.caseIds.filter((id: unknown): id is string => typeof id === "string") : [];
  const subjects = await db.case.findMany({ where: { id: { in: caseIds } }, include: { files: { orderBy: { createdAt: "desc" } } } });
  const tasks: any[] = []; const skipped: Array<{ caseId: string; reason: string }> = [];
  for (const subject of subjects) { if (!canAccessCase(req.user, subject)) { skipped.push({ caseId: subject.id, reason: "Access denied." }); continue; } if (!subject.files[0]) { skipped.push({ caseId: subject.id, reason: "No PLY scan." }); continue; } tasks.push(await queue(req.user, subject.id, subject.files[0].id, "batch_analysis")); }
  for (const caseId of caseIds.filter((id) => !subjects.some((subject) => subject.id === id))) skipped.push({ caseId, reason: "Case not found." });
  void enqueueAnalysisTasksSequentially(tasks.map((task) => task.id));
  res.status(202).json({ success: true, data: { tasks, skipped } });
});
router.get("/check/:caseId", async (req: any, res) => {
  const subject = await db.case.findUnique({ where: { id: req.params.caseId }, include: { files: true } });
  if (!subject || !canAccessCase(req.user, subject)) return res.status(404).json({ success: false, message: "Case not found." });
  const hasBasicInfo = Boolean(subject.gender && subject.heightCm > 0 && subject.weightKg > 0); const hasFile = subject.files.length > 0;
  res.json({ success: true, data: { hasBasicInfo, hasFile, canAnalyze: hasBasicInfo && hasFile } });
});
export default router;
