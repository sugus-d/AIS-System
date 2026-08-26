import crypto from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { Router } from "express";
import { db, audit } from "../services/database";
import { canAccessCase, requireRoles } from "../middleware/access";
import { enqueueAnalysisTask } from "../services/analysis-runner";

const router = Router();
const baseUrl = process.env.ANNOTATION_BASE_URL || "http://127.0.0.1:18765";
const secret = process.env.ANNOTATION_TOKEN_SECRET || "replace-this-deployment-secret";
const sign = (value: string) => crypto.createHmac("sha256", secret).update(value).digest("hex");
const bilateral = ["neck_root", "shoulder_transition", "scapular_peaks", "axilla", "waist", "waist_lower"];
function flattenGroundTruth(raw: Record<string, any>) {
  const flat: Record<string, unknown> = {};
  for (const name of bilateral) { if (Array.isArray(raw[name])) { flat[`${name}_L`] = raw[name][0]; flat[`${name}_R`] = raw[name][1]; } else if (raw[name] && typeof raw[name] === "object") { flat[`${name}_L`] = raw[name].L; flat[`${name}_R`] = raw[name].R; } }
  const spineKeys = ["neck_root_spine_point", "scapular_spine_point", "axilla_spine_point", "waist_spine_point", "waist_lower_spine_point", "thoracic_spine_point"];
  spineKeys.forEach((key, index) => { flat[key] = raw.spine_points?.[index]; });
  const missing = Object.entries(flat).filter(([, value]) => !Array.isArray(value) || value.length !== 3).map(([key]) => key);
  if (missing.length) throw new Error(`Annotation is incomplete: ${missing.join(", ")}`);
  return flat;
}
/** 找最新的笔刷编辑后 ROI（roi_edited_*.ply），优先于算法原始 roi.ply。 */
function findEditedRoi(caseId: string): string | null {
  const resultsRoot = process.env.AIS_RESULTS_ROOT || "";
  if (!resultsRoot) return null;
  const dir = path.join(resultsRoot, "labeling", "cache", caseId, "extract_roi");
  if (!existsSync(dir)) return null;
  let latest: string | null = null;
  let latestTime = 0;
  for (const file of readdirSync(dir)) {
    if (!file.startsWith("roi_edited_") || !file.endsWith(".ply")) continue;
    const full = path.join(dir, file);
    const stat = statSync(full);
    if (stat.mtimeMs > latestTime) { latestTime = stat.mtimeMs; latest = full; }
  }
  return latest;
}
/** 用标注 landmarks + 编辑后 ROI 构建 predict 重分析任务；无标注或 ROI 不可用时返回 null。 */
async function buildManualReanalysisTask(report: any, user: any) {
  const groundTruth = path.join(process.env.AIS_RESULTS_ROOT || "", "ground-truth", report.caseId, "ground_truth.json");
  if (!existsSync(groundTruth)) return null;
  const landmarks = flattenGroundTruth(JSON.parse(readFileSync(groundTruth, "utf8")));
  const original = JSON.parse(report.resultJson);
  const manualRoiPath = findEditedRoi(report.caseId) || original.outputs?.roi;
  if (typeof manualRoiPath !== "string" || !existsSync(manualRoiPath)) return null;
  return db.analysisTask.create({ data: { type: "annotation_reanalysis", caseId: report.caseId, fileId: report.fileId, submittedById: user.id, resultJson: JSON.stringify({ landmarks, manualRoiPath, sourceReportId: report.id }) } });
}
router.get("/subjects", requireRoles("system_admin"), async (_req, res) => { const reports = await db.report.findMany({ select: { id: true } }); res.json({ success: true, data: { subjects: reports.map((report) => report.id) } }); });
router.post("/sessions", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
  const report = await db.report.findUnique({ where: { id: req.body?.reportId || "" }, include: { case: true } });
  if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." });
  const subjectId = report.caseId; const expiresAt = new Date(Date.now() + 30 * 60 * 1000); const payload = `${report.id}.${subjectId}.${expiresAt.getTime()}`; const token = `${payload}.${sign(payload)}`;
  await db.annotationSession.create({ data: { reportId: report.id, subjectId, tokenHash: crypto.createHash("sha256").update(token).digest("hex"), expiresAt, updatedById: req.user.id } });
  await db.report.update({ where: { id: report.id }, data: { annotationStatus: "in_progress" } }); await audit(req.user.id, "annotation_session", "Report", report.id);
  const nodeBaseUrl = process.env.AIS_NODE_BASE_URL || "http://127.0.0.1:18080";
  const annotationUrl = new URL(`/subject/${encodeURIComponent(subjectId)}/3d`, baseUrl); annotationUrl.searchParams.set("reportId", report.id); annotationUrl.searchParams.set("subjectId", subjectId); annotationUrl.searchParams.set("token", token); annotationUrl.searchParams.set("returnUrl", `${nodeBaseUrl}/analysis-report?reportId=${encodeURIComponent(report.id)}`);
  res.json({ success: true, data: { token, annotationUrl: annotationUrl.toString(), expiresAt: expiresAt.toISOString() } });
});
router.post("/reports/:reportId/completed", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
  const report = await db.report.findUnique({ where: { id: req.params.reportId }, include: { case: true } }); if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." });
  const groundTruth = path.join(process.env.AIS_RESULTS_ROOT || "", "ground-truth", report.caseId, "ground_truth.json");
  if (!existsSync(groundTruth)) return res.status(409).json({ success: false, message: "No saved annotation ground truth was found." });
  try {
    const landmarks = flattenGroundTruth(JSON.parse(readFileSync(groundTruth, "utf8"))); const original = JSON.parse(report.resultJson); const manualRoiPath = findEditedRoi(report.caseId) || original.outputs?.roi;
    if (typeof manualRoiPath !== "string" || !existsSync(manualRoiPath)) return res.status(409).json({ success: false, message: "The source ROI artifact is unavailable for manual reanalysis." });
    const task = await db.analysisTask.create({ data: { type: "annotation_reanalysis", caseId: report.caseId, fileId: report.fileId, submittedById: req.user.id, resultJson: JSON.stringify({ landmarks, manualRoiPath, sourceReportId: report.id }) } });
    await db.annotationSession.updateMany({ where: { reportId: report.id, status: "active" }, data: { status: "completed", updatedById: req.user.id } }); await db.report.update({ where: { id: report.id }, data: { annotationStatus: "updated" } }); await audit(req.user.id, "annotation_completed", "Report", report.id, { reanalysisTaskId: task.id }); void enqueueAnalysisTask(task.id);
    res.status(202).json({ success: true, data: { reportId: report.id, status: "updated", reanalysisTaskId: task.id, updatedAt: new Date().toISOString(), updatedBy: req.user.username } });
  } catch (error) { res.status(422).json({ success: false, message: error instanceof Error ? error.message : "Annotation data is invalid." }); }
});
router.post("/reports/:reportId/reanalyze", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
  const report = await db.report.findUnique({ where: { id: req.params.reportId }, include: { case: true } });
  if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." });
  const inFlight = await db.analysisTask.findFirst({ where: { fileId: report.fileId, status: { in: ["pending", "running"] } } });
  if (inFlight) return res.status(409).json({ success: false, message: "该文件已有分析任务进行中，请稍后再试。" });
  const manual = await buildManualReanalysisTask(report, req.user);
  if (manual) { void enqueueAnalysisTask(manual.id); return res.status(202).json({ success: true, data: { id: manual.id, type: "annotation_reanalysis" } }); }
  const task = await db.analysisTask.create({ data: { id: crypto.randomUUID(), type: "single_analysis", caseId: report.caseId, fileId: report.fileId, submittedById: req.user.id } });
  void enqueueAnalysisTask(task.id);
  res.status(202).json({ success: true, data: { id: task.id, type: "single_analysis" } });
});
export default router;
