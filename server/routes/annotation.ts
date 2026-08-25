import crypto from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
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
router.get("/subjects", requireRoles("system_admin"), async (_req, res) => { const reports = await db.report.findMany({ select: { id: true } }); res.json({ success: true, data: { subjects: reports.map((report) => report.id) } }); });
router.post("/sessions", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
  const report = await db.report.findUnique({ where: { id: req.body?.reportId || "" }, include: { case: true } });
  if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." });
  const subjectId = report.caseId; const expiresAt = new Date(Date.now() + 30 * 60 * 1000); const payload = `${report.id}.${subjectId}.${expiresAt.getTime()}`; const token = `${payload}.${sign(payload)}`;
  await db.annotationSession.create({ data: { reportId: report.id, subjectId, tokenHash: crypto.createHash("sha256").update(token).digest("hex"), expiresAt, updatedById: req.user.id } });
  await db.report.update({ where: { id: report.id }, data: { annotationStatus: "in_progress" } }); await audit(req.user.id, "annotation_session", "Report", report.id);
  const nodeBaseUrl = process.env.AIS_NODE_BASE_URL || "http://127.0.0.1:18080";
  const annotationUrl = new URL(`/subject/${encodeURIComponent(subjectId)}/2d`, baseUrl); annotationUrl.searchParams.set("reportId", report.id); annotationUrl.searchParams.set("subjectId", subjectId); annotationUrl.searchParams.set("token", token); annotationUrl.searchParams.set("returnUrl", `${nodeBaseUrl}/analysis-report?reportId=${encodeURIComponent(report.id)}`);
  res.json({ success: true, data: { token, annotationUrl: annotationUrl.toString(), expiresAt: expiresAt.toISOString() } });
});
router.post("/reports/:reportId/completed", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
  const report = await db.report.findUnique({ where: { id: req.params.reportId }, include: { case: true } }); if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." });
  const groundTruth = path.join(process.env.AIS_RESULTS_ROOT || "", "ground-truth", report.caseId, "ground_truth.json");
  if (!existsSync(groundTruth)) return res.status(409).json({ success: false, message: "No saved annotation ground truth was found." });
  try {
    const landmarks = flattenGroundTruth(JSON.parse(readFileSync(groundTruth, "utf8"))); const original = JSON.parse(report.resultJson); const manualRoiPath = original.outputs?.roi;
    if (typeof manualRoiPath !== "string" || !existsSync(manualRoiPath)) return res.status(409).json({ success: false, message: "The source ROI artifact is unavailable for manual reanalysis." });
    const task = await db.analysisTask.create({ data: { type: "annotation_reanalysis", caseId: report.caseId, fileId: report.fileId, submittedById: req.user.id, resultJson: JSON.stringify({ landmarks, manualRoiPath, sourceReportId: report.id }) } });
    await db.annotationSession.updateMany({ where: { reportId: report.id, status: "active" }, data: { status: "completed", updatedById: req.user.id } }); await db.report.update({ where: { id: report.id }, data: { annotationStatus: "updated" } }); await audit(req.user.id, "annotation_completed", "Report", report.id, { reanalysisTaskId: task.id }); void enqueueAnalysisTask(task.id);
    res.status(202).json({ success: true, data: { reportId: report.id, status: "updated", reanalysisTaskId: task.id, updatedAt: new Date().toISOString(), updatedBy: req.user.username } });
  } catch (error) { res.status(422).json({ success: false, message: error instanceof Error ? error.message : "Annotation data is invalid." }); }
});
export default router;
