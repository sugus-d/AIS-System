import path from "node:path";
import { existsSync } from "node:fs";
import { Router } from "express";
import { db, audit } from "../services/database";
import { canAccessCase } from "../middleware/access";
import { severityZh } from "../services/algorithm";

const router = Router();
const images = new Set(["curvature_mean", "curvature_gauss", "roughness", "normal_angle", "landmarks", "back", "moire", "waterfall"]);
const csv = (value: unknown) => `"${String(value ?? "").replaceAll('"', '""')}"`;
const html = (value: unknown) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
function decode(report: any) { try { return JSON.parse(report.resultJson); } catch { return {}; } }
function present(report: any) { const result = decode(report); return { id: report.id, caseId: report.caseId, fileId: report.fileId, version: report.version, cobb: report.cobbAngle, cobbAngle: report.cobbAngle, severity: report.severity, aisLevel: severityZh(report.severity), modelId: report.modelId, subjectId: result.subject_id, clinical: result.clinical, indices: result.indices, bodyParams: result.body_params, landmarks: result.landmarks, outputs: result.outputs || {}, createdAt: report.createdAt, completeTime: report.createdAt, status: report.review?.status ?? "under_review", ...result.clinician }; }
router.get("/", async (req: any, res) => {
  const reports = await db.report.findMany({ include: { case: true, file: true, review: true }, orderBy: { createdAt: "desc" } });
  let list = reports.filter((report) => canAccessCase(req.user, report.case));
  if (typeof req.query.caseId === "string") list = list.filter((report) => report.caseId === req.query.caseId);
  if (typeof req.query.caseName === "string") list = list.filter((report) => report.case.name.includes(req.query.caseName));
  if (typeof req.query.aisLevel === "string") list = list.filter((report) => severityZh(report.severity) === req.query.aisLevel || report.severity === req.query.aisLevel);
  const page = Math.max(1, Number(req.query.page || 1)); const pageSize = Math.min(100, Math.max(1, Number(req.query.pageSize || 20)));
  const data = list.slice((page - 1) * pageSize, page * pageSize).map((report) => ({ ...present(report), caseName: report.case.name, caseGender: report.case.gender, screeningDate: report.file.scanTime || report.file.createdAt }));
  res.json({ success: true, data: { list: data, total: list.length, page, pageSize } });
});
router.get("/:id/images/:image", async (req: any, res) => {
  if (!images.has(req.params.image)) return res.status(404).json({ success: false, message: "Report image not found." });
  const report = await db.report.findUnique({ where: { id: req.params.id }, include: { case: true } });
  if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." });
  const root = path.resolve(report.artifactDirectory); const image = path.resolve(root, "report", `${req.params.image}.png`);
  if (!image.startsWith(`${root}${path.sep}`) || !existsSync(image)) return res.status(404).json({ success: false, message: "Report image has not been generated." });
  return res.sendFile(image);
});
router.get("/:id", async (req: any, res) => { const report = await db.report.findUnique({ where: { id: req.params.id }, include: { case: true, file: true, task: true, review: true } }); if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." }); const data = present(report); const token = typeof req.headers.authorization === "string" ? req.headers.authorization.replace("Bearer ", "") : (typeof req.query.token === "string" ? req.query.token : ""); const imageUrl = (name: string) => { const base = `/api/reports/${report.id}/images/${name}`; return token ? `${base}?token=${encodeURIComponent(token)}` : base; }; res.json({ success: true, data: { ...data, case: report.case, file: report.file, backImage: imageUrl("back"), annotatedImage: imageUrl("landmarks"), heatmapImage: imageUrl("curvature_mean"), moireImage: imageUrl("moire"), normalAngleImage: imageUrl("normal_angle"), versions: [{ version: report.version, createdAt: report.createdAt, createdBy: report.task?.submittedById || "system" }] } }); });
router.put("/:id/diagnosis", async (req: any, res) => { const report = await db.report.findUnique({ where: { id: req.params.id }, include: { case: true, review: true } }); if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." }); const result = decode(report); result.clinician = { clinicalDiagnosis: req.body?.clinicalDiagnosis || "", followUpAdvice: req.body?.followUpAdvice || "", treatmentPlan: req.body?.treatmentPlan || "", diagnosisEdited: true, editedAt: new Date().toISOString(), editedBy: req.user.id }; await db.report.update({ where: { id: report.id }, data: { resultJson: JSON.stringify(result) } }); await audit(req.user.id, "update_diagnosis", "Report", report.id); res.json({ success: true, data: present({ ...report, resultJson: JSON.stringify(result) }) }); });
router.get("/:id/versions", async (req: any, res) => { const report = await db.report.findUnique({ where: { id: req.params.id }, include: { case: true } }); if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." }); res.json({ success: true, data: [{ version: report.version, createdAt: report.createdAt, diagnosisEdited: Boolean(decode(report).clinician?.diagnosisEdited) }] }); });
router.get("/:id/export", async (req: any, res) => {
  const report = await db.report.findUnique({ where: { id: req.params.id }, include: { case: true, file: true } });
  if (!report || !canAccessCase(req.user, report.case)) return res.status(404).json({ success: false, message: "Report not found." });
  const result = decode(report); const title = `AIS 报告 ${report.case.caseNumber || report.id}`;
  const document = `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>${html(title)}</title><style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:36px;color:#172033}h1{font-size:24px}table{border-collapse:collapse;width:100%;margin-top:20px}th,td{border:1px solid #cbd5e1;padding:9px;text-align:left}th{width:34%;background:#eff6ff}@media print{body{margin:14mm}}</style><h1>${html(title)}</h1><p>生成时间：${html(new Date().toLocaleString("zh-CN"))}</p><table><tr><th>病例编号</th><td>${html(report.case.caseNumber)}</td></tr><tr><th>姓名</th><td>${html(report.case.name)}</td></tr><tr><th>Cobb 角</th><td>${html(report.cobbAngle)}°</td></tr><tr><th>AIS 等级</th><td>${html(severityZh(report.severity))}</td></tr><tr><th>算法等级</th><td>${html(report.severity)}</td></tr><tr><th>模型版本</th><td>${html(report.modelId)}</td></tr><tr><th>扫描时间</th><td>${html(report.file.scanTime || report.file.createdAt)}</td></tr><tr><th>临床数据</th><td><pre>${html(JSON.stringify(result.clinical || {}, null, 2))}</pre></td></tr></table></html>`;
  res.setHeader("Content-Type", "text/html; charset=utf-8"); res.setHeader("Content-Disposition", `attachment; filename="${report.case.caseNumber || report.id}-AIS-report.html"`); return res.send(document);
});
router.post("/batchExport", async (req: any, res) => {
  const ids = Array.isArray(req.body?.reportIds) ? req.body.reportIds.filter((id: unknown): id is string => typeof id === "string") : undefined;
  const reports = await db.report.findMany({ where: ids?.length ? { id: { in: ids } } : undefined, include: { case: true, file: true }, orderBy: { createdAt: "desc" } });
  const allowed = reports.filter((report) => canAccessCase(req.user, report.case));
  const body = [["报告ID", "病例编号", "姓名", "Cobb角", "AIS等级", "模型版本", "生成时间"], ...allowed.map((report) => [report.id, report.case.caseNumber, report.case.name, report.cobbAngle, severityZh(report.severity), report.modelId, report.createdAt.toISOString()])].map((row) => row.map(csv).join(",")).join("\r\n");
  res.setHeader("Content-Type", "text/csv; charset=utf-8"); res.setHeader("Content-Disposition", 'attachment; filename="AIS-reports.csv"'); return res.send(`\uFEFF${body}`);
});
export default router;
