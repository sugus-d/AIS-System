import { createHash, randomUUID } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { rename, unlink } from "node:fs/promises";
import path from "node:path";
import multer from "multer";
import { Router } from "express";
import { db, localPaths, audit } from "../services/database";
import { canAccessCase, requireRoles } from "../middleware/access";

const router = Router();
const upload = multer({ dest: path.join(localPaths.dataRoot, "uploads"), limits: { fileSize: Number(process.env.AIS_MAX_UPLOAD_BYTES || 200 * 1024 * 1024) } });
// kind：scan = 3D 扫描（PLY），xray = X 光影像（JPG/PNG/WebP）
const kindOf = (file: any) => { const n = String(file.originalName || "").toLowerCase(); return n.endsWith(".ply") ? "scan" : "xray"; };
const isPly = (name: string) => name.endsWith(".ply");
const isImage = (name: string) => /\.(jpe?g|png|webp)$/.test(name);
const present = (file: any) => ({ id: file.id, caseId: file.caseId, fileName: file.originalName, fileSize: file.sizeBytes, scanTime: file.scanTime, uploadTime: file.createdAt, path: file.storedPath, status: file.status, sha256: file.sha256, department: file.department ?? null, doctor: file.doctor ?? null, kind: kindOf(file) });
async function checksum(filePath: string) { return await new Promise<string>((resolve, reject) => { const hash = createHash("sha256"); createReadStream(filePath).on("data", (chunk) => hash.update(chunk)).on("error", reject).on("end", () => resolve(hash.digest("hex"))); }); }

router.get("/", async (req: any, res) => {
  const caseId = typeof req.query.caseId === "string" ? req.query.caseId : undefined;
  const files = await db.scanFile.findMany({ where: caseId ? { caseId } : {}, include: { case: true }, orderBy: { createdAt: "desc" } });
  const visible = files.filter((file) => canAccessCase(req.user, file.case));
  res.json({ success: true, data: { list: visible.map(present), total: visible.length, page: 1, pageSize: visible.length } });
});

router.post("/", requireRoles("system_admin", "institution_admin", "operator"), upload.single("file"), async (req: any, res) => {
  const uploaded = req.file;
  const discard = async () => { if (uploaded) await unlink(uploaded.path).catch(() => undefined); };
  if (!uploaded) return res.status(400).json({ success: false, message: "A file is required." });
  const originalName = (uploaded.originalname || "").toLowerCase();
  if (!isPly(originalName) && !isImage(originalName)) { await discard(); return res.status(422).json({ success: false, message: "仅支持 .ply 扫描文件或 .jpg/.png/.webp 影像。" }); }
  const caseId = typeof req.body.caseId === "string" ? req.body.caseId : "";
  const subject = await db.case.findUnique({ where: { id: caseId } });
  if (!subject || !canAccessCase(req.user, subject)) { await discard(); return res.status(404).json({ success: false, message: "Case not found." }); }
  // scan → scans/<caseId>/<id>.ply；xray → scans/<caseId>/xray/<id>.<ext>
  const directory = isPly(originalName) ? path.join(localPaths.scans, caseId) : path.join(localPaths.scans, caseId, "xray");
  mkdirSync(directory, { recursive: true });
  const id = randomUUID();
  const ext = isPly(originalName) ? "ply" : (originalName.match(/\.([^.]+)$/)?.[1] || "png");
  const destination = path.join(directory, `${id}.${ext}`);
  try {
    const sha256 = await checksum(uploaded.path); await rename(uploaded.path, destination);
    // 记录上传者（新建报告者）身份：科室/医生，供分析完成前（待分析阶段）的列表展示
    let identity: { department: string | null; doctor: string | null } | null = null;
    if (isPly(originalName)) {
      const operator = await db.user.findUnique({ where: { id: req.user.id }, select: { department: true, displayName: true } }).catch(() => null);
      identity = operator ? { department: operator.department ?? null, doctor: operator.displayName ?? null } : null;
    }
    const file = await db.scanFile.create({ data: { id, caseId, originalName: uploaded.originalname, storedPath: destination, sha256, sizeBytes: uploaded.size, scanTime: req.body.scanTime ? new Date(req.body.scanTime) : null, department: identity?.department ?? null, doctor: identity?.doctor ?? null } });
    await db.case.update({ where: { id: caseId }, data: { status: "pending_analysis" } });
    await audit(req.user.id, "upload", "ScanFile", file.id, { caseId, sha256, sizeBytes: file.sizeBytes, kind: kindOf(file) });
    return res.status(201).json({ success: true, data: present(file) });
  } catch (error) { await discard(); return res.status(500).json({ success: false, message: error instanceof Error ? error.message : "File storage failed." }); }
});
router.get("/:id/download", async (req: any, res) => {
  const file = await db.scanFile.findUnique({ where: { id: req.params.id }, include: { case: true } });
  if (!file || !canAccessCase(req.user, file.case)) return res.status(404).json({ success: false, message: "File not found." });
  if (!existsSync(file.storedPath)) return res.status(404).json({ success: false, message: "File content missing." });
  return res.sendFile(path.resolve(file.storedPath));
});

// 某个文件对应的 3D 网格：优先本文件分析产出的 ROI → （档案只有一个扫描文件时）标注编辑后的 ROI → 本文件原始上传
// 注意：ROI 产物是按“档案 / 分析任务”落盘的，必须按本文件的任务去找，否则会把别的文件的模型显示出来
async function resolveMeshPathForFile(file: { id: string; caseId: string; storedPath: string }): Promise<{ path: string; source: "analysis" | "annotated" | "original" } | null> {
  const resultsRoot = process.env.AIS_RESULTS_ROOT || localPaths.results;
  try {
    const tasks = await db.analysisTask.findMany({ where: { fileId: file.id }, orderBy: { createdAt: "desc" }, select: { id: true } });
    for (const task of tasks) {
      const roi = path.join(resultsRoot, "prediction-outputs", `${file.caseId}-${task.id}`, "roi.ply");
      if (existsSync(roi)) return { path: roi, source: "analysis" };
    }
  } catch { /* 忽略查询异常，继续回退 */ }
  try {
    const caseFiles = await db.scanFile.findMany({ where: { caseId: file.caseId }, select: { id: true, originalName: true } });
    const scanFiles = caseFiles.filter((row) => String(row.originalName || "").toLowerCase().endsWith(".ply"));
    // 标注编辑后的 ROI 只按档案存放，无法区分具体文件：仅在档案只有一个扫描文件时才敢用
    if (scanFiles.length === 1 && scanFiles[0].id === file.id) {
      const labelingDir = path.join(resultsRoot, "labeling", "cache", file.caseId, "extract_roi");
      if (existsSync(labelingDir)) {
        const edited = readdirSync(labelingDir)
          .filter((name) => name.startsWith("roi_edited_") && name.endsWith(".ply"))
          .map((name) => path.join(labelingDir, name))
          .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
        if (edited.length) return { path: edited[0], source: "annotated" };
      }
    }
  } catch { /* 忽略目录异常 */ }
  if (existsSync(file.storedPath)) return { path: file.storedPath, source: "original" };
  return null;
}
router.get("/:id/mesh", async (req: any, res) => {
  const file = await db.scanFile.findUnique({ where: { id: req.params.id }, include: { case: true } });
  if (!file || !canAccessCase(req.user, file.case)) return res.status(404).json({ success: false, message: "File not found." });
  const resolved = await resolveMeshPathForFile(file);
  if (!resolved) return res.status(404).json({ success: false, message: "Mesh content missing." });
  res.setHeader("X-AIS-Mesh-Source", resolved.source);
  return res.sendFile(path.resolve(resolved.path));
});
router.delete("/:id", async (req: any, res) => {
  const file = await db.scanFile.findUnique({ where: { id: req.params.id }, include: { case: true } });
  if (!file || !canAccessCase(req.user, file.case)) return res.status(404).json({ success: false, message: "File not found." });
  await db.$transaction(async (tx) => {
    await tx.reportReview.deleteMany({ where: { report: { fileId: file.id } } });
    await tx.annotationSession.deleteMany({ where: { report: { fileId: file.id } } });
    await tx.report.deleteMany({ where: { fileId: file.id } });
    await tx.analysisTask.deleteMany({ where: { fileId: file.id } });
    await tx.scanFile.delete({ where: { id: file.id } });
  });
  const expectedRoot = path.resolve(localPaths.scans); const stored = path.resolve(file.storedPath);
  if (stored.startsWith(`${expectedRoot}${path.sep}`)) await unlink(stored).catch(() => undefined);
  await audit(req.user.id, "delete", "ScanFile", file.id);
  res.json({ success: true });
});

export default router;
