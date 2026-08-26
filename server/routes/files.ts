import { createHash, randomUUID } from "node:crypto";
import { createReadStream, mkdirSync } from "node:fs";
import { rename, unlink } from "node:fs/promises";
import path from "node:path";
import multer from "multer";
import { Router } from "express";
import { db, localPaths, audit } from "../services/database";
import { canAccessCase, requireRoles } from "../middleware/access";

const router = Router();
const upload = multer({ dest: path.join(localPaths.dataRoot, "uploads"), limits: { fileSize: Number(process.env.AIS_MAX_UPLOAD_BYTES || 200 * 1024 * 1024) } });
const present = (file: any) => ({ id: file.id, caseId: file.caseId, fileName: file.originalName, fileSize: file.sizeBytes, scanTime: file.scanTime, uploadTime: file.createdAt, path: file.storedPath, status: file.status, sha256: file.sha256 });
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
  if (!uploaded) return res.status(400).json({ success: false, message: "A PLY file is required." });
  if (!uploaded.originalname.toLowerCase().endsWith(".ply")) { await discard(); return res.status(422).json({ success: false, message: "Only .ply files are supported." }); }
  const caseId = typeof req.body.caseId === "string" ? req.body.caseId : "";
  const subject = await db.case.findUnique({ where: { id: caseId } });
  if (!subject || !canAccessCase(req.user, subject)) { await discard(); return res.status(404).json({ success: false, message: "Case not found." }); }
  const directory = path.join(localPaths.scans, caseId); mkdirSync(directory, { recursive: true });
  const id = randomUUID(); const destination = path.join(directory, `${id}.ply`);
  try {
    const sha256 = await checksum(uploaded.path); await rename(uploaded.path, destination);
    const file = await db.scanFile.create({ data: { id, caseId, originalName: uploaded.originalname, storedPath: destination, sha256, sizeBytes: uploaded.size, scanTime: req.body.scanTime ? new Date(req.body.scanTime) : null } });
    await db.case.update({ where: { id: caseId }, data: { status: "pending_analysis" } });
    await audit(req.user.id, "upload", "ScanFile", file.id, { caseId, sha256, sizeBytes: file.sizeBytes });
    return res.status(201).json({ success: true, data: present(file) });
  } catch (error) { await discard(); return res.status(500).json({ success: false, message: error instanceof Error ? error.message : "File storage failed." }); }
});
router.delete("/:id", requireRoles("system_admin", "institution_admin", "operator"), async (req: any, res) => {
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
