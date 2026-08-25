import path from "node:path";
import { db, audit } from "./database";
import { predict, severityZh } from "./algorithm";

const taskView = (task: any) => ({ ...task, submitTime: task.createdAt, endTime: task.finishedAt, failureReason: task.errorMessage, relatedData: { caseId: task.caseId, fileId: task.fileId } });
const active = new Set<string>();
let queueTail: Promise<void> = Promise.resolve();

/** One local algorithm worker protects GPU/Open3D resources and preserves batch order. */
export function enqueueAnalysisTask(taskId: string) {
  if (active.has(taskId)) return queueTail;
  active.add(taskId);
  queueTail = queueTail.catch(() => undefined).then(async () => {
    try { await runAnalysisTask(taskId); } finally { active.delete(taskId); }
  });
  return queueTail;
}
export async function enqueueAnalysisTasksSequentially(taskIds: string[]) {
  for (const taskId of taskIds) await enqueueAnalysisTask(taskId);
}
export async function runAnalysisTask(taskId: string) {
  const task = await db.analysisTask.findUnique({ where: { id: taskId }, include: { file: { include: { case: true } } } });
  if (!task || task.status === "cancelled") return;
  await db.analysisTask.update({ where: { id: taskId }, data: { status: "running", progress: 10, startedAt: new Date(), errorMessage: null } });
  try {
    const subject = task.file.case;
    let taskInput: any = {}; try { taskInput = task.resultJson ? JSON.parse(task.resultJson) : {}; } catch { /* task has no manual input */ }
    const inputPath = taskInput.manualRoiPath || task.file.storedPath;
    const result = await predict(inputPath, `${subject.id}-${task.id}`, { gender: /female|女/i.test(subject.gender) ? "Female" : "Male", height_cm: subject.heightCm, weight_kg: subject.weightKg }, taskInput.landmarks);
    const current = await db.analysisTask.findUnique({ where: { id: taskId } });
    if (!current || current.status === "cancelled") return;
    const report = await db.$transaction(async (tx) => {
      const count = await tx.report.count({ where: { caseId: subject.id } });
      const outputRoot = process.env.AIS_RESULTS_ROOT ? path.join(process.env.AIS_RESULTS_ROOT, "prediction-outputs") : path.join(process.env.AIS_CORE_ROOT || path.join(process.cwd(), "AIS_core_algo"), "prediction", "outputs");
      const report = await tx.report.create({ data: { caseId: subject.id, fileId: task.fileId, taskId, version: count + 1, cobbAngle: Number(result.cobb), severity: String(result.severity), modelId: String(result.model_id || "v1.0.0"), resultJson: JSON.stringify(result), artifactDirectory: path.join(outputRoot, String(result.subject_id)) } });
      await tx.reportReview.create({ data: { reportId: report.id } });
      return report;
    });
    await db.analysisTask.update({ where: { id: taskId }, data: { status: "success", progress: 100, resultJson: JSON.stringify({ reportId: report.id, cobb: result.cobb, severity: result.severity }), finishedAt: new Date() } });
    await db.scanFile.update({ where: { id: task.fileId }, data: { status: "analyzed" } });
    await db.case.update({ where: { id: subject.id }, data: { status: "under_review" } });
    await audit(task.submittedById, "analysis_completed", "Report", report.id, { taskId, cobb: result.cobb, severity: severityZh(String(result.severity)) });
  } catch (error) { await db.analysisTask.update({ where: { id: taskId }, data: { status: "failed", progress: 100, errorMessage: error instanceof Error ? error.message : "AIS prediction failed.", finishedAt: new Date() } }); }
}
/** Resume work left queued or running when the local application stopped. */
export async function resumeIncompleteAnalysisTasks() {
  const tasks = await db.analysisTask.findMany({ where: { status: { in: ["pending", "running"] } }, select: { id: true }, orderBy: { createdAt: "asc" } });
  for (const task of tasks) await enqueueAnalysisTask(task.id);
  return tasks.length;
}
export { taskView };
