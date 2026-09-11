import { Router } from "express";
import { db } from "../services/database";
import { canAccessCase } from "../middleware/access";
import { exportLabels, resolveExportLang, severityLabel } from "../services/export-labels";

const router = Router();
const severity = [{ name: "Normal", color: "#22c55e" }, { name: "Mild", color: "#eab308" }, { name: "Moderate", color: "#f97316" }, { name: "Severe", color: "#ef4444" }];

// 可见病例（含报告/任务/文件身份），供统计使用
async function scoped(user: any) {
  const cases = await db.case.findMany({
    include: {
      files: true,
      reports: {
        include: {
          task: true,
          file: { select: { doctor: true, department: true } },
        },
      },
      owner: true,
    },
  });
  return cases.filter((item) => canAccessCase(user, item));
}

// 报告身份：优先分析时写入 clinician（提交者身份快照），回退到文件/受检者
function reportIdentity(report: any) {
  let clinician: any = {};
  try {
    clinician = report.resultJson ? (JSON.parse(report.resultJson)?.clinician ?? {}) : {};
  } catch { /* 忽略非法 JSON */ }
  return {
    doctor: String(clinician.doctor || report.file?.doctor || ""),
    department: String(clinician.department || report.file?.department || ""),
  };
}

// 分析时间：最近一次任务完成时间（重分析原地更新后 taskId 指向最新任务），无任务回退报告创建时间
function analysisTime(report: any) {
  const finished = report.task?.finishedAt ? new Date(report.task.finishedAt).getTime() : NaN;
  return Number.isFinite(finished) ? finished : new Date(report.createdAt).getTime();
}

// 把可见病例整理为 (caseMap, allReports)，便于各接口按需过滤
function flatten(cases: any[]) {
  const caseById = new Map(cases.map((c) => [c.id, c]));
  const reports = cases.flatMap((c) => c.reports);
  return { caseById, reports };
}

// 筛选：时间（按分析时间）/ 机构 / 科室 / 人员（医生）
function filterReports(reports: any[], caseById: Map<string, any>, q: any) {
  const dateFrom = typeof q.dateFrom === "string" && q.dateFrom ? new Date(`${q.dateFrom}T00:00:00`).getTime() : null;
  const dateTo = typeof q.dateTo === "string" && q.dateTo ? new Date(`${q.dateTo}T23:59:59.999`).getTime() : null;
  const institutionId = typeof q.institutionId === "string" && q.institutionId ? q.institutionId : null;
  const department = typeof q.department === "string" && q.department ? q.department : null;
  const doctor = typeof q.doctor === "string" && q.doctor ? q.doctor : null;
  return reports.filter((report) => {
    const caseItem = caseById.get(report.caseId) || {};
    if (institutionId && String(caseItem.institutionId || "") !== institutionId) return false;
    const time = analysisTime(report);
    if (dateFrom !== null && time < dateFrom) return false;
    if (dateTo !== null && time > dateTo) return false;
    const { doctor: d, department: dep } = reportIdentity(report);
    if (doctor && d !== doctor) return false;
    if (department && (dep || "未分配") !== department) return false;
    return true;
  });
}

// 每个受检者取「分析时间最新」的一份报告（AIS 分级按人，用最新报告结果）
function latestPerCase(reports: any[]) {
  const best = new Map<string, any>();
  for (const report of reports) {
    const prev = best.get(report.caseId);
    if (!prev || analysisTime(report) > analysisTime(prev)) best.set(report.caseId, report);
  }
  return [...best.values()];
}

function hasReportLevelFilter(q: any) {
  return Boolean((typeof q.department === "string" && q.department) || (typeof q.doctor === "string" && q.doctor) || (typeof q.dateFrom === "string" && q.dateFrom) || (typeof q.dateTo === "string" && q.dateTo));
}

router.get("/overview", async (req: any, res) => {
  const cases = await scoped(req.user);
  const institutionId = typeof req.query.institutionId === "string" && req.query.institutionId ? req.query.institutionId : null;
  const baseCases = institutionId ? cases.filter((c) => String(c.institutionId || "") === institutionId) : cases;
  const ids = baseCases.map((item) => item.id);
  const tasks = ids.length ? await db.analysisTask.findMany({ where: { caseId: { in: ids } } }) : [];
  const { caseById, reports: allReports } = flatten(baseCases);
  const reports = filterReports(allReports, caseById, req.query);
  const distinctCases = new Set(reports.map((r) => r.caseId)).size;
  const count = (name: string) => reports.filter((report) => report.severity === name).length;
  const caseTotal = hasReportLevelFilter(req.query) ? distinctCases : baseCases.length;

  // ① 受检者状态（当前档案状态快照，口径与“能否发起分析”一致）
  const inFlightCases = new Set(tasks.filter((task) => task.status === "pending" || task.status === "running").map((task) => task.caseId));
  const hasBasicInfo = (item: any) => Boolean(item.gender && item.heightCm > 0 && item.weightKg > 0);
  const hasScanFile = (item: any) => (item.files || []).some((file: any) => String(file.originalName || "").toLowerCase().endsWith(".ply"));
  let analyzedCases = 0; let analyzingCases = 0; let pendingCases = 0; let insufficientCases = 0;
  for (const item of baseCases) {
    if (item.reports.length > 0) { analyzedCases += 1; continue; }
    if (inFlightCases.has(item.id)) { analyzingCases += 1; continue; }
    if (hasBasicInfo(item) && hasScanFile(item)) pendingCases += 1; else insufficientCases += 1;
  }

  // ② 分析结果：以每位受检者的最新报告为准（与 AIS 分级口径一致）
  const latest = latestPerCase(reports);
  const latestCount = latest.length;
  const avg = latestCount ? latest.reduce((total, report) => total + report.cobbAngle, 0) / latestCount : 0;
  const rate = (predicate: (report: any) => boolean) => (latestCount ? (latest.filter(predicate).length * 100 / latestCount).toFixed(1) : "0");

  // ③ 分析运行：当前筛选范围内的报告 / 成功 / 成功率 / 待审核
  const reportSuccess = reports.filter((report) => !report.task || report.task.status === "success").length;
  const pendingReview = reports.filter((report) => !report.review?.status || report.review.status === "under_review").length;

  res.json({
    success: true,
    data: {
      cases: { total: caseTotal, male: baseCases.filter((item) => /male|男/i.test(item.gender)).length, female: baseCases.filter((item) => /female|女/i.test(item.gender)).length },
      caseStatus: { total: baseCases.length, analyzed: analyzedCases, analyzing: analyzingCases, pending: pendingCases, insufficient: insufficientCases },
      files: { total: baseCases.reduce((total, item) => total + item.files.length, 0) },
      reports: { total: reports.length, completed: reports.filter((report) => report.annotationStatus === "approved").length, success: reportSuccess, pendingReview, successRate: reports.length ? (reportSuccess * 100 / reports.length).toFixed(1) : "0" },
      aisDistribution: { normal: count("Normal"), mild: count("Mild"), moderate: count("Moderate"), severe: count("Severe") },
      tasks: { total: tasks.length, success: tasks.filter((task) => task.status === "success").length, failed: tasks.filter((task) => task.status === "failed").length, successRate: tasks.length ? (tasks.filter((task) => task.status === "success").length * 100 / tasks.length).toFixed(1) : "0" },
      metrics: {
        avgCobbAngle: avg.toFixed(1),
        positiveRate: rate((report) => report.severity !== "Normal"),
        moderateOrAboveRate: rate((report) => report.severity === "Moderate" || report.severity === "Severe"),
      },
    },
  });
});

router.get("/cases-distribution", async (req: any, res) => {
  const cases = await scoped(req.user);
  const { caseById, reports: allReports } = flatten(cases);
  const reports = filterReports(allReports, caseById, req.query);
  const field = req.query.type === "doctor" ? "doctor" : req.query.type === "gender" ? "gender" : "department";
  const distribution: Record<string, number> = {};
  if (field === "doctor") {
    for (const report of reports) {
      const { doctor } = reportIdentity(report);
      const key = doctor || "未分配";
      distribution[key] = (distribution[key] || 0) + 1;
    }
  } else {
    for (const item of cases) {
      let raw = item[field];
      if (!raw && field === "department") raw = item.owner?.department;
      const value = String(raw || "未分配");
      distribution[value] = (distribution[value] || 0) + 1;
    }
  }
  res.json({ success: true, data: Object.entries(distribution).map(([name, value]) => ({ name, value })) });
});

// AIS 分级：按「人」统计 —— 每受检者用其最新一份报告的分级（Req 5）
router.get("/ais-distribution", async (req: any, res) => {
  const cases = await scoped(req.user);
  const institutionId = typeof req.query.institutionId === "string" && req.query.institutionId ? req.query.institutionId : null;
  const baseCases = institutionId ? cases.filter((c) => String(c.institutionId || "") === institutionId) : cases;
  const { caseById, reports: allReports } = flatten(baseCases);
  const reports = filterReports(allReports, caseById, req.query);
  const latest = latestPerCase(reports);
  res.json({ success: true, data: severity.map((item) => ({ ...item, value: latest.filter((report) => report.severity === item.name).length })) });
});

// 医生分析统计：每个医生做了多少患者的报告（患者有该医生的报告即记 1，多医生各自 +1）(Req 7)
router.get("/doctor-distribution", async (req: any, res) => {
  const cases = await scoped(req.user);
  const institutionId = typeof req.query.institutionId === "string" && req.query.institutionId ? req.query.institutionId : null;
  const baseCases = institutionId ? cases.filter((c) => String(c.institutionId || "") === institutionId) : cases;
  const { caseById, reports: allReports } = flatten(baseCases);
  const reports = filterReports(allReports, caseById, req.query);
  const pairs = new Set<string>(); // `${doctor}||${caseId}`
  const meta = new Map<string, { name: string; department: string }>();
  for (const report of reports) {
    const { doctor, department } = reportIdentity(report);
    if (!doctor) continue;
    pairs.add(`${doctor}||${report.caseId}`);
    if (!meta.has(doctor)) meta.set(doctor, { name: doctor, department: department || "未分配" });
  }
  const counts = new Map<string, number>();
  for (const pair of pairs) {
    const doctor = pair.split("||")[0];
    counts.set(doctor, (counts.get(doctor) || 0) + 1);
  }
  const data = [...meta.values()].map((m) => ({ ...m, patientCount: counts.get(m.name) || 0 })).sort((a, b) => b.patientCount - a.patientCount);
  res.json({ success: true, data });
});

// 新增受检者趋势：随所选时间范围联动；一天/一周/一月均按天，更长范围自动按周、按月，且范围内无数据的时段补 0
router.get("/time-series", async (req: any, res) => {
  const metric = req.query.metric;
  const cases = await scoped(req.user);
  const institutionId = typeof req.query.institutionId === "string" && req.query.institutionId ? req.query.institutionId : null;
  const baseCases = institutionId ? cases.filter((item) => String(item.institutionId || "") === institutionId) : cases;
  const caseIds = baseCases.map((item) => item.id);
  const rows: any[] = metric === "analyses"
    ? (caseIds.length ? await db.analysisTask.findMany({ where: { caseId: { in: caseIds } } }) : [])
    : metric === "reports" ? baseCases.flatMap((item) => item.reports) : baseCases;

  const today = new Date();
  const startOfDay = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate(), 0, 0, 0, 0);
  const endOfDay = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate(), 23, 59, 59, 999);
  const hasFrom = typeof req.query.dateFrom === "string" && req.query.dateFrom;
  const hasTo = typeof req.query.dateTo === "string" && req.query.dateTo;
  const from = hasFrom ? startOfDay(new Date(`${req.query.dateFrom}T00:00:00`)) : startOfDay(new Date(today.getFullYear(), today.getMonth(), today.getDate() - 6));
  const to = hasTo ? endOfDay(new Date(`${req.query.dateTo}T00:00:00`)) : endOfDay(today);
  const spanDays = Math.max(1, Math.round((to.getTime() - from.getTime()) / 86400000) + 1);
  const granularity: "day" | "week" | "month" = spanDays <= 31 ? "day" : spanDays <= 182 ? "week" : "month";

  const pad = (value: number) => String(value).padStart(2, "0");
  const weekStart = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate() - ((date.getDay() + 6) % 7));
  const bucketOf = (date: Date) =>
    granularity === "month" ? `${date.getFullYear()}-${pad(date.getMonth() + 1)}` :
      granularity === "week" ? `${pad(weekStart(date).getMonth() + 1)}-${pad(weekStart(date).getDate())}` :
        `${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;

  // 先铺满时间轴，保证没有数据的时段显示为 0（而不是断线）
  const labels: string[] = [];
  if (granularity === "month") {
    for (const cursor = new Date(from.getFullYear(), from.getMonth(), 1); cursor <= to; cursor.setMonth(cursor.getMonth() + 1)) labels.push(bucketOf(cursor));
  } else if (granularity === "week") {
    for (const cursor = weekStart(from); cursor <= to; cursor.setDate(cursor.getDate() + 7)) labels.push(bucketOf(cursor));
  } else {
    for (const cursor = new Date(from); cursor <= to; cursor.setDate(cursor.getDate() + 1)) labels.push(bucketOf(cursor));
  }
  const counts = new Map<string, number>(labels.map((label) => [label, 0]));
  for (const row of rows) {
    const time = new Date(row.createdAt);
    if (time < from || time > to) continue;
    const key = bucketOf(time);
    if (counts.has(key)) counts.set(key, (counts.get(key) || 0) + 1);
  }
  res.json({ success: true, data: labels.map((label) => ({ date: granularity === "month" ? label : label.slice(0, 5), value: counts.get(label) || 0 })), granularity, from: from.toISOString().slice(0, 10), to: to.toISOString().slice(0, 10) });
});

router.post("/export", async (req: any, res) => {
  const cases = await scoped(req.user);
  const { caseById, reports: allReports } = flatten(cases);
  const reports = filterReports(allReports, caseById, req.query);
  const lang = resolveExportLang(req); const labels = exportLabels(lang);
  const body = [
    [labels.metric, labels.value],
    [labels.caseCount, new Set(reports.map((r) => r.caseId)).size],
    [labels.reportCount, reports.length],
    [severityLabel("Normal", lang), reports.filter((item) => item.severity === "Normal").length],
    [severityLabel("Mild", lang), reports.filter((item) => item.severity === "Mild").length],
    [severityLabel("Moderate", lang), reports.filter((item) => item.severity === "Moderate").length],
    [severityLabel("Severe", lang), reports.filter((item) => item.severity === "Severe").length],
  ].map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\r\n");
  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.setHeader("Content-Disposition", "attachment; filename=\"AIS-statistics.csv\"");
  return res.send(`\uFEFF${body}`);
});

export default router;
