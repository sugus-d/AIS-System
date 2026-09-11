import { Fragment, useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  FilePlus2,
  FileText,
  Loader2,
  Play,
  RotateCw,
  ScanLine,
  Trash2,
  X,
} from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import PlyViewer from "@/components/PlyViewer";
import { confirmDialog } from "@/components/ConfirmDialog";
import { toast } from "sonner";
import api from "@/lib/api";
import { translateBackendMessage } from "@/lib/backendMsg";

type FileItem = {
  id: string;
  name: string;
  size: number;
  uploadTime?: string;
  scanTime?: string;
  status?: string;
  taskStatus?: string;
  taskError?: string;
  department?: string;
  doctor?: string;
  kind: "scan" | "xray";
};
type XrayItem = {
  id: string;
  name: string;
  uploadTime?: string;
  scanTime?: string;
};
type Report = {
  id: string;
  fileId: string;
  cobbAngle?: number;
  cobb?: number;
  aisLevel?: string;
  completeTime?: string;
  createdAt?: string;
  status?: string;
  department?: string;
  screeningDate?: string;
  doctor?: string;
  remarks?: string;
};
type Detail = {
  id: string;
  caseNumber: string;
  patientName: string;
  gender: string;
  age: number;
  birthday: string;
  height: number;
  weight: number;
  idNumber: string;
  phone: string;
  medicalHistory: string;
  files: FileItem[];
};

const shortDate = (v?: string) => {
  if (!v) return "--";
  const d = new Date(v);
  return Number.isNaN(d.getTime())
    ? v
    : `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
};
const screeningDate = (r: Report, f: FileItem) =>
  shortDate(r.screeningDate || f.scanTime || f.uploadTime);

// 后端返回的 aisLevel（简体中文数据值）→ 翻译 key；未命中则原样返回
const aisLevelKey = (level?: string): string => {
  const map: Record<string, string> = {
    "正常": "enums.severityNegative",
    "轻度": "enums.severityMild",
    "中度": "enums.severityModerate",
    "重度": "enums.severitySevere",
    "未分级": "enums.severityUnclassified",
  };
  return map[level || ""] || "";
};

export default function CaseDetail() {
  const navigate = useNavigate();
  const { caseId } = useParams();
  const { t } = useTranslation();
  const isAdmin =
    sessionStorage.getItem("user_role") === "admin" ||
    sessionStorage.getItem("user_role") === "system_admin";
  const [detail, setDetail] = useState<Detail | null>(null);
  const [grouped, setGrouped] = useState<Record<string, Report[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadType, setUploadType] = useState<"scan" | "xray">("scan");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  // 行内 3D 展开（默认收起，点击「查看 3D」再展开）
  const [expandedScanFileId, setExpandedScanFileId] = useState<string | null>(null);
  // 受检者库 X 光放大预览
  const [previewXray, setPreviewXray] = useState<XrayItem | null>(null);
  // 正在删除的扫描文件（筛查报告）id
  const [deletingFile, setDeletingFile] = useState<string | null>(null);

  const load = async (silent = false) => {
    if (!caseId) return;
    try {
      if (!silent) setLoading(true);
      const [cr, rr] = await Promise.all([
        api.getCase(caseId),
        api.getReports({ caseId, pageSize: 100 }),
      ]);
      const c = cr.case || cr;
      const files = (c.files || []).map((f: any) => {
        const name = f.originalName || f.fileName || f.name || f.id;
        const kind: "scan" | "xray" = String(name).toLowerCase().endsWith(".ply") ? "scan" : "xray";
        return {
          id: f.id,
          name,
          size: Number(f.sizeBytes || f.fileSize || f.size || 0),
          uploadTime: f.createdAt || f.uploadTime,
          scanTime: f.scanTime,
          status: f.status,
          taskStatus: f.tasks?.[0]?.status,
          taskError: f.tasks?.[0]?.errorMessage,
          department: f.department ?? undefined,
          doctor: f.doctor ?? undefined,
          kind,
        };
      });
      setDetail({
        id: c.id,
        caseNumber: c.caseNumber || c.id,
        patientName: c.name || c.patientName || "--",
        gender: c.gender === "female" || c.gender === "女" ? "女" : "男",
        age: c.birthDate
          ? new Date().getFullYear() - new Date(c.birthDate).getFullYear()
          : Number(c.age || 0),
        birthday: c.birthDate ? String(c.birthDate).slice(0, 10) : "--",
        height: Number(c.height || 0),
        weight: Number(c.weight || 0),
        idNumber: c.idNumber || "--",
        phone: c.phone || "--",
        medicalHistory: c.medicalHistory || "",
        files,
      });
      const g: Record<string, Report[]> = {};
      (rr.list || rr.data?.list || []).forEach((r: Report) => {
        if (r.fileId) (g[r.fileId] ||= []).push(r);
      });
      Object.values(g).forEach((a) =>
        a.sort((x, y) =>
          (y.completeTime || y.createdAt || "").localeCompare(
            x.completeTime || x.createdAt || "",
          ),
        ),
      );
      setGrouped(g);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("caseDetail.loadFailed"));
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    load();
  }, [caseId]);

  // 分析由他人（如临床操作员）发起时，本页也要自动刷新，避免一直停留在“分析中”
  useEffect(() => {
    const analyzing = (detail?.files ?? []).some((f) => f.taskStatus === "pending" || f.taskStatus === "running");
    if (!analyzing) return;
    const timer = window.setInterval(() => { void load(true); }, 4000);
    return () => window.clearInterval(timer);
  }, [detail]);

  const files = useMemo(
    () =>
      detail
        ? [...detail.files].sort((a, b) =>
            (b.uploadTime || "").localeCompare(a.uploadTime || ""),
          )
        : [],
    [detail],
  );
  const scanFiles = useMemo(() => files.filter((f) => f.kind === "scan"), [files]);
  const xrayFiles = useMemo(() => files.filter((f) => f.kind === "xray"), [files]);
  const analyze = async (f: FileItem) => {
    if (!caseId || busy || f.status === "deleted") return;
    try {
      setBusy(f.id);
      const task = await api.analyzeSingle(caseId, f.id);
      for (let attempt = 0; attempt < 150; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
        const current = await api.getTask(task.id);
        if (current.status === "success") break;
        if (current.status === "failed" || current.status === "cancelled") {
          throw new Error(current.failureReason || t("caseDetail.analyzeFailed"));
        }
      }
      await load();
    } catch (e) {
      toast.error(translateBackendMessage(e instanceof Error ? e.message : t("caseDetail.analyzeFailed")));
    } finally {
      setBusy(null);
    }
  };
  const openUpload = (type: "scan" | "xray") => {
    setUploadType(type);
    setUploadError("");
    setSelectedFiles([]);
    setUploadOpen(true);
  };
  const uploadReport = async () => {
    if (!caseId || selectedFiles.length === 0) return;
    // 校验所有待上传文件
    for (const file of selectedFiles) {
      const name = file.name.toLowerCase();
      if (uploadType === "scan" && !name.endsWith(".ply")) {
        setUploadError(t("caseDetail.plyRequired"));
        return;
      }
      if (uploadType === "xray" && !/\.(jpe?g|png|webp)$/.test(name)) {
        setUploadError(t("caseDetail.xrayFormatRequired"));
        return;
      }
    }
    try {
      setUploading(true);
      setUploadError("");
      // X 光支持一次选择多张，逐张上传
      for (const file of selectedFiles) {
        await api.uploadFile({ caseId, file, scanTime: new Date().toISOString() });
      }
      setSelectedFiles([]);
      setUploadOpen(false);
      await load();
    } catch (caught) {
      setUploadError(caught instanceof Error ? caught.message : t("caseDetail.uploadFailed"));
    } finally {
      setUploading(false);
    }
  };
  const toggleScanRow = (f: FileItem) => {
    setExpandedScanFileId((prev) => (prev === f.id ? null : f.id));
  };
  const deleteXray = async (x: XrayItem) => {
    if (!(await confirmDialog(t("caseDetail.xrayDeleteConfirm", { name: x.name }), { destructive: true }))) return;
    try {
      await api.deleteFile(x.id);
      await load();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : t("caseDetail.deleteFailed"));
    }
  };
  const deleteScanReport = async (f: FileItem) => {
    if (deletingFile) return;
    if (!(await confirmDialog(t("caseDetail.deleteScanConfirm", { name: f.name }), { destructive: true }))) return;
    try {
      setDeletingFile(f.id);
      await api.deleteFile(f.id);
      await load();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : t("caseDetail.deleteFailed"));
    } finally {
      setDeletingFile(null);
    }
  };

  if (loading)
    return (
      <Shell isAdmin={isAdmin}>
        <p className="py-12 text-center">{t("common.loading")}</p>
      </Shell>
    );
  if (!detail || error)
    return (
      <Shell isAdmin={isAdmin}>
        <div className="py-12 text-center">
          <p className="mb-4 text-[color:var(--color-error)]">
            {error || t("caseDetail.notFound")}
          </p>
          <button className="btn-secondary" onClick={() => navigate("/cases")}>
            {t("common.backToList")}
          </button>
        </div>
      </Shell>
    );
  return (
    <Shell isAdmin={isAdmin}>
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-8">
        <div>
          <p className="text-sm text-muted-foreground uppercase tracking-wider text-[color:var(--color-primary)] mb-2">
            {t("caseDetail.eyebrow")}
          </p>
          <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("caseDetail.title")}</h1>
          <p className="text-sm mt-2">
            {detail.caseNumber} · {detail.patientName}
          </p>
        </div>
        <button className="btn-secondary" onClick={() => navigate("/cases")}>
          {t("common.backToList")}
        </button>
      </div>

      {/* ── 基本信息 与 X 光结果 并列 ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
        <section className="card-base p-6 md:p-7">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-semibold text-foreground">{t("caseDetail.basicInfo")}</h2>
            <button
              className="btn-text"
              onClick={() => navigate(`/case-record?caseId=${caseId}&mode=edit`)}
            >
              {t("caseDetail.editProfile")}
            </button>
          </div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-4">
            <Info label={t("caseDetail.name")} value={detail.patientName} />
            <Info label={t("caseDetail.gender")} value={detail.gender === "女" ? t("enums.genderFemale") : t("enums.genderMale")} />
            <Info label={t("caseDetail.age")} value={t("caseDetail.ageValue", { value: detail.age })} />
            <Info label={t("caseDetail.birthday")} value={detail.birthday} />
            <Info label={t("caseDetail.height")} value={t("caseDetail.heightValue", { value: detail.height })} />
            <Info label={t("caseDetail.weight")} value={t("caseDetail.weightValue", { value: detail.weight })} />
            <Info label={t("caseDetail.idNumber")} value={detail.idNumber} />
            <Info label={t("caseDetail.phone")} value={detail.phone} />
          </div>
          {detail.medicalHistory && (
            <div className="mt-5 pt-5 border-t">
              <Info label={t("caseDetail.medicalHistory")} value={detail.medicalHistory} />
            </div>
          )}
        </section>

        <section className="card-base p-6 md:p-7">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-semibold text-foreground">{t("caseDetail.xray")}</h2>
            <button className="btn-secondary inline-flex items-center justify-center gap-2" onClick={() => openUpload("xray")}>
              <FilePlus2 size={16} />{t("caseDetail.uploadXray")}
            </button>
          </div>
          {xrayFiles.length === 0 ? (
            <div className="py-14 text-center text-sm text-slate-500">
              {t("caseDetail.xrayEmpty")}
            </div>
          ) : (
            <div className="grid grid-cols-2 items-start gap-3 sm:grid-cols-3">
              {xrayFiles.map((x) => (
                <div key={x.id} className="group relative overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
                  <button
                    className="block h-56 w-full cursor-pointer"
                    onClick={() => setPreviewXray(x)}
                    title={x.name}
                  >
                    <img
                      src={api.fileDownloadUrl(x.id)}
                      alt={x.name}
                      loading="lazy"
                      className="h-full w-full object-contain"
                    />
                    <span className="absolute bottom-0 inset-x-0 bg-black/55 px-1.5 py-0.5 text-[10px] text-white truncate text-left">
                      {shortDate(x.uploadTime)}
                    </span>
                  </button>
                  <button
                    className="absolute top-1 right-1 z-10 rounded-full bg-red-600 p-1 text-white opacity-0 shadow transition-opacity hover:bg-red-700 group-hover:opacity-100"
                    aria-label={t("caseDetail.xray")}
                    title={t("caseDetail.xray")}
                    onClick={() => void deleteXray(x)}
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {/* ── 筛查报告：扫描 ↔ 报告一一绑定；上传扫描即新建报告 ── */}
      <section>
        <div className="mb-4 flex flex-col gap-3 md:mb-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-foreground">{t("caseDetail.reportTitle")}</h2>
            <p className="text-sm text-muted-foreground mt-1">
              {t("caseDetail.reportHint")}
            </p>
          </div>
          <button className="btn-primary inline-flex items-center justify-center gap-2 self-start sm:self-auto" onClick={() => openUpload("scan")}>
            <FilePlus2 size={17} />{t("caseDetail.newReport")}
          </button>
        </div>
        {scanFiles.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-[color:var(--color-border)] bg-white/70 px-6 py-16 text-center">
            <ScanLine size={30} strokeWidth={1.5} className="mb-3 text-slate-300" />
            <p className="text-sm text-slate-500">
              {t("caseDetail.reportEmpty")}
            </p>
          </div>
        ) : (
          <div className="card-base overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[780px] text-sm">
                <thead>
                  <tr className="border-b text-left text-sm text-muted-foreground text-[color:var(--color-text-tertiary)]">
                    <th className="px-5 py-3.5 font-medium">{t("caseDetail.colTime")}</th>
                    <th className="px-5 py-3.5 font-medium">
                      {t("caseDetail.colDept")} / {t("caseDetail.colDoctor")}
                    </th>
                    <th className="px-5 py-3.5 font-medium">{t("caseDetail.colResult")}</th>
                    <th className="px-5 py-3.5 font-medium">{t("caseDetail.colStatus")}</th>
                    <th className="px-5 py-3.5 text-right font-medium">{t("caseDetail.colOps")}</th>
                  </tr>
                </thead>
                <tbody>
                  {scanFiles.map((f) => {
              const report = (grouped[f.id] || [])[0] || null;
              const row = report || ({} as Report);
              const scanExpanded = expandedScanFileId === f.id;
              // 分析中：当前行正在跑任务，或任务处于队列 pending / running
              const analyzing =
                busy === f.id ||
                f.taskStatus === "pending" ||
                f.taskStatus === "running";
              // Cobb 角四舍五入取整显示，并按严重程度着色：Normal <10 绿、Mild 10~20 黄、Moderate 20~40 橙、Severe ≥40 红
              const cobbValue = report ? Number(report.cobbAngle ?? report.cobb) : null;
              const cobbText =
                cobbValue != null && Number.isFinite(cobbValue)
                  ? Math.round(cobbValue)
                  : "-";
              let cobbColor = "text-slate-600";
              if (cobbValue != null && Number.isFinite(cobbValue)) {
                if (cobbValue < 10) {
                  cobbColor = "text-green-600";
                } else if (cobbValue < 20) {
                  cobbColor = "text-yellow-600";
                } else if (cobbValue < 40) {
                  cobbColor = "text-orange-600";
                } else {
                  cobbColor = "text-red-600";
                }
              }
              const levelText = report?.aisLevel
                ? aisLevelKey(report.aisLevel)
                  ? t(aisLevelKey(report.aisLevel))
                  : report.aisLevel
                : "";
              return (
                <Fragment key={f.id}>
                  <tr className={`border-b align-middle ${scanExpanded ? "bg-slate-50/60" : "hover:bg-slate-50/40"}`}>
                    <td className="px-5 py-4">
                      <p className="whitespace-nowrap font-semibold text-[color:var(--color-text-primary)]">{screeningDate(row, f)}</p>
                    </td>
                    <td className="px-5 py-4">
                      <p className="whitespace-nowrap">{(row.department || f.department || "--")} / {(row.doctor || f.doctor || "--")}</p>
                    </td>
                    <td className="px-5 py-4">
                      {report ? (
                        <span className={`whitespace-nowrap font-bold ${cobbColor}`}>
                          {t("caseDetail.cobbResult", {
                            value: cobbText,
                            level: levelText || t("caseDetail.colResult"),
                          })}
                        </span>
                      ) : (
                        <span className="text-[color:var(--color-text-tertiary)]">--</span>
                      )}
                    </td>
                    <td className="px-5 py-4">
                      <ReportStatus
                        t={t}
                        status={
                          analyzing
                            ? "analyzing"
                            : report
                              ? report.status || "under_review"
                              : "pending_analysis"
                        }
                      />
                    </td>
                    <td className="px-5 py-4">
                      <div className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1">
                        <button
                          className="btn-text inline-flex items-center gap-1.5 whitespace-nowrap"
                          onClick={() => toggleScanRow(f)}
                          aria-expanded={scanExpanded}
                        >
                          {scanExpanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                          {scanExpanded ? t("caseDetail.collapse3D") : t("caseDetail.view3D")}
                        </button>
                        {report && (
                          <button
                            className="btn-text inline-flex items-center gap-1.5 whitespace-nowrap"
                            onClick={() =>
                              navigate(
                                `/analysis-report?caseId=${caseId}&fileId=${f.id}&reportId=${report.id}`,
                              )
                            }
                          >
                            <FileText size={15} />
                            {t("caseDetail.viewReport")}
                          </button>
                        )}
                        <button
                          className={`inline-flex items-center justify-center gap-2 whitespace-nowrap ${report ? "btn-secondary" : "btn-primary"}`}
                          onClick={() => analyze(f)}
                          disabled={f.status === "deleted" || analyzing}
                        >
                          {analyzing ? (
                            <Loader2 size={15} className="animate-spin" />
                          ) : report ? (
                            <RotateCw size={15} />
                          ) : (
                            <Play size={15} />
                          )}
                          {analyzing
                            ? t("caseDetail.analyzing")
                            : report
                              ? t("caseDetail.reanalyze")
                              : t("caseDetail.startAnalyze")}
                        </button>
                        <button
                          className="btn-text inline-flex items-center gap-1.5 whitespace-nowrap text-[color:var(--color-error)] disabled:opacity-50 disabled:cursor-not-allowed"
                          onClick={() => void deleteScanReport(f)}
                          disabled={deletingFile === f.id || analyzing || f.status === "deleted"}
                          title={t("caseDetail.delete")}
                        >
                          {deletingFile === f.id ? (
                            <Loader2 size={15} className="animate-spin" />
                          ) : (
                            <Trash2 size={15} />
                          )}
                          {t("caseDetail.delete")}
                        </button>
                      </div>
                    </td>
                  </tr>
                  {!analyzing && f.taskStatus === "failed" && f.taskError && (
                    <tr className="border-b bg-red-50/60">
                      <td colSpan={5} className="px-5 py-2.5 text-sm text-[color:var(--color-error)]">
                        {translateBackendMessage(f.taskError)}
                      </td>
                    </tr>
                  )}
                  {report?.remarks && (
                    <tr className="border-b bg-slate-50/70">
                      <td colSpan={5} className="px-5 py-2.5">
                        <span className="flex items-start gap-1.5 text-sm text-muted-foreground text-[color:var(--color-text-secondary)]">
                          <span className="whitespace-nowrap font-semibold">{t("caseDetail.colRemark")}</span>
                          <span className="min-w-0">{report.remarks}</span>
                        </span>
                      </td>
                    </tr>
                  )}
                  {scanExpanded && (
                    <tr className="border-b">
                      <td colSpan={5} className="p-0">
                        <div className="bg-[#0d1117] p-4">
                          <div className="mb-2 flex flex-wrap items-center justify-between gap-x-6 gap-y-1">
                            <span className="max-w-[55%] truncate text-xs font-medium text-white/90" title={f.name}>
                              {f.name}
                            </span>
                            <span className="text-xs text-white/60">
                              {t("caseDetail.uploadTimeHint", { time: shortDate(f.uploadTime) })}
                            </span>
                          </div>
                          <div className="h-[480px]">
                            <PlyViewer key={f.id} fileId={f.id} />
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* ── 上传弹窗（扫描 PLY / X 光图片）── */}
      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4" role="dialog" aria-modal="true" aria-labelledby="new-report-title">
          <div className="card-base w-full max-w-lg p-6 shadow-xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id="new-report-title" className="text-lg font-semibold text-foreground">{uploadType === "scan" ? t("caseDetail.uploadScanTitle") : t("caseDetail.uploadXrayTitle")}</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  {uploadType === "scan" ? t("caseDetail.uploadScanHint") : t("caseDetail.uploadXrayHint")}
                </p>
              </div>
              <button className="btn-text p-1" aria-label={t("common.close")} onClick={() => { if (!uploading) { setSelectedFiles([]); setUploadError(""); setUploadOpen(false); } }} disabled={uploading}><X size={18} /></button>
            </div>
            <div className="mt-6">
              <label className="block text-sm font-semibold mb-2" htmlFor="report-file">{uploadType === "scan" ? t("caseDetail.scanFileLabel") : t("caseDetail.xrayFileLabel")}</label>
              <input id="report-file" className="input-base file:mr-4 file:border-0 file:bg-blue-50 file:px-3 file:py-1 file:text-[color:var(--color-primary)] file:font-semibold" type="file" accept={uploadType === "scan" ? ".ply" : ".jpg,.jpeg,.png,.webp"} multiple={uploadType === "xray"} onChange={(event) => { setUploadError(""); setSelectedFiles(event.target.files ? Array.from(event.target.files) : []); }} disabled={uploading} />
              {selectedFiles.length > 0 && <p className="text-sm text-muted-foreground mt-2">{t("caseDetail.selectedFiles", { count: selectedFiles.length, names: selectedFiles.map((f) => f.name).join("、") })}</p>}
              {uploadError && <p className="text-sm text-muted-foreground text-[color:var(--color-error)] mt-2">{uploadError}</p>}
            </div>
            <div className="mt-6 flex justify-end gap-3">
              <button className="btn-secondary" onClick={() => { setSelectedFiles([]); setUploadError(""); setUploadOpen(false); }} disabled={uploading}>{t("common.cancel")}</button>
              <button className="btn-primary" onClick={() => void uploadReport()} disabled={selectedFiles.length === 0 || uploading}>{uploading ? t("common.uploading") : selectedFiles.length > 1 ? t("caseDetail.uploadBtnCount", { count: selectedFiles.length }) : t("common.upload")}</button>
            </div>
          </div>
        </div>
      )}

      {/* ── X 光放大预览 ── */}
      {previewXray && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-6" role="dialog" aria-modal="true" onClick={() => setPreviewXray(null)}>
          <button
            className="absolute top-4 right-4 z-20 rounded-full bg-white text-slate-800 p-2 shadow hover:bg-slate-200"
            aria-label={t("common.close")}
            onClick={() => setPreviewXray(null)}
          >
            <X size={18} />
          </button>
          <div className="relative max-w-4xl w-full" onClick={(e) => e.stopPropagation()}>
            <img src={api.fileDownloadUrl(previewXray.id)} alt={previewXray.name} className="w-full max-h-[82vh] object-contain rounded-lg" />
            <p className="mt-2 text-center text-xs text-white/80">{previewXray.name} · {shortDate(previewXray.uploadTime)}</p>
          </div>
        </div>
      )}
    </Shell>
  );
}
function Shell({
  isAdmin,
  children,
}: {
  isAdmin: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header">
        <Header isAdmin={isAdmin} />
      </div>
      <div className="layout-content">
        <div className="content-wrapper">{children}</div>
      </div>
    </div>
  );
}
function Info({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-sm text-muted-foreground text-[color:var(--color-text-tertiary)] mb-1">
        {label}
      </p>
      <p className="text-sm font-semibold">{value || "--"}</p>
    </div>
  );
}
function ReportStatus({ status, t }: { status: string; t: (key: string) => string }) {
  const map: Record<string, string> = {
    analyzing: "enums.statusAnalyzing",
    under_review: "enums.statusUnderReview",
    approved: "enums.statusApproved",
    pending_analysis: "enums.statusPendingAnalysis",
    pending_upload: "enums.statusPendingUpload",
    review_returned: "enums.statusReviewReturned",
  };
  const cls: Record<string, string> = {
    analyzing: "tag-warning",
    under_review: "tag-warning",
    approved: "tag-success",
    pending_analysis: "tag-error",
    pending_upload: "tag-error",
    review_returned: "tag-error",
  };
  const key = map[status] || "enums.statusPendingAnalysis";
  return <span className={cls[status] || "tag-error"}>{t(key)}</span>;
}
