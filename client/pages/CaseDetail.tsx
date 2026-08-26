import { useEffect, useMemo, useState } from "react";
import { FilePlus2, X } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import api from "@/lib/api";

type FileItem = {
  id: string;
  name: string;
  size: number;
  uploadTime?: string;
  scanTime?: string;
  status?: string;
  taskStatus?: string;
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

export default function CaseDetail() {
  const navigate = useNavigate();
  const { caseId } = useParams();
  const isAdmin =
    localStorage.getItem("user_role") === "admin" ||
    localStorage.getItem("user_role") === "system_admin";
  const [detail, setDetail] = useState<Detail | null>(null);
  const [grouped, setGrouped] = useState<Record<string, Report[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const load = async () => {
    if (!caseId) return;
    try {
      setLoading(true);
      const [cr, rr] = await Promise.all([
        api.getCase(caseId),
        api.getReports({ caseId, pageSize: 100 }),
      ]);
      const c = cr.case || cr;
      const files = (c.files || []).map((f: any) => ({
        id: f.id,
        name: f.originalName || f.fileName || f.name || f.id,
        size: Number(f.sizeBytes || f.fileSize || f.size || 0),
        uploadTime: f.createdAt || f.uploadTime,
        scanTime: f.scanTime,
        status: f.status,
        taskStatus: f.tasks?.[0]?.status,
      }));
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
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    load();
  }, [caseId]);
  const files = useMemo(
    () =>
      detail
        ? [...detail.files].sort((a, b) =>
            (b.uploadTime || "").localeCompare(a.uploadTime || ""),
          )
        : [],
    [detail],
  );
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
          throw new Error(current.failureReason || "真实算法分析失败");
        }
      }
      await load();
    } catch (e) {
      alert(e instanceof Error ? e.message : "分析失败");
    } finally {
      setBusy(null);
    }
  };
  const uploadReport = async () => {
    if (!caseId || !selectedFile) return;
    if (!selectedFile.name.toLowerCase().endsWith(".ply")) {
      setUploadError("请选择 PLY 格式的筛查文件。");
      return;
    }
    try {
      setUploading(true);
      setUploadError("");
      await api.uploadFile({ caseId, file: selectedFile, scanTime: new Date().toISOString() });
      setSelectedFile(null);
      setUploadOpen(false);
      await load();
    } catch (caught) {
      setUploadError(caught instanceof Error ? caught.message : "文件上传失败。");
    } finally {
      setUploading(false);
    }
  };
  if (loading)
    return (
      <Shell isAdmin={isAdmin}>
        <p className="py-12 text-center">加载中...</p>
      </Shell>
    );
  if (!detail || error)
    return (
      <Shell isAdmin={isAdmin}>
        <div className="py-12 text-center">
          <p className="mb-4 text-[color:var(--color-error)]">
            {error || "病例不存在"}
          </p>
          <button className="btn-secondary" onClick={() => navigate("/cases")}>
            返回列表
          </button>
        </div>
      </Shell>
    );
  return (
    <Shell isAdmin={isAdmin}>
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-8">
        <div>
          <p className="text-helper uppercase tracking-wider text-[color:var(--color-primary)] mb-2">
            Patient profile
          </p>
          <h1 className="text-page-title">受检者详情</h1>
          <p className="text-body mt-2">
            {detail.caseNumber} · {detail.patientName}
          </p>
        </div>
        <button className="btn-secondary" onClick={() => navigate("/cases")}>
          返回列表
        </button>
      </div>
      <section className="card-base p-6 md:p-8 mb-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-card-title">基本信息</h2>
          <button
            className="btn-text"
            onClick={() => navigate(`/case-record?caseId=${caseId}&mode=edit`)}
          >
            编辑资料
          </button>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-5">
          <Info label="姓名" value={detail.patientName} />
          <Info label="性别" value={detail.gender} />
          <Info label="年龄" value={`${detail.age} 岁`} />
          <Info label="生日" value={detail.birthday} />
          <Info label="身高" value={`${detail.height} cm`} />
          <Info label="体重" value={`${detail.weight} kg`} />
          <Info label="身份证" value={detail.idNumber} />
          <Info label="电话" value={detail.phone} />
        </div>
        {detail.medicalHistory && (
          <div className="mt-6 pt-6 border-t">
            <Info label="既往病史" value={detail.medicalHistory} />
          </div>
        )}
      </section>
      <section className="card-base overflow-hidden">
        <div className="p-6 md:p-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-card-title">筛查报告</h2>
            <p className="text-helper mt-1">
              每行对应一份独立报告，状态和分析操作互不影响。
            </p>
          </div>
          <button className="btn-primary inline-flex items-center justify-center gap-2" onClick={() => { setUploadError(""); setUploadOpen(true); }}>
            <FilePlus2 size={17} />新建报告
          </button>
        </div>
        {files.length === 0 ? (
          <div className="py-16 text-center text-body text-slate-500">
            暂无筛查报告
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1300px] text-left">
              <thead>
                <tr>
                  <th className="px-5 py-3">筛查时间</th>
                  <th className="px-5 py-3">文件大小</th>
                  <th className="px-5 py-3">状态</th>
                  <th className="px-5 py-3">科室</th>
                  <th className="px-5 py-3">医生</th>
                  <th className="px-5 py-3">备注</th>
                  <th className="px-5 py-3">分析结果</th>
                  <th className="px-5 py-3">分析时间</th>
                  <th className="px-5 py-3 text-right">操作</th>
                </tr>
              </thead>
              <tbody>
                {files.flatMap((f) =>
                  (grouped[f.id] || [null]).map((r, i) => (
                    <tr
                      key={`${f.id}-${r?.id || i}`}
                      className="border-t hover:bg-blue-50/40 transition-colors"
                    >
                      <td className="px-5 py-4 whitespace-nowrap">
                        {screeningDate((r || {}) as Report, f)}
                      </td>
                      <td className="px-5 py-4 whitespace-nowrap">
                        {(f.size / 1024 / 1024).toFixed(2)} MB
                      </td>
                      <td className="px-5 py-4">
                        <ReportStatus
                          status={
                            busy === f.id ||
                            f.taskStatus === "pending" ||
                            f.taskStatus === "running"
                              ? "analyzing"
                              : r
                                ? r.status || "under_review"
                                : "pending_analysis"
                          }
                        />
                      </td>
                      <td className="px-5 py-4">{r?.department || "--"}</td>
                      <td className="px-5 py-4">{r?.doctor || "--"}</td>
                      <td
                        className="px-5 py-4 max-w-[180px] truncate"
                        title={r?.remarks || ""}
                      >
                        {r?.remarks || "--"}
                      </td>
                      <td className="px-5 py-4">
                        {r
                          ? `Cobb 角 ${r.cobbAngle ?? r.cobb ?? "-"}° · ${r.aisLevel || "未分级"}`
                          : "--"}
                      </td>
                      <td className="px-5 py-4 whitespace-nowrap">
                        {shortDate(r?.completeTime || r?.createdAt)}
                      </td>
                      <td className="px-5 py-4 text-right whitespace-nowrap">
                        {r && (
                          <button
                            className="btn-text mr-2"
                            onClick={() =>
                              navigate(
                                `/analysis-report?caseId=${caseId}&fileId=${f.id}&reportId=${r.id}`,
                              )
                            }
                          >
                            查看报告
                          </button>
                        )}
                        <button
                          className="btn-text"
                          onClick={() => analyze(f)}
                          disabled={
                            busy === f.id ||
                            f.status === "deleted" ||
                            f.taskStatus === "pending" ||
                            f.taskStatus === "running"
                          }
                        >
                          {busy === f.id ||
                          f.taskStatus === "pending" ||
                          f.taskStatus === "running"
                            ? "分析中"
                            : r
                              ? "重新分析"
                              : "开始分析"}
                        </button>
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4" role="dialog" aria-modal="true" aria-labelledby="new-report-title">
          <div className="card-base w-full max-w-lg p-6 shadow-xl">
            <div className="flex items-start justify-between gap-4">
              <div><h2 id="new-report-title" className="text-card-title">新建筛查报告</h2><p className="text-helper mt-1">选择该受检者的 PLY 筛查文件，上传后将新增一条待分析记录。</p></div>
              <button className="btn-text p-1" aria-label="关闭" onClick={() => { if (!uploading) { setSelectedFile(null); setUploadError(""); setUploadOpen(false); } }} disabled={uploading}><X size={18} /></button>
            </div>
            <div className="mt-6">
              <label className="block text-body font-semibold mb-2" htmlFor="report-file">PLY 文件</label>
              <input id="report-file" className="input-base file:mr-4 file:border-0 file:bg-blue-50 file:px-3 file:py-1 file:text-[color:var(--color-primary)] file:font-semibold" type="file" accept=".ply" onChange={(event) => { setUploadError(""); setSelectedFile(event.target.files?.[0] || null); }} disabled={uploading} />
              {selectedFile && <p className="text-helper mt-2">已选择：{selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)</p>}
              {uploadError && <p className="text-helper text-[color:var(--color-error)] mt-2">{uploadError}</p>}
            </div>
            <div className="mt-6 flex justify-end gap-3">
              <button className="btn-secondary" onClick={() => { setSelectedFile(null); setUploadError(""); setUploadOpen(false); }} disabled={uploading}>取消</button>
              <button className="btn-primary" onClick={() => void uploadReport()} disabled={!selectedFile || uploading}>{uploading ? "上传中..." : "上传并新建"}</button>
            </div>
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
      <p className="text-helper text-[color:var(--color-text-tertiary)] mb-1">
        {label}
      </p>
      <p className="text-body font-semibold">{value || "--"}</p>
    </div>
  );
}
function ReportStatus({ status }: { status: string }) {
  const map: Record<string, [string, string]> = {
    analyzing: ["分析中", "tag-warning"],
    under_review: ["审核中", "tag-warning"],
    approved: ["已通过", "tag-success"],
    pending_analysis: ["待分析", "tag-error"],
    pending_upload: ["待上传", "tag-error"],
    review_returned: ["待修改", "tag-error"],
  };
  const [label, cls] = map[status] || ["待分析", "tag-error"];
  return <span className={cls}>{label}</span>;
}
