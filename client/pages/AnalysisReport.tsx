import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import i18next from "@/i18n";
import { translateBackendMessage } from "@/lib/backendMsg";
import Header from "@/components/layout/Header";
import logoInline from "@/assets/logo-pdf.png?inline";
import Sidebar from "@/components/layout/Sidebar";
import { getLinkedReports } from "@/lib/workflowStore";
import api from "@/lib/api";
import { toast } from "sonner";

type Severity = "negative" | "mild" | "moderate" | "severe";

interface AnalysisResult {
    reportNumber: string;
    caseId: string;
    caseNumber: string;
    patientName: string;
    gender: "M" | "F";
    age: number;
    birthday: string;
    height: number;
    weight: number;
    department: string;
    screeningDate: string;
    doctor: string;
    fileNumber: string;
    fileName: string;
    filePath: string;
    fileSize: string;
    uploadTime: string;
    linkedDeletedRemark?: string;
    indices: Record<string, number>;
    predictedCobbAngle: number;
    severity: Severity;
    backImage: string;
    annotatedImage: string;
    heatmapImage: string;
    moireImage: string;
    normalAngleImage: string;
    diagnosis: string;
    followupSuggestion: string;
    treatment: string;
    analysisTime: string;
    reportGeneratedTime: string;
    status?: string;
    reviewComment?: string;
    annotation?: { subjectId: string; status: "bound" | "updated"; updatedAt?: string; updatedBy?: string } | null;
}

const formatDateTime = (value?: string) => {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
};

const formatDate = (value?: string) => {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
};

const escapeHtml = (value: unknown) =>
    String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");

const metricText = (value: number | undefined | null) =>
    value == null || Number.isNaN(Number(value)) ? "—" : String(Number(value));

// 生成打印用报告 HTML（字段与页面展示一致），图片以 base64 内嵌，保证打印窗口离线渲染
const buildReportPdfHtml = (r: AnalysisResult, images: Record<string, string | null>, t: (key: string, opts?: any) => string, xrayImages?: { dataUrl: string | null; uploadTime?: string }[]): string => {
    const severityZh: Record<string, string> = {
        negative: t("enums.severityNegative"),
        mild: t("enums.severityMild"),
        moderate: t("enums.severityModerate"),
        severe: t("enums.severitySevere"),
    };
    const figure = (src: string | null | undefined, title: string) =>
        src ? `<figure><img src="${src}" /><figcaption>${title}</figcaption></figure>` : "";
    const opinion = (title: string, value: string) => `<p class="opinion"><b>${title}：</b>${escapeHtml(value) || "—"}</p>`;
    // X 光影像（受检者级附加资料）与页面一致地逐张展示
    const xraySection = (xrayImages && xrayImages.length > 0)
        ? `<h2>${t("report.xrayImaging")}</h2><div class="xray-grid">${xrayImages.map((x) => figure(x.dataUrl, x.uploadTime ? `${t("report.xrayImage")} ${escapeHtml(formatDateTime(x.uploadTime))}` : t("report.xrayImage"))).join("")}</div>`
        : "";
    return `<!doctype html><html lang="${i18next.language || "zh-CN"}"><head><meta charset="utf-8"><title>${t("report.pdfDocTitle")} ${escapeHtml(r.reportNumber)}</title><style>
@page { size: A4; margin: 14mm; }
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; color: #172033; margin: 0; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: #64748b; font-size: 13px; margin-bottom: 14px; }
h2 { font-size: 15px; border-left: 4px solid #2563eb; padding-left: 8px; margin: 18px 0 8px; }
table { border-collapse: collapse; width: 100%; margin-top: 6px; }
th, td { border: 1px solid #cbd5e1; padding: 6px 8px; font-size: 13px; text-align: left; }
th { background: #eff6ff; width: 30%; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.xray-grid { display: grid; grid-template-columns: 1fr; gap: 8px; }
figure { margin: 0; break-inside: avoid; text-align: center; }
/* 算法结果图与 X 光统一尺寸（同高、整体更紧凑），完整显示不裁切 */
figure img { display: block; width: auto; max-width: 100%; max-height: 200px; margin: 0 auto; border: 1px solid #e2e8f0; border-radius: 4px; }
figcaption { font-size: 12px; color: #475569; text-align: center; margin-top: 3px; }
.opinion { font-size: 13px; line-height: 1.7; margin: 6px 0; white-space: pre-wrap; }
.logo { text-align: center; margin-bottom: 10px; }
.logo img { height: 56px; width: auto; border-radius: 10px; }
.page-break { break-before: page; page-break-before: always; }
</style></head><body>
<div class="logo"><img src="${logoInline}" alt="AIS" /></div>
<h1>${t("report.pdfTitle")}</h1>
<div class="sub">${t("report.pdfReportNo", { no: escapeHtml(r.reportNumber) })}${r.screeningDate ? `　|　${t("report.pdfScreeningDate", { date: escapeHtml(r.screeningDate) })}` : ""}</div>
<h2>${t("report.patientInfo")}</h2>
<table>
<tr><th>${t("report.caseNumber")}</th><td>${escapeHtml(r.caseNumber)}</td><th>${t("report.name")}</th><td>${escapeHtml(r.patientName)}</td></tr>
<tr><th>${t("report.gender")}</th><td>${r.gender === "M" ? t("enums.genderMale") : t("enums.genderFemale")}</td><th>${t("report.age")}</th><td>${t("report.ageValue", { value: escapeHtml(r.age) })}</td></tr>
<tr><th>${t("report.birthday")}</th><td>${escapeHtml(r.birthday)}</td><th>${t("report.heightWeight")}</th><td>${t("report.pdfHeightWeight", { height: escapeHtml(r.height), weight: escapeHtml(r.weight) })}</td></tr>
<tr><th>${t("report.screeningDate")}</th><td>${escapeHtml(r.screeningDate)}</td><th>${t("report.department")}</th><td>${escapeHtml(r.department)}</td></tr>
<tr><th>${t("report.doctor")}</th><td colspan="3">${escapeHtml(r.doctor)}</td></tr>
</table>
<h2>${t("report.algorithmOutput")}</h2>
<table>
<tr><th>Asymmetric Index</th><td>${metricText(r.indices.asymmetric_index)}</td></tr>
<tr><th>Curvature Index</th><td>${metricText(r.indices.curvature_index)}</td></tr>
<tr><th>Height Index</th><td>${metricText(r.indices.height_index)}</td></tr>
<tr><th>Normal Angle Index</th><td>${metricText(r.indices.normal_angle_index)}</td></tr>
<tr><th>${t("report.pdfCobb")}</th><td>${Math.round(r.predictedCobbAngle)}°</td></tr>
<tr><th>${t("report.pdfSeverity")}</th><td>${severityZh[r.severity] || escapeHtml(r.severity)}</td></tr>
</table>
<h2>${t("report.opinion")}</h2>
${opinion(t("report.pdfDiagnosis"), r.diagnosis)}
${opinion(t("report.pdfFollowup"), r.followupSuggestion)}
${opinion(t("report.pdfTreatment"), r.treatment)}
<div class="page-break">
<h2>${t("report.imaging")}</h2>
<div class="grid">
${figure(images.annotatedImage, t("report.annotatedImage"))}
${figure(images.moireImage, t("report.moireImage"))}
${figure(images.heatmapImage, t("report.heatmapImage"))}
${figure(images.normalAngleImage, t("report.normalAngleImage"))}
</div>
${xraySection}
</div>
</body></html>`;
};

export default function AnalysisReport() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const { t, i18n } = useTranslation();
    const role = sessionStorage.getItem("user_role");
    const [isAdmin] = useState(role === "system_admin" || role === "institution_admin" || role === "admin");
    const isSystemAdmin = role === "system_admin" || role === "institution_admin" || role === "admin";
    const reportIdParam = searchParams.get("reportId");
    const caseIdParam = searchParams.get("caseId");

    const [report, setReport] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(true);
    const [isEditingOpinion, setIsEditingOpinion] = useState(false);
    const [editedOpinion, setEditedOpinion] = useState({
        diagnosis: "",
        followupSuggestion: "",
        treatment: "",
    });
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [selectedImageTitle, setSelectedImageTitle] = useState("");
    const [selectedImageIsAnnotated, setSelectedImageIsAnnotated] = useState(false);
    const [isReanalyzing, setIsReanalyzing] = useState(false);
    const [imagesLoading, setImagesLoading] = useState(false);
    const reanalyzingRef = useRef(false);
    const [reviewing, setReviewing] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [annotationLoading, setAnnotationLoading] = useState(false);
    // Kept for compatibility with the legacy hidden annotation card while the
    // visible entry lives inside the annotated-image preview.
    const [annotationSubjects] = useState<string[]>([]);
    const [annotationSubjectId, setAnnotationSubjectId] = useState("");
    const annotationUpdated = searchParams.get("annotationUpdated") === "1";
    // 受检者级 X 光影像（附加资料，与报告不绑定）
    const [xrayFiles, setXrayFiles] = useState<{ id: string; uploadTime?: string }[]>([]);

    // 当 report 加载后同步诊断意见编辑状态
    useEffect(() => {
        if (report) {
            setEditedOpinion({
                diagnosis: report.diagnosis,
                followupSuggestion: report.followupSuggestion,
                treatment: report.treatment,
            });
        }
    }, [report]);

    useEffect(() => {
        const fetchReport = async () => {
            setLoading(true);
            try {
                if (reportIdParam) {
                    const data = await api.getReport(reportIdParam);
                    setReport(adaptReportData(data));
                } else if (caseIdParam) {
                    const result = await api.getReports({ caseId: caseIdParam, pageSize: 1 });
                    const list = result.list || result.data?.list || [];
                    if (list.length > 0) {
                        setReport(adaptReportData(list[0]));
                    } else {
                        toast.error(t("report.noReportFound"));
                        navigate(caseIdParam ? `/case-detail/${caseIdParam}` : "/cases");
                    }
                } else {
                    // 从 localStorage 获取最近一份报告
                    const linkedReports = getLinkedReports();
                    if (linkedReports.length > 0) {
                        setReport(adaptReportData(linkedReports[0]));
                    } else {
                        toast.error(t("report.noReportFound"));
                        navigate(caseIdParam ? `/case-detail/${caseIdParam}` : "/cases");
                    }
                }
            } catch (err) {
                console.error("获取报告失败:", err);
                toast.error(t("report.fetchFailed"));
                navigate(caseIdParam ? `/case-detail/${caseIdParam}` : "/cases");
            } finally {
                setLoading(false);
            }
        };

        fetchReport();
    }, [reportIdParam, caseIdParam]);

    // 加载该受检者的 X 光影像（附加数据，与报告不绑定）
    useEffect(() => {
        if (!report?.caseId) return;
        let cancelled = false;
        api.getFiles({ caseId: report.caseId, pageSize: 100 })
            .then((d) => {
                if (cancelled) return;
                const list = (d.list || d.data?.list || []).filter((f: any) =>
                    (f.kind || (String(f.fileName || f.originalName || "").toLowerCase().endsWith(".ply") ? "scan" : "xray")) === "xray",
                );
                setXrayFiles(list.map((f: any) => ({ id: f.id, uploadTime: f.uploadTime || f.createdAt })));
            })
            .catch(() => undefined);
        return () => { cancelled = true; };
    }, [report?.caseId]);

    const adaptReportData = (data: any): AnalysisResult => {
        // 计算严重程度
        const cobbAngle = data.predictedCobbAngle || data.cobbAngle || data.latestCobbAngle || 0;
        const aisLevel = data.aisLevel || data.latestAISLevel || '';
        let severity: Severity = "mild";
        if (aisLevel === '正常' || cobbAngle < 10) severity = "negative";
        else if (aisLevel === '轻度' || cobbAngle < 20) severity = "mild";
        else if (aisLevel === '中度' || cobbAngle < 40) severity = "moderate";
        else if (aisLevel === '重度' || cobbAngle >= 40) severity = "severe";

        // 处理嵌套的 case 和 file 对象
        const caseData = data.case || {};
        const fileData = data.file || {};

        return {
            reportNumber: data.id || data.reportNumber || data.reportNo || "",
            caseId: data.caseId || caseData.id || "",
            caseNumber: data.caseNumber || caseData.caseNumber || data.caseId || caseData.id || "",
            patientName: data.name || data.patientName || caseData.name || "",
            gender: (data.gender === '男' || caseData.gender === '男' || data.gender === "male" || caseData.gender === "male") ? 'M' : (data.gender === '女' || caseData.gender === '女' || data.gender === "female" || caseData.gender === "female") ? 'F' : 'M',
            age: data.birthDate || caseData.birthDate ? new Date().getFullYear() - new Date(data.birthDate || caseData.birthDate).getFullYear() : data.age || 0,
            birthday: formatDate(data.birthDate || caseData.birthDate || data.birthday),
            height: data.height || caseData.heightCm || caseData.height || 0,
            weight: data.weight || caseData.weightKg || caseData.weight || 0,
            department: data.department || caseData.department || "",
            screeningDate: formatDate(data.screeningDate || caseData.screeningDate || fileData.scanTime || fileData.createdAt),
            doctor: data.doctor || caseData.doctor || "",
            fileNumber: data.fileId || fileData.id || data.fileNumber || "",
            fileName: fileData.originalName || data.fileName || "",
            filePath: fileData.storedPath || data.path || data.filePath || "",
            fileSize: (fileData.sizeBytes || data.fileSize) ? `${((fileData.sizeBytes || data.fileSize) / 1024 / 1024).toFixed(1)} MB` : "",
            uploadTime: fileData.createdAt || data.uploadTime || "",
            indices: Object.fromEntries(Object.entries(data.indices || {}).map(([key, value]) => [key, Number(value)]).filter(([, value]) => Number.isFinite(value)).map(([key, value]) => [key, Math.round(Number(value) * 10) / 10])),
            predictedCobbAngle: cobbAngle,
            severity,
            backImage: data.backImage || "",
            annotatedImage: data.annotatedImage || "",
            heatmapImage: data.heatmapImage || "",
            moireImage: data.moireImage || "",
            normalAngleImage: data.normalAngleImage || "",
            diagnosis: data.clinicalDiagnosis || data.diagnosis || "",
            followupSuggestion: data.followUpAdvice || data.followupSuggestion || "",
            treatment: data.treatmentPlan || data.treatment || "",
            analysisTime: formatDateTime(data.completeTime || data.submitTime || data.analysisTime),
            reportGeneratedTime: formatDateTime(data.completeTime || data.reportGeneratedTime || data.reportTime),
            status: data.status || data.reportStatus,
            reviewComment: data.reviewComment || "",
            annotation: data.annotation || null,
        };
    };

    const hasOpinionChanges = report ? (
        editedOpinion.diagnosis !== report.diagnosis ||
        editedOpinion.followupSuggestion !== report.followupSuggestion ||
        editedOpinion.treatment !== report.treatment
    ) : false;

    // 轮询分析任务，完成后拉取最新报告并刷新页面；成功返回 true
    const pollTaskForReport = async (taskId: string, fallbackReportId: string): Promise<boolean> => {
        const deadline = Date.now() + 10 * 60 * 1000;
        while (Date.now() < deadline) {
            await new Promise((resolve) => window.setTimeout(resolve, 1000));
            let current: any;
            try {
                current = await api.getTask(taskId);
            } catch {
                continue; // 网络抖动则继续轮询
            }
            if (current?.status === "success") {
                const result = typeof current.resultJson === "string" ? JSON.parse(current.resultJson) : current.resultJson;
                const reportId = result?.reportId || fallbackReportId;
                const fresh = await api.getReport(reportId);
                setReport(adaptReportData(fresh));
                navigate(`/analysis-report?reportId=${encodeURIComponent(reportId)}`, { replace: true });
                return true;
            }
            if (["failed", "cancelled"].includes(current?.status)) {
                // 失败时带出真实原因（经后端消息转换），便于页面提示
                throw new Error(translateBackendMessage(current?.failureReason || t("report.reanalyzeFailed")));
            }
        }
        return false;
    };

    useEffect(() => {
        if (!annotationUpdated || !reportIdParam) return;
        let cancelled = false;
        (async () => {
            try {
                const data = await api.completeAnnotation(reportIdParam);
                if (cancelled) return;
                // ① 标注连线图已在服务端同步渲染好：立即重拉报告，秒级显示标注后的图像
                try {
                    const fresh = await api.getReport(reportIdParam);
                    if (!cancelled) setReport(adaptReportData(fresh));
                } catch { /* 拉取失败则等待重分析结果覆盖 */ }
                // ② 标注完成会自动触发 annotation_reanalysis：图片区显示 loading，等待其完成后刷新（Cobb/热力图）
                const taskId = data?.reanalysisTaskId;
                if (taskId) {
                    setImagesLoading(true);
                    try {
                        await pollTaskForReport(taskId, reportIdParam);
                    } finally {
                        if (!cancelled) setImagesLoading(false);
                    }
                }
            } catch { /* 静默：无标注/接口异常时保留当前数据，用户可手动"重新分析" */ }
        })();
        return () => { cancelled = true; };
    }, [annotationUpdated, reportIdParam]);

    const openAnnotation = async () => {
        if (!report) return;
        setAnnotationLoading(true);
        try {
            const session = await api.createAnnotationSession(report.reportNumber);
            window.location.href = session.annotationUrl;
        } catch (error) {
            toast.error(error instanceof Error ? error.message : t("report.openAnnotationFailed"));
        } finally {
            setAnnotationLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="layout-main">
                <Sidebar isAdmin={isAdmin} />
                <div className="layout-header">
                    <Header isAdmin={isAdmin} />
                </div>
                <div className="layout-content">
                    <div className="content-wrapper flex items-center justify-center h-64">
                        <p className="text-sm text-[color:var(--color-text-secondary)]">{t("common.loading")}</p>
                    </div>
                </div>
            </div>
        );
    }

    if (!report) {
        return (
            <div className="layout-main">
                <Sidebar isAdmin={isAdmin} />
                <div className="layout-header">
                    <Header isAdmin={isAdmin} />
                </div>
                <div className="layout-content">
                    <div className="content-wrapper flex items-center justify-center h-64">
                        <p className="text-sm text-[color:var(--color-text-secondary)]">{t("report.notFound")}</p>
                    </div>
                </div>
            </div>
        );
    }

    const handleSaveOpinion = async () => {
        if (!report) return;
        try {
            await api.updateReportDiagnosis(report.reportNumber, {
                clinicalDiagnosis: editedOpinion.diagnosis,
                followUpAdvice: editedOpinion.followupSuggestion,
                treatmentPlan: editedOpinion.treatment,
            });
            setReport((prev) => prev ? ({
                ...prev,
                diagnosis: editedOpinion.diagnosis,
                followupSuggestion: editedOpinion.followupSuggestion,
                treatment: editedOpinion.treatment,
            }) : null);
            setIsEditingOpinion(false);
        } catch (err) {
            console.error("保存诊断意见失败", err);
            toast.error(t("report.saveFailed"));
        }
    };

    const handleReanalyze = async () => {
        if (!report || reanalyzingRef.current) return;
        reanalyzingRef.current = true;
        setIsReanalyzing(true);
        setImagesLoading(true);
        try {
            const task = await api.reanalyzeReport(report.reportNumber);
            // 幂等：若后端返回进行中的任务（awaited）则等待其完成，同样会刷新页面
            const ok = await pollTaskForReport(task.id, report.reportNumber);
            if (!ok) throw new Error(t("report.reanalyzeFailed"));
        } catch (err) {
            console.error("Failed to reanalyze report:", err);
            toast.error(err instanceof Error ? err.message : t("report.reanalyzeFailed"));
        } finally {
            reanalyzingRef.current = false;
            setIsReanalyzing(false);
            setImagesLoading(false);
        }
    };

    const handleReview = async (action: "approve" | "return") => {
        if (!report) return;
        setReviewing(true);
        try {
            if (action === "approve") await api.approveReview(report.reportNumber);
            else await api.returnReview(report.reportNumber);
            setReport((prev) => prev ? { ...prev, status: action === "approve" ? "approved" : "review_returned" } : null);
            toast.success(action === "approve" ? t("report.reviewApproved") : t("report.reviewReturnedMsg"));
        } catch (err) {
            console.error("审核操作失败", err);
            toast.error(t("report.reviewOpFailed"));
        } finally {
            setReviewing(false);
        }
    };

    const toDataUrl = async (url: string): Promise<string | null> => {
        try {
            const res = await fetch(url);
            if (!res.ok) return null;
            const blob = await res.blob();
            return await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(String(reader.result));
                reader.onerror = () => reject(reader.error);
                reader.readAsDataURL(blob);
            });
        } catch {
            return null;
        }
    };

    const handleSavePdf = async () => {
        if (!report || isExporting) return;
        setIsExporting(true);
        try {
            // 图片转 base64 内嵌，保证打印窗口离线渲染、无加载时序问题
            const [annotatedImage, moireImage, heatmapImage, normalAngleImage] = await Promise.all([
                toDataUrl(report.annotatedImage),
                toDataUrl(report.moireImage),
                toDataUrl(report.heatmapImage),
                toDataUrl(report.normalAngleImage),
            ]);
            // X 光影像同样内嵌，使 PDF 内容与页面 X 光影像区块一致
            const xrayImages = await Promise.all(
                xrayFiles.map(async (x) => ({ dataUrl: await toDataUrl(api.fileDownloadUrl(x.id)), uploadTime: x.uploadTime })),
            );
            const html = buildReportPdfHtml(report, { annotatedImage, moireImage, heatmapImage, normalAngleImage }, t, xrayImages);
            if (window.ais) {
                const res = await window.ais.saveReportPdf(html, `${report.reportNumber || t("report.pdfDefaultName")}.pdf`, i18n.language);
                if (res?.ok) toast.success(t("report.pdfSaved", { path: res.filePath }));
                else if (!res?.canceled) toast.error(t("report.pdfFailed"));
            } else {
                // 非 Electron 环境回退到系统打印对话框
                window.print();
            }
        } catch (err) {
            console.error("导出 PDF 失败:", err);
            toast.error(t("report.pdfFailed"));
        } finally {
            setIsExporting(false);
        }
    };

    const getSeverityColor = (severity: Severity) => {
        const colors: Record<Severity, string> = {
            negative: "text-[color:var(--color-success)]",
            mild: "text-[color:var(--color-warning)]",
            moderate: "text-[color:var(--color-error)]",
            severe: "text-[color:var(--color-error)] font-bold",
        };
        return colors[severity];
    };

    const getSeverityLabel = (severity: Severity) => {
        const labels: Record<Severity, string> = {
            negative: t("enums.severityNegative"),
            mild: t("enums.severityMild"),
            moderate: t("enums.severityModerate"),
            severe: t("enums.severitySevere"),
        };
        return labels[severity];
    };

    return (
        <div className="layout-main">
            <Sidebar isAdmin={isAdmin} />
            <div className="layout-header">
                <Header isAdmin={isAdmin} />
            </div>

            <div className="layout-content">
                <div className="content-wrapper">
                    <div className="flex items-center justify-between mb-4">
                        <div>
                            <h1 className="text-3xl font-semibold tracking-tight text-foreground text-[color:var(--color-text-primary)] mb-2">{t("report.title")}</h1>
                            <p className="text-sm text-[color:var(--color-text-secondary)]">
                                {t("report.reportNo", { no: report.reportNumber, name: report.patientName })}
                            </p>
                        </div>

                        <div className="flex flex-wrap items-center justify-end gap-2 max-w-2xl">
                            {/* Review actions are grouped in the status panel below. */}
                            {false && isSystemAdmin && report.status === "under_review" && <>
                                <button onClick={() => handleReview("approve")} disabled={reviewing} className="btn-primary">{t("report.approve")}</button>
                                <button onClick={() => handleReview("return")} disabled={reviewing} className="btn-secondary">{t("report.return")}</button>
                            </>}
                            <button onClick={() => navigate(caseIdParam ? `/case-detail/${caseIdParam}` : `/case-detail/${report.caseId}`)} className="btn-secondary">
                                {t("report.backToCase")}
                            </button>
                            <button onClick={handleReanalyze} disabled={isReanalyzing} className="btn-secondary">
                                {isReanalyzing ? t("report.analyzing") : t("report.reanalyze")}
                            </button>
                            <button onClick={handleSaveOpinion} disabled={!hasOpinionChanges} className="btn-primary">
                                {t("report.save")}
                            </button>
                            <button onClick={handleSavePdf} disabled={isExporting} className="btn-secondary">{isExporting ? t("report.exporting") : t("report.savePdf")}</button>
                        </div>
                    </div>

                    {isReanalyzing && <div className="card-base p-4 mb-5 border-l-4 border-[color:var(--color-warning)] bg-amber-50"><p className="font-semibold text-[color:var(--color-warning)]">{t("report.statusAnalyzing")}</p></div>}
                    {report.status === "under_review" && (
                        <div className="card-base p-4 mb-5 border-l-4 border-[color:var(--color-warning)] bg-amber-50">
                            <div className="flex flex-col md:flex-row md:items-center gap-3 md:justify-between">
                                <div>
                                    <p className="font-semibold text-[color:var(--color-warning)]">{t("report.statusUnderReview")}</p>
                                    <p className="text-sm text-muted-foreground text-[color:var(--color-text-secondary)]">{isSystemAdmin ? t("report.statusUnderReviewHintAdmin") : t("report.statusUnderReviewHintUser")}</p>
                                </div>
                                {isSystemAdmin && <div className="flex gap-2"><button className="btn-primary" disabled={reviewing} onClick={() => handleReview("approve")}>{t("report.approve")}</button><button className="btn-secondary" disabled={reviewing} onClick={() => handleReview("return")}>{t("report.return")}</button></div>}
                            </div>
                        </div>
                    )}
                    {report.status === "review_returned" && <div className="card-base p-4 mb-4 text-[color:var(--color-error)]">{t("report.statusReviewReturned", { comment: report.reviewComment || t("report.statusReviewReturnedDefault") })}</div>}
                    {report.status === "approved" && <div className="card-base p-4 mb-4 border-l-4 border-[color:var(--color-success)] bg-emerald-50"><p className="font-semibold text-[color:var(--color-success)]">{t("report.statusApproved")}</p></div>}

                    <div className="space-y-4">
                        {/* 第一行：受检者信息（左） | 算法输出结果（右） */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="card-base p-4">
                                <h2 className="text-lg font-semibold text-foreground text-[color:var(--color-text-primary)] mb-4 flex items-center gap-2">
                                    <span className="inline-block h-4 w-1 rounded-full bg-[color:var(--color-primary)]" />
                                    {t("report.patientInfo")}
                                </h2>
                                <div className="grid grid-cols-2 gap-2">
                                    <Field label={t("report.caseNumber")} value={report.caseNumber} />
                                    <Field label={t("report.name")} value={report.patientName} />
                                    <Field label={t("report.gender")} value={report.gender === "M" ? t("enums.genderMale") : t("enums.genderFemale")} />
                                    <Field label={t("report.age")} value={t("report.ageValue", { value: report.age })} />
                                    <Field label={t("report.birthday")} value={report.birthday} />
                                    <Field label={t("report.heightWeight")} value={t("report.heightWeightValue", { height: report.height, weight: report.weight })} />
                                    <Field label={t("report.screeningDate")} value={report.screeningDate} />
                                    <Field label={t("report.department")} value={report.department} />
                                    <Field label={t("report.doctor")} value={report.doctor} />
                                </div>
                                {false && isSystemAdmin && report.status === "under_review" && (
                                    <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4">
                                        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                                            <div>
                                                <p className="font-semibold text-blue-900">{t("report.annotationTitle")}</p>
                                                <p className="text-sm text-muted-foreground text-blue-700">{t("report.annotationHint")}</p>
                                                <p className="mt-1 text-sm text-muted-foreground text-blue-700">{t("report.annotationStatus")}：{report.annotation?.status === "updated" ? t("report.annotationStatusUpdated") : report.annotation ? t("report.annotationStatusBound", { subject: report.annotation.subjectId }) : t("report.annotationStatusPending")}</p>
                                            </div>
                                            <div className="flex flex-wrap items-center gap-2">
                                                {annotationSubjects.length > 0 && <select className="input-base w-auto min-w-[180px] bg-white" value={annotationSubjectId} onChange={(e) => setAnnotationSubjectId(e.target.value)} aria-label={t("report.selectSubject")}>
                                                    {annotationSubjects.map((id) => <option key={id} value={id}>{id}</option>)}
                                                </select>}
                                                <button className="btn-primary whitespace-nowrap" onClick={openAnnotation} disabled={annotationLoading || !annotationSubjectId}>{annotationLoading ? t("report.opening") : t("report.openAnnotation")}</button>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="card-base p-4">
                                <h2 className="text-lg font-semibold text-foreground text-[color:var(--color-text-primary)] mb-4 flex items-center gap-2">
                                    <span className="inline-block h-4 w-1 rounded-full bg-[color:var(--color-primary)]" />
                                    {t("report.algorithmOutput")}
                                </h2>
                                <div className="grid grid-cols-2 gap-2">
                                    <Field label="Asymmetric Index" value={report.indices.asymmetric_index} />
                                    <Field label="Curvature Index" value={report.indices.curvature_index} />
                                    <Field label="Height Index" value={report.indices.height_index} />
                                    <Field label="Normal Angle Index" value={report.indices.normal_angle_index} />
                                    <Field label="Cobb Angle" value={Math.round(report.predictedCobbAngle)} unit="°" critical={report.predictedCobbAngle >= 15} />
                                    <Field label={t("report.severity")} value={getSeverityLabel(report.severity)} valueClassName={getSeverityColor(report.severity)} />
                                </div>
                            </div>
                        </div>

                        {/* 第二行：影像结果（左） | X 光影像（右） */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="card-base p-4">
                                <h2 className="text-lg font-semibold text-foreground text-[color:var(--color-text-primary)] mb-3">{t("report.imaging")}</h2>
                                <div className="grid grid-cols-2 gap-4">
                                    <ImageCard title={t("report.annotatedImage")} image={report.annotatedImage} loading={imagesLoading} onClick={() => { setSelectedImageTitle(t("report.annotatedImage")); setSelectedImageIsAnnotated(true); setSelectedImage(report.annotatedImage); }} />
                                    <ImageCard title={t("report.moireImage")} image={report.moireImage} loading={imagesLoading} onClick={() => { setSelectedImageTitle(t("report.moireImage")); setSelectedImageIsAnnotated(false); setSelectedImage(report.moireImage); }} />
                                    <ImageCard title={t("report.heatmapImage")} image={report.heatmapImage} loading={imagesLoading} onClick={() => { setSelectedImageTitle(t("report.heatmapImage")); setSelectedImageIsAnnotated(false); setSelectedImage(report.heatmapImage); }} />
                                    <ImageCard title={t("report.normalAngleImage")} image={report.normalAngleImage} loading={imagesLoading} onClick={() => { setSelectedImageTitle(t("report.normalAngleImage")); setSelectedImageIsAnnotated(false); setSelectedImage(report.normalAngleImage); }} />
                                </div>
                            </div>

                            <div className="card-base p-4">
                                <h2 className="text-lg font-semibold text-foreground text-[color:var(--color-text-primary)] mb-3">{t("report.xrayImaging")}</h2>
                                {xrayFiles.length === 0 ? (
                                    <p className="text-sm text-muted-foreground text-[color:var(--color-text-tertiary)]">{t("report.xrayEmpty")}</p>
                                ) : (
                                    <div className="grid grid-cols-2 gap-4">
                                        {xrayFiles.map((x) => (
                                            <button key={x.id} className="card-base overflow-hidden transition-shadow hover:shadow-lg cursor-pointer" onClick={() => { setSelectedImageTitle(t("report.xrayImaging")); setSelectedImageIsAnnotated(false); setSelectedImage(api.fileDownloadUrl(x.id)); }}>
                                                <div className="w-full aspect-[3/4] max-h-[280px] bg-gray-900 flex items-center justify-center overflow-hidden">
                                                    <img src={api.fileDownloadUrl(x.id)} alt="X-ray" className="max-w-full max-h-full object-contain" />
                                                </div>
                                                <div className="p-3">
                                                    <p className="text-sm text-muted-foreground font-semibold text-[color:var(--color-text-primary)]">{t("report.xrayImage")} {formatDateTime(x.uploadTime)}</p>
                                                    <p className="text-sm text-muted-foreground text-[color:var(--color-text-tertiary)]">{t("report.clickToEnlarge")}</p>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* 最后一行（通栏）：诊断意见 */}
                        <div className="card-base p-4">
                            <div className="flex items-center justify-between mb-3">
                                <h2 className="text-lg font-semibold text-foreground text-[color:var(--color-text-primary)]">{t("report.opinion")}</h2>
                                <button onClick={() => setIsEditingOpinion((prev) => !prev)} className="btn-secondary">
                                    {isEditingOpinion ? t("report.cancelEdit") : t("report.edit")}
                                </button>
                            </div>

                            {isEditingOpinion ? (
                                <div className="space-y-4">
                                    <EditorField label={t("report.clinicalDiagnosis")} value={editedOpinion.diagnosis} onChange={(value) => setEditedOpinion((prev) => ({ ...prev, diagnosis: value }))} rows={4} />
                                    <EditorField label={t("report.followupSuggestion")} value={editedOpinion.followupSuggestion} onChange={(value) => setEditedOpinion((prev) => ({ ...prev, followupSuggestion: value }))} rows={3} />
                                    <EditorField label={t("report.treatmentPlan")} value={editedOpinion.treatment} onChange={(value) => setEditedOpinion((prev) => ({ ...prev, treatment: value }))} rows={3} />
                                    <div className="flex justify-end gap-3 pt-1">
                                        <button onClick={handleSaveOpinion} disabled={!hasOpinionChanges} className="btn-primary">{t("report.save")}</button>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <OpinionBlock title={t("report.clinicalDiagnosis")} content={report.diagnosis} />
                                    <OpinionBlock title={t("report.followupSuggestion")} content={report.followupSuggestion} />
                                    <OpinionBlock title={t("report.treatmentPlan")} content={report.treatment} />
                                </div>
                            )}
                        </div>
                    </div>

                </div>
            </div>

                            {selectedImage && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedImage(null)}>
                    <div className="card-base max-w-3xl w-full p-6" onClick={(e) => e.stopPropagation()}>
                        <div className="w-full bg-gray-900 rounded-btn flex items-center justify-center" style={{ maxHeight: '70vh' }}>
                            <img src={selectedImage} alt="Enlarged" className="max-w-full max-h-[70vh] object-contain rounded-btn" />
                        </div>
                        <div className="mt-4 flex flex-wrap justify-end gap-2">
                            {isSystemAdmin && report.status === "under_review" && selectedImageIsAnnotated && <button onClick={openAnnotation} disabled={annotationLoading} className="btn-primary">{annotationLoading ? t("report.opening") : t("report.enterAnnotation")}</button>}
                            <button onClick={() => setSelectedImage(null)} className="btn-secondary">{t("common.close")}</button>
                        </div>
                    </div>
                </div>
                            )}

        </div>
    );
}

function Field({ label, value, unit = "", critical = false, valueClassName = "" }: { label: string; value: React.ReactNode; unit?: string; critical?: boolean; valueClassName?: string }) {
    return (
        <div className={`rounded-lg border px-3 py-2.5 min-w-0 ${critical ? "border-[color:var(--color-error)] bg-red-50" : "border-[color:var(--color-border)] bg-[color:var(--color-neutral)]"}`}>
            <p className="text-xs text-[color:var(--color-text-tertiary)] mb-1">{label}</p>
            <div className="flex items-baseline gap-1">
                <span className={`text-sm font-semibold truncate ${critical ? "text-[color:var(--color-error)]" : valueClassName || "text-[color:var(--color-text-primary)]"}`}>{value}</span>
                {unit && <span className="text-xs text-[color:var(--color-text-secondary)] shrink-0">{unit}</span>}
            </div>
        </div>
    );
}

function ImageCard({ title, image, onClick, loading = false }: { title: string; image: string; onClick: () => void; loading?: boolean }) {
    const { t } = useTranslation();
    const hasImage = image && image.trim().length > 0;
    const handleClick = () => {
        if (hasImage && !loading) {
            onClick();
        }
    };

    return (
        <button
            onClick={handleClick}
            className={`card-base overflow-hidden transition-shadow ${hasImage && !loading ? 'hover:shadow-lg cursor-pointer' : 'cursor-not-allowed opacity-60'}`}
            disabled={!hasImage || loading}
        >
            {loading ? (
                <div className="w-full aspect-[3/4] max-h-[280px] bg-gray-900 flex flex-col items-center justify-center gap-2 overflow-hidden">
                    <span className="h-8 w-8 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    <span className="text-sm text-white/80">{t("report.analyzing")}</span>
                </div>
            ) : hasImage ? (
                <div className="w-full aspect-[3/4] max-h-[280px] bg-gray-900 flex items-center justify-center overflow-hidden">
                    <img src={image} alt={title} className="max-w-full max-h-full object-contain" />
                </div>
            ) : (
                <div className="w-full aspect-[3/4] max-h-[280px] bg-[color:var(--color-neutral)] flex items-center justify-center">
                    <span className="text-sm text-muted-foreground text-[color:var(--color-text-tertiary)]">{t("report.noImage")}</span>
                </div>
            )}
            <div className="p-3">
                <p className="text-sm text-muted-foreground font-semibold text-[color:var(--color-text-primary)]">{title}</p>
                <p className="text-sm text-muted-foreground text-[color:var(--color-text-tertiary)]">{loading ? t("report.analyzing") : hasImage ? t("report.clickToEnlarge") : t("report.noImageHint")}</p>
            </div>
        </button>
    );
}

function EditorField({ label, value, onChange, rows }: { label: string; value: string; onChange: (value: string) => void; rows: number }) {
    return (
        <div>
            <label className="block text-sm font-semibold text-[color:var(--color-text-primary)] mb-2">{label}</label>
            <textarea className="input-base" rows={rows} value={value} onChange={(e) => onChange(e.target.value)} />
        </div>
    );
}

function OpinionBlock({ title, content }: { title: string; content: string }) {
    return (
        <div>
            <h3 className="text-sm font-semibold text-[color:var(--color-text-primary)] mb-2">{title}</h3>
            <p className="text-sm text-[color:var(--color-text-secondary)] leading-relaxed">{content}</p>
        </div>
    );
}
