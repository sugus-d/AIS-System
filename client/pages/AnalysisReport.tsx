import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { getLinkedReports } from "@/lib/workflowStore";
import api from "@/lib/api";

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
    diagnosis: string;
    followupSuggestion: string;
    treatment: string;
    analysisTime: string;
    reportGeneratedTime: string;
    status?: string;
    reviewComment?: string;
    annotation?: { subjectId: string; status: "bound" | "updated"; updatedAt?: string; updatedBy?: string } | null;
}

export default function AnalysisReport() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const role = localStorage.getItem("user_role");
    const [isAdmin] = useState(role === "system_admin" || role === "admin");
    const isSystemAdmin = role === "system_admin" || role === "admin";
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
    const [reviewing, setReviewing] = useState(false);
    const [annotationLoading, setAnnotationLoading] = useState(false);
    // Kept for compatibility with the legacy hidden annotation card while the
    // visible entry lives inside the annotated-image preview.
    const [annotationSubjects] = useState<string[]>([]);
    const [annotationSubjectId, setAnnotationSubjectId] = useState("");
    const annotationUpdated = searchParams.get("annotationUpdated") === "1";

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
                        alert("未找到相关报告");
                        navigate(caseIdParam ? `/case-detail/${caseIdParam}` : "/cases");
                    }
                } else {
                    // 从 localStorage 获取最近一份报告
                    const linkedReports = getLinkedReports();
                    if (linkedReports.length > 0) {
                        setReport(adaptReportData(linkedReports[0]));
                    } else {
                        alert("未找到相关报告");
                        navigate(caseIdParam ? `/case-detail/${caseIdParam}` : "/cases");
                    }
                }
            } catch (err) {
                console.error("获取报告失败:", err);
                alert("获取报告失败");
                navigate(caseIdParam ? `/case-detail/${caseIdParam}` : "/cases");
            } finally {
                setLoading(false);
            }
        };

        fetchReport();
    }, [reportIdParam, caseIdParam]);

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
            caseNumber: data.caseId || caseData.id || data.caseNumber || data.caseNo || "",
            patientName: data.name || data.patientName || caseData.name || "",
            gender: (data.gender === '男' || caseData.gender === '男') ? 'M' : (data.gender === '女' || caseData.gender === '女') ? 'F' : data.gender === "male" ? "M" : data.gender === "female" ? "F" : "M",
            age: data.birthDate || caseData.birthDate ? new Date().getFullYear() - new Date(data.birthDate || caseData.birthDate).getFullYear() : data.age || 0,
            birthday: data.birthDate || caseData.birthDate || data.birthday || "",
            height: data.height || caseData.height || 0,
            weight: data.weight || caseData.weight || 0,
            department: data.department || caseData.department || "",
            screeningDate: data.screeningDate || caseData.screeningDate || "",
            doctor: data.doctor || caseData.doctor || "",
            fileNumber: data.fileId || fileData.id || data.fileNumber || "",
            fileName: data.fileName || fileData.fileName || "",
            filePath: data.originalPath || fileData.originalPath || data.filePath || "",
            fileSize: data.fileSize || fileData.fileSize ? `${((data.fileSize || fileData.fileSize) / 1024 / 1024).toFixed(1)} MB` : "",
            uploadTime: data.uploadTime || fileData.uploadTime || "",
            indices: Object.fromEntries(Object.entries(data.indices || {}).map(([key, value]) => [key, Number(value)]).filter(([, value]) => Number.isFinite(value))),
            predictedCobbAngle: cobbAngle,
            severity,
            backImage: data.backImage || "",
            annotatedImage: data.annotatedImage || "",
            heatmapImage: data.heatmapImage || "",
            moireImage: data.moireImage || "",
            diagnosis: data.clinicalDiagnosis || data.diagnosis || "",
            followupSuggestion: data.followUpAdvice || data.followupSuggestion || "",
            treatment: data.treatmentPlan || data.treatment || "",
            analysisTime: data.completeTime || data.submitTime || data.analysisTime || "",
            reportGeneratedTime: data.completeTime || data.reportGeneratedTime || data.reportTime || "",
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

    useEffect(() => {
        if (!annotationUpdated || !reportIdParam) return;
        api.completeAnnotation(reportIdParam).then(() => {
            setReport((prev) => prev ? { ...prev, annotation: prev.annotation ? { ...prev.annotation, status: "updated", updatedAt: new Date().toISOString() } : prev.annotation } : prev);
        }).catch(() => undefined);
    }, [annotationUpdated, reportIdParam]);

    const openAnnotation = async () => {
        if (!report) return;
        setAnnotationLoading(true);
        try {
            const session = await api.createAnnotationSession(report.reportNumber);
            window.location.href = session.annotationUrl;
        } catch (error) {
            alert(error instanceof Error ? error.message : "无法打开标注工具");
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
                        <p className="text-body text-[color:var(--color-text-secondary)]">加载中...</p>
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
                        <p className="text-body text-[color:var(--color-text-secondary)]">报告不存在</p>
                    </div>
                </div>
            </div>
        );
    }

    const handleSaveOpinion = async () => {
        if (!report) return;
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
    };

    const handleReanalyze = async () => {
        if (!report) return;
        setIsReanalyzing(true);
        try {
            const task = await api.analyzeSingle(report.caseId, report.fileNumber);
            const deadline = Date.now() + 10 * 60 * 1000;
            while (Date.now() < deadline) {
                await new Promise((resolve) => window.setTimeout(resolve, 1000));
                const current = await api.getTask(task.id);
                if (current.status === "success") {
                    const result = typeof current.resultJson === "string" ? JSON.parse(current.resultJson) : current.resultJson;
                    if (!result?.reportId) throw new Error("Analysis completed without a report ID.");
                    navigate(`/analysis-report?reportId=${result.reportId}`, { replace: true });
                    return;
                }
                if (["failed", "cancelled"].includes(current.status)) throw new Error(current.failureReason || "AIS analysis failed.");
            }
            throw new Error("AIS analysis timed out.");
        } catch (err) {
            console.error("Failed to reanalyze report:", err);
            alert("重新分析失败，请稍后重试");
        } finally {
            setIsReanalyzing(false);
        }
    };

    const handleReview = async (action: "approve" | "return") => {
        if (!report) return;
        setReviewing(true);
        try {
            if (action === "approve") await api.approveReview(report.reportNumber);
            else await api.returnReview(report.reportNumber);
            setReport((prev) => prev ? { ...prev, status: action === "approve" ? "analyzed" : "review_returned" } : null);
            alert(action === "approve" ? "审核通过，报告已分析" : "报告已退回操作员修改");
        } catch (err) {
            console.error("审核操作失败", err);
            alert("审核操作失败，请重试");
        } finally {
            setReviewing(false);
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
            negative: "正常",
            mild: "轻度",
            moderate: "中度",
            severe: "严重",
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
                            <h1 className="text-page-title text-[color:var(--color-text-primary)] mb-2">报告详情</h1>
                            <p className="text-body text-[color:var(--color-text-secondary)]">
                                报告编号：{report.reportNumber} | 受检者：{report.patientName}
                            </p>
                        </div>

                        <div className="flex flex-wrap items-center justify-end gap-2 max-w-2xl">
                            {/* Review actions are grouped in the status panel below. */}
                            {false && isSystemAdmin && report.status === "under_review" && <>
                                <button onClick={() => handleReview("approve")} disabled={reviewing} className="btn-primary">审核通过</button>
                                <button onClick={() => handleReview("return")} disabled={reviewing} className="btn-secondary">审核不通过</button>
                            </>}
                            <button onClick={() => navigate(caseIdParam ? `/case-detail/${caseIdParam}` : `/case-detail/${report.caseId}`)} className="btn-secondary">
                                返回受检者详情
                            </button>
                            <button onClick={handleReanalyze} disabled={isReanalyzing} className="btn-secondary">
                                {isReanalyzing ? "分析中..." : "重新分析"}
                            </button>
                            <button onClick={handleSaveOpinion} disabled={!hasOpinionChanges} className="btn-primary">
                                保存
                            </button>
                            <button onClick={() => window.print()} className="btn-secondary">打印 / 另存为 PDF</button>
                        </div>
                    </div>

                    {report.status === "under_review" && (
                        <div className="card-base p-4 mb-5 border-l-4 border-[color:var(--color-warning)] bg-amber-50">
                            <div className="flex flex-col md:flex-row md:items-center gap-3 md:justify-between">
                                <div>
                                    <p className="font-semibold text-[color:var(--color-warning)]">当前状态：审核中</p>
                                    <p className="text-helper text-[color:var(--color-text-secondary)]">{isSystemAdmin ? "请核对本报告内容后，在右侧完成审核。" : "报告正在等待系统管理员审核。"}</p>
                                </div>
                                {isSystemAdmin && <div className="flex gap-2"><button className="btn-primary" disabled={reviewing} onClick={() => handleReview("approve")}>审核通过</button><button className="btn-secondary" disabled={reviewing} onClick={() => handleReview("return")}>退回修改</button></div>}
                            </div>
                        </div>
                    )}
                    {report.status === "review_returned" && <div className="card-base p-4 mb-4 text-[color:var(--color-error)]">审核退回：{report.reviewComment || "请根据审核意见修改后重新提交。"}</div>}

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                        <div className="space-y-3">
                            <div className="card-base p-4">
                                <h2 className="text-card-title text-[color:var(--color-text-primary)] mb-3">受检者信息</h2>
                                <div className="grid grid-cols-2 gap-x-4">
                                    <InfoItem label="受检者编号" value={report.caseNumber} />
                                    <InfoItem label="姓名" value={report.patientName} />
                                    <InfoItem label="性别" value={report.gender === "M" ? "男" : "女"} />
                                    <InfoItem label="年龄" value={`${report.age} 岁`} />
                                    <InfoItem label="出生日期" value={report.birthday} />
                                    <InfoItem label="身高 / 体重" value={`${report.height} cm / ${report.weight} kg`} />
                                    <InfoItem label="筛查日期" value={report.screeningDate} />
                                    <InfoItem label="就诊科室" value={report.department} />
                                    <InfoItem label="就诊医生" value={report.doctor} />
                                </div>
                                {false && isSystemAdmin && report.status === "under_review" && (
                                    <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4">
                                        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                                            <div>
                                                <p className="font-semibold text-blue-900">3D 标注修正</p>
                                                <p className="text-helper text-blue-700">修正 Landmark / ROI 后返回本报告重新分析。</p>
                                                <p className="mt-1 text-helper text-blue-700">状态：{report.annotation?.status === "updated" ? "已更新，可重新分析" : report.annotation ? `已绑定 ${report.annotation.subjectId}` : "待绑定"}</p>
                                            </div>
                                            <div className="flex flex-wrap items-center gap-2">
                                                {annotationSubjects.length > 0 && <select className="input-base w-auto min-w-[180px] bg-white" value={annotationSubjectId} onChange={(e) => setAnnotationSubjectId(e.target.value)} aria-label="选择标注数据">
                                                    {annotationSubjects.map((id) => <option key={id} value={id}>{id}</option>)}
                                                </select>}
                                                <button className="btn-primary whitespace-nowrap" onClick={openAnnotation} disabled={annotationLoading || !annotationSubjectId}>{annotationLoading ? "打开中..." : "打开标注工具"}</button>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="card-base p-4">
                                <h2 className="text-card-title text-[color:var(--color-text-primary)] mb-3">算法输出结果</h2>
                                <div className="grid grid-cols-2 gap-2">
                                    <MetricDisplay label="Asymmetric Index" value={report.indices.asymmetric_index} unit="" />
                                    <MetricDisplay label="Curvature Index" value={report.indices.curvature_index} unit="" />
                                    <MetricDisplay label="Height Index" value={report.indices.height_index} unit="" />
                                    <MetricDisplay label="Normal Angle Index" value={report.indices.normal_angle_index} unit="" />
                                    <MetricDisplay label="Cobb Angle" value={report.predictedCobbAngle} unit="°" critical={report.predictedCobbAngle >= 15} />
                                </div>
                                <div className="mt-3 pt-3 border-t border-[color:var(--color-border)] flex items-center gap-3">
                                    <span className="text-helper text-[color:var(--color-text-tertiary)]">AIS 严重等级</span>
                                    <span className={`text-body font-bold ${getSeverityColor(report.severity)}`}>{getSeverityLabel(report.severity)}</span>
                                </div>
                            </div>

                            <div className="card-base p-4">
                                <div className="flex items-center justify-between mb-3">
                                    <h2 className="text-card-title text-[color:var(--color-text-primary)]">诊断意见</h2>
                                    <button onClick={() => setIsEditingOpinion((prev) => !prev)} className="btn-secondary">
                                        {isEditingOpinion ? "取消编辑" : "编辑"}
                                    </button>
                                </div>

                                {isEditingOpinion ? (
                                    <div className="space-y-4">
                                        <EditorField label="临床诊断" value={editedOpinion.diagnosis} onChange={(value) => setEditedOpinion((prev) => ({ ...prev, diagnosis: value }))} rows={4} />
                                        <EditorField label="随访建议" value={editedOpinion.followupSuggestion} onChange={(value) => setEditedOpinion((prev) => ({ ...prev, followupSuggestion: value }))} rows={3} />
                                        <EditorField label="治疗方案" value={editedOpinion.treatment} onChange={(value) => setEditedOpinion((prev) => ({ ...prev, treatment: value }))} rows={3} />
                                    </div>
                                ) : (
                                    <div className="space-y-2">
                                        <OpinionBlock title="临床诊断" content={report.diagnosis} />
                                        <OpinionBlock title="随访建议" content={report.followupSuggestion} />
                                        <OpinionBlock title="治疗方案" content={report.treatment} />
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="space-y-3">
                            <div className="card-base p-4">
                                <h2 className="text-card-title text-[color:var(--color-text-primary)] mb-3">影像结果</h2>
                                <div className="grid grid-cols-2 gap-4">
                                    <ImageCard title="初始背部图像" image={report.backImage} onClick={() => { setSelectedImageTitle("初始背部图像"); setSelectedImageIsAnnotated(false); setSelectedImage(report.backImage); }} />
                                    <ImageCard title="带标注的背部图像" image={report.annotatedImage} onClick={() => { setSelectedImageTitle("带标注的背部图像"); setSelectedImageIsAnnotated(true); setSelectedImage(report.annotatedImage); }} />
                                    <ImageCard title="背部曲率热力图" image={report.heatmapImage} onClick={() => { setSelectedImageTitle("背部曲率热力图"); setSelectedImageIsAnnotated(false); setSelectedImage(report.heatmapImage); }} />
                                    <ImageCard title="Moire 影像" image={report.moireImage} onClick={() => { setSelectedImageTitle("Moire 影像"); setSelectedImageIsAnnotated(false); setSelectedImage(report.moireImage); }} />
                                </div>
                            </div>

                            <div className="card-base p-4">
                                <h2 className="text-card-title text-[color:var(--color-text-primary)] mb-3">运行数据</h2>
                                <div className="grid grid-cols-2 gap-x-4">
                                    <InfoItem label="提交分析时间" value={report.analysisTime} />
                                    <InfoItem label="完成分析时间" value={report.reportGeneratedTime} />
                                </div>
                            </div>

                            <div className="card-base p-4">
                                <h2 className="text-card-title text-[color:var(--color-text-primary)] mb-3">文件信息</h2>
                                <div className="grid grid-cols-2 gap-x-4">
                                    <InfoItem label="文件编号" value={report.fileNumber} />
                                    <InfoItem label="文件名" value={report.fileName} />
                                    <InfoItem label="文件大小" value={report.fileSize} />
                                    <InfoItem label="上传时间" value={report.uploadTime} />
                                    <InfoItem label="文件路径" value={report.filePath} />
                                    <InfoItem label="关联状态" value={report.linkedDeletedRemark || "正常"} />
                                </div>
                            </div>
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
                            {isSystemAdmin && report.status === "under_review" && selectedImageIsAnnotated && <button onClick={openAnnotation} disabled={annotationLoading} className="btn-primary">{annotationLoading ? "打开中..." : "进入标注平台"}</button>}
                            <button onClick={() => setSelectedImage(null)} className="btn-secondary">关闭</button>
                        </div>
                    </div>
                </div>
                            )}

        </div>
    );
}

function InfoRow({ label, value }: { label: string; value: string | React.ReactNode }) {
    return (
        <div className="flex justify-between items-start py-2 border-b border-[color:var(--color-border)] last:border-b-0 gap-4">
            <span className="text-body text-[color:var(--color-text-tertiary)] shrink-0">{label}</span>
            <span className="text-body font-semibold text-[color:var(--color-text-primary)] text-right break-all">{value}</span>
        </div>
    );
}

function MetricDisplay({ label, value, unit, critical }: { label: string; value: number; unit: string; critical?: boolean }) {
    return (
        <div className={`p-3 rounded-btn border ${critical ? "border-[color:var(--color-error)] bg-red-50" : "border-[color:var(--color-border)] bg-[color:var(--color-neutral)]"}`}>
            <p className="text-xs text-[color:var(--color-text-tertiary)] mb-0.5">{label}</p>
            <div className="flex items-baseline gap-1">
                <span className={`text-body font-bold ${critical ? "text-[color:var(--color-error)]" : "text-[color:var(--color-primary)]"}`}>{value}</span>
                <span className="text-xs text-[color:var(--color-text-secondary)]">{unit}</span>
            </div>
        </div>
    );
}

function InfoItem({ label, value }: { label: string; value: string | React.ReactNode }) {
    return (
        <div className="py-1.5 border-b border-[color:var(--color-border)] min-w-0">
            <p className="text-xs text-[color:var(--color-text-tertiary)] mb-0.5">{label}</p>
            <p className="text-sm font-semibold text-[color:var(--color-text-primary)] truncate">{value}</p>
        </div>
    );
}

function ImageCard({ title, image, onClick }: { title: string; image: string; onClick: () => void }) {
    const hasImage = image && image.trim().length > 0;
    const handleClick = () => {
        if (hasImage) {
            onClick();
        }
    };

    return (
        <button
            onClick={handleClick}
            className={`card-base overflow-hidden transition-shadow ${hasImage ? 'hover:shadow-lg cursor-pointer' : 'cursor-not-allowed opacity-60'}`}
            disabled={!hasImage}
        >
            {hasImage ? (
                <div className="w-full h-52 bg-gray-900 flex items-center justify-center overflow-hidden">
                    <img src={image} alt={title} className="max-w-full max-h-full object-contain" />
                </div>
            ) : (
                <div className="w-full h-52 bg-[color:var(--color-neutral)] flex items-center justify-center">
                    <span className="text-helper text-[color:var(--color-text-tertiary)]">暂无图片</span>
                </div>
            )}
            <div className="p-3">
                <p className="text-helper font-semibold text-[color:var(--color-text-primary)]">{title}</p>
                <p className="text-helper text-[color:var(--color-text-tertiary)]">{hasImage ? '点击放大' : '无图像'}</p>
            </div>
        </button>
    );
}

function EditorField({ label, value, onChange, rows }: { label: string; value: string; onChange: (value: string) => void; rows: number }) {
    return (
        <div>
            <label className="block text-body font-semibold text-[color:var(--color-text-primary)] mb-2">{label}</label>
            <textarea className="input-base" rows={rows} value={value} onChange={(e) => onChange(e.target.value)} />
        </div>
    );
}

function OpinionBlock({ title, content }: { title: string; content: string }) {
    return (
        <div>
            <h3 className="text-body font-semibold text-[color:var(--color-text-primary)] mb-2">{title}</h3>
            <p className="text-body text-[color:var(--color-text-secondary)] leading-relaxed">{content}</p>
        </div>
    );
}
