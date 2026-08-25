export type WorkflowTaskStatus = "pending" | "running" | "success" | "failed" | "cancelled" | "partial_success";

export type WorkflowTaskType =
    | "single_record"
    | "batch_record"
    | "single_upload"
    | "batch_upload"
    | "single_analysis"
    | "batch_analysis";

export type WorkflowChildTask = {
    id: string;
    status: WorkflowTaskStatus;
    relatedData: string;
    submitTime?: string;
    endTime?: string;
    errorReason?: string;
    reportNumber?: string;
    cobbAngle?: number;
    severity?: string;
    filePath?: string;
};

export type WorkflowTask = {
    id: string;
    taskType: WorkflowTaskType;
    status: WorkflowTaskStatus;
    submitTime: string;
    submitUser: string;
    relatedData: string;
    progress?: number;
    errorReason?: string;
    duration?: number;
    endTime?: string;
    isBatch?: boolean;
    childTasks?: WorkflowChildTask[];
};

export type WorkflowReport = {
    id: string;
    caseId: string;
    caseNumber: string;
    patientName: string;
    gender: "M" | "F";
    fileNumber: string;
    scanTime: string;
    reportNumber: string;
    reportTime: string;
    cobbAngle: number;
    severity: "negative" | "mild" | "moderate" | "severe";
    doctor: string;
    department: string;
    operator: string;
};

const TASKS_KEY = "ais_linked_tasks";
const REPORTS_KEY = "ais_linked_reports";

function readLocalArray<T>(key: string): T[] {
    if (typeof window === "undefined") return [];
    const raw = window.localStorage.getItem(key);
    if (!raw) return [];
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? (parsed as T[]) : [];
    } catch {
        return [];
    }
}

function writeLocalArray<T>(key: string, data: T[]) {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(key, JSON.stringify(data));
}

export function getLinkedTasks() {
    return readLocalArray<WorkflowTask>(TASKS_KEY);
}

export function upsertLinkedTask(task: WorkflowTask) {
    const all = getLinkedTasks();
    const idx = all.findIndex((item) => item.id === task.id);
    if (idx >= 0) {
        all[idx] = task;
    } else {
        all.unshift(task);
    }
    writeLocalArray(TASKS_KEY, all);
}

export function getLinkedReports() {
    return readLocalArray<WorkflowReport>(REPORTS_KEY);
}

export function appendLinkedReports(reports: WorkflowReport[]) {
    if (reports.length === 0) return;
    const current = getLinkedReports();
    const merged = [...reports, ...current];
    const deduped = merged.filter(
        (item, index, arr) => arr.findIndex((r) => r.reportNumber === item.reportNumber) === index,
    );
    writeLocalArray(REPORTS_KEY, deduped);
}
