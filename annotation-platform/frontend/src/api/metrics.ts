import { apiGet, apiPost } from './client';
import type { MetricsResults, ValidationResult, ClinicalData, ExportTaskStatus } from '../types';

export function fetchMetrics(id: string): Promise<MetricsResults> {
  return apiGet<MetricsResults>(`/subjects/${id}/metrics`);
}

export function fetchValidate(id: string): Promise<ValidationResult> {
  return apiGet<ValidationResult>(`/subjects/${id}/validate`);
}

export function batchGenerate(): Promise<{ status: string; count: number }> {
  return apiPost('/batch/generate');
}

export function fetchClinicalData(): Promise<ClinicalData> {
  return apiGet<ClinicalData>('/clinical-data');
}

export function startDataExport(): Promise<{ task_id: string } | { error: string }> {
  return apiPost('/export/data-export');
}

export function getDataExportStatus(taskId: string): Promise<ExportTaskStatus> {
  return apiGet<ExportTaskStatus>(`/export/data-export/${taskId}`);
}
