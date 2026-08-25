// api/subjects.ts
import { apiGet, apiPost, apiPut } from "./client";
import type { SubjectInfo, SubjectDetail, Mapping } from "../types";

export function fetchSubjects(): Promise<SubjectInfo[]> {
  return apiGet<SubjectInfo[]>("/subjects");
}

export function fetchSubject(id: string): Promise<SubjectDetail> {
  return apiGet<SubjectDetail>(`/subjects/${id}`);
}

export function getCurvatureImageUrl(id: string, version?: number): string {
  return `/api/subjects/${id}/curvature-image${version != null ? '?v=' + version : ''}`;
}

export function fetchCurvatureMapping(id: string): Promise<Mapping> {
  return apiGet<Mapping>(`/subjects/${id}/curvature-mapping`);
}

export function getMeshUrl(id: string, clothed?: boolean): string {
  return `/api/subjects/${id}/mesh${clothed ? "?clothed=true" : ""}`;
}

export function setLabelingStatus(id: string, status: string): Promise<{ labeling_status: string }> {
  return apiPut(`/subjects/${id}/labeling-status`, { status });
}

// ── 无状态笔刷提交 ──────────────────────────────────
export function brushCommit(
  id: string,
  body: { points?: number[][]; cloth_indices?: number[] },
): Promise<any> {
  return apiPost(`/subjects/${id}/brush/commit`, body);
}
