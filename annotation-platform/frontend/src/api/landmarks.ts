// src/api/landmarks.ts
import { apiGet, apiPost, apiPut } from './client';
import type { Landmarks, Point3D, ValidationIssue } from '../types';

export function fetchLandmarks(id: string): Promise<Landmarks> {
  return apiGet<Landmarks>(`/subjects/${id}/landmarks`);
}

export function saveLandmarks(id: string, landmarks: Landmarks, bypassValidation = false): Promise<{ status: string; labeling_status?: string; issues?: ValidationIssue[] }> {
  return apiPut(`/subjects/${id}/landmarks`, { landmarks, bypass_validation: bypassValidation });
}

export function resetLandmarks(id: string): Promise<Landmarks> {
  return apiPost<Landmarks>(`/subjects/${id}/landmarks/reset`);
}

export function lift(id: string, x: number, y: number): Promise<Point3D> {
  return apiPost<Point3D>(`/subjects/${id}/landmarks/lift`, { x, y });
}

export function validateLandmarks(id: string, landmarks: Landmarks): Promise<Landmarks> {
  return apiPost<any>(`/subjects/${id}/landmarks/validate`, { landmarks })
    .then((resp) => {
      // 新响应格式：{landmarks: {...}, issues: [...]}，向后兼容旧格式
      if (resp && resp.landmarks) return resp.landmarks;
      return resp;
    });
}

export function validateLandmarksWithIssues(id: string, landmarks: Landmarks): Promise<{ landmarks: Landmarks; issues: ValidationIssue[] }> {
  return apiPost<any>(`/subjects/${id}/landmarks/validate`, { landmarks })
    .then((resp) => ({
      landmarks: resp.landmarks || resp,
      issues: resp.issues || [],
    }));
}
