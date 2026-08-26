// src/stores/landmarkStore.ts
import { create } from 'zustand';
import { fetchLandmarks, saveLandmarks, resetLandmarks, lift } from '../api/landmarks';
import { fetchCurvatureMapping as getMapping } from '../api/subjects';
import { useUIStore } from './uiStore';
import type { Landmarks, Mapping, Point3D } from '../types';

/** 不依赖 React 组件树的 toast 快捷方式 */
function toast(msg: string, type: 'success' | 'info' | 'warn' | 'error' = 'error') {
  useUIStore.getState().addToast(msg, type);
}

interface LandmarkState {
  landmarks: Landmarks;
  mapping: Mapping | null;
  saving: boolean;
  loading: boolean;
  error: string | null;
  isDirty: boolean;
  currentSubjectId: string | null;
  loadLandmarks: (id: string) => Promise<void>;
  updateLandmark2D: (name: string, index: number, x: number, y: number) => void;
  updateLandmark3D: (name: string, index: number, pt: Point3D) => void;
  save: (id: string) => Promise<{ status: string; labeling_status?: string; issues?: import('../types').ValidationIssue[] }>;
  autoSave: (id: string) => Promise<void>;
  reset: (id: string) => Promise<void>;
  fetchLift: (id: string, x: number, y: number) => Promise<Point3D>;
}

export const useLandmarkStore = create<LandmarkState>((set, get) => ({
  landmarks: {},
  mapping: null,
  saving: false,
  loading: false,
  error: null,
  isDirty: false,
  currentSubjectId: null,

  loadLandmarks: async (id) => {
    // Immediately destroy old mapping + landmarks — prevents ghost rendering
    // where old landmarks show on new image during transition
    set({ loading: true, error: null, currentSubjectId: id, landmarks: {}, mapping: null });
    try {
      const [landmarks, mapping] = await Promise.all([
        fetchLandmarks(id),
        getMapping(id),
      ]);
      set({ landmarks, mapping, loading: false, error: null, isDirty: false });
    } catch (e: any) {
      const msg = e.name === 'AbortError' ? '请求超时' : e.message;
      set({ error: msg, loading: false });
      toast(`标注/曲率数据加载失败：${msg}，请重试`);
    }
  },

  updateLandmark2D: (name, index, x, y) => {
    set((state) => {
      const pts = [...(state.landmarks[name] || [])];
      // 补齐数组到所需长度（新增 landmark 时创建条目）
      while (pts.length <= index) pts.push(null);
      // x, y 是 PCA 空间坐标（PC2→X, PC1→Y），逆变换回原始 3D
      const mapping = state.mapping;
      if (mapping?.pca_mean && mapping?.pca_Vt) {
        const mean = mapping.pca_mean;
        const Vt = mapping.pca_Vt;
        const pc1 = y, pc2 = x;
        const ox = +(mean[0] + pc1 * Vt[0][0] + pc2 * Vt[1][0]).toFixed(1);
        const oy = +(mean[1] + pc1 * Vt[0][1] + pc2 * Vt[1][1]).toFixed(1);
        const oz = +(mean[2] + pc1 * Vt[0][2] + pc2 * Vt[1][2]).toFixed(1);
        pts[index] = [ox, oy, oz];
      } else {
        const existing = pts[index] as number[];
        pts[index] = [x, y, existing?.[2] ?? 0];
      }
      return { landmarks: { ...state.landmarks, [name]: pts }, isDirty: true };
    });
  },

  updateLandmark3D: (name, index, pt) => {
    set((state) => {
      const pts = [...(state.landmarks[name] || [])];
      // 补齐数组到所需长度（3D 点击放置新 landmark 时创建条目）
      while (pts.length <= index) pts.push(null);
      pts[index] = [pt.x, pt.y, pt.z];
      return { landmarks: { ...state.landmarks, [name]: pts }, isDirty: true };
    });
  },

  save: async (id) => {
    set({ saving: true, error: null });
    try {
      const result = await saveLandmarks(id, get().landmarks);
      set({ saving: false, isDirty: false });
      return result;
    } catch (e: any) {
      const msg = e.name === 'AbortError' ? '请求超时' : e.message;
      set({ error: msg, saving: false });
      toast(`标注保存失败：${msg}，请重试`);
      return { status: 'error' };
    }
  },

  autoSave: async (id) => {
    if (!id) return;
    try {
      await saveLandmarks(id, get().landmarks);
    } catch (e: any) {
      // silent auto-save
    }
  },

  reset: async (id) => {
    try {
      const landmarks = await resetLandmarks(id);
      set({ landmarks, error: null });
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  fetchLift: async (id, x, y) => {
    return lift(id, x, y);
  },
}));
