// src/stores/uiStore.ts
import { create } from "zustand";
import type { Toast, ToastType, ValidationIssue } from "../types";

const PANEL_WIDTH_KEY = "labeling_platform_panel_widths";

interface UIState {
  viewMode: "2d" | "3d";
  leftWidth: number;
  rightWidth: number;
  brushMode: "erase" | "restore" | null;
  brushSize: number;
  rotationAngle: number;
  rotationMode: boolean;
  toasts: Toast[];
  detailPanelOpen: boolean;
  overlayVisible: boolean;
  clothOverlay: boolean;
  brushPoints: number[][]; // frontend brush (erase) marks
  restorePoints: number[][]; // frontend restore marks (取消擦除)
  restoreClothPoints: number[][]; // 服装恢复标记（绿色）3D坐标
  restoreClothIndices: number[]; // 对应 cloth_verts 索引
  meshVersion: number; // commit 后自增，驱动 3D mesh 重新加载
  pendingLandmark: { name: string; index: number } | null; // 待放置的 landmark
  validationIssues: ValidationIssue[];
  setValidationIssues: (issues: ValidationIssue[]) => void;
  setPendingLandmark: (v: { name: string; index: number } | null) => void;
  setViewMode: (mode: "2d" | "3d") => void;
  setLeftWidth: (w: number) => void;
  setRightWidth: (w: number) => void;
  setBrushMode: (mode: "erase" | "restore" | null) => void;
  setBrushSize: (size: number) => void;
  setRotationAngle: (deg: number) => void;
  setRotationMode: (v: boolean) => void;
  addToast: (message: string, type: ToastType) => void;
  removeToast: (id: string) => void;
  toggleDetailPanel: () => void;
  setOverlayVisible: (v: boolean) => void;
  setClothOverlay: (v: boolean) => void;
  addBrushPoints: (pts: number[][]) => void; // 覆盖式写入（替换而非追加）
  clearBrushPoints: () => void;
  addRestorePoints: (pts: number[][]) => void;
  clearRestorePoints: () => void;
  setRestoreCloth: (pts: number[][], indices: number[]) => void;
  clearRestoreCloth: () => void;
  incrementMeshVersion: () => void;
}

function loadPanelWidths(): { left: number; right: number } {
  try {
    const raw = localStorage.getItem(PANEL_WIDTH_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        left: typeof parsed.left === "number" ? parsed.left : 200,
        right: typeof parsed.right === "number" ? parsed.right : 250,
      };
    }
  } catch {
    /* ignore */
  }
  return { left: 200, right: 250 };
}

function savePanelWidths(left: number, right: number) {
  try {
    localStorage.setItem(PANEL_WIDTH_KEY, JSON.stringify({ left, right }));
  } catch {
    /* ignore */
  }
}

const widths = loadPanelWidths();
let toastId = 0;

export const useUIStore = create<UIState>((set) => ({
  viewMode: "2d",
  leftWidth: widths.left,
  rightWidth: widths.right,
  brushMode: null,
  brushSize: 35,
  rotationAngle: 0,
  rotationMode: false,
  toasts: [],
  detailPanelOpen: false,
  overlayVisible: false,
  clothOverlay: false,
  brushPoints: [],
  restorePoints: [],
  restoreClothPoints: [],
  restoreClothIndices: [],
  meshVersion: 0,
  pendingLandmark: null,
  validationIssues: [],

  setViewMode: (viewMode) => set({ viewMode }),
  setLeftWidth: (leftWidth) => {
    set({ leftWidth });
    set((s) => {
      savePanelWidths(leftWidth, s.rightWidth);
      return {};
    });
  },
  setRightWidth: (rightWidth) => {
    set({ rightWidth });
    set((s) => {
      savePanelWidths(s.leftWidth, rightWidth);
      return {};
    });
  },
  setBrushMode: (brushMode) => set({ brushMode }),
  setBrushSize: (brushSize) =>
    set({ brushSize: Math.max(5, Math.min(80, brushSize)) }),
  setRotationAngle: (rotationAngle) => set({ rotationAngle }),
  setRotationMode: (rotationMode) => set({ rotationMode }),
  addToast: (message, type) => {
    const id = String(++toastId);
    set((s) => ({ toasts: [...s.toasts, { id, message, type }] }));
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, 3000);
  },
  removeToast: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  toggleDetailPanel: () =>
    set((s) => ({ detailPanelOpen: !s.detailPanelOpen })),
  setOverlayVisible: (overlayVisible) => set({ overlayVisible }),
  setClothOverlay: (clothOverlay) => set({ clothOverlay }),
  addBrushPoints: (pts) =>
    set((s) => ({ brushPoints: [...s.brushPoints, ...pts] })),
  clearBrushPoints: () => set({ brushPoints: [] }),
  addRestorePoints: (pts) =>
    set((s) => ({ restorePoints: [...s.restorePoints, ...pts] })),
  clearRestorePoints: () => set({ restorePoints: [] }),
  setRestoreCloth: (pts, indices) =>
    set({ restoreClothPoints: pts, restoreClothIndices: indices }),
  clearRestoreCloth: () =>
    set({ restoreClothPoints: [], restoreClothIndices: [] }),
  incrementMeshVersion: () => set((s) => ({ meshVersion: s.meshVersion + 1 })),
  setPendingLandmark: (v) => set({ pendingLandmark: v }),
  setValidationIssues: (validationIssues) => set({ validationIssues }),
}));
