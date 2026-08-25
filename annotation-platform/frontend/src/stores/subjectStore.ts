// src/stores/subjectStore.ts
import { create } from 'zustand';
import { fetchSubjects, fetchSubject, setLabelingStatus } from '../api/subjects';
import type { SubjectInfo, SubjectDetail } from '../types';

interface SubjectState {
  subjects: SubjectInfo[];
  currentId: string | null;
  currentDetail: SubjectDetail | null;
  loading: boolean;
  error: string | null;
  filter: 'all' | 'unlabeled' | 'prelabeled' | 'labeled';
  searchQuery: string;
  /** 保存后临时置顶的 subject，切换 filter 后取消 */
  pinnedSubjects: string[];
  loadSubjects: () => Promise<void>;
  selectSubject: (id: string) => Promise<void>;
  setFilter: (filter: SubjectState['filter']) => void;
  setSearchQuery: (q: string) => void;
  /** 本地更新 subject 的状态（不重新拉列表） */
  patchStatus: (id: string, status: SubjectInfo['labeling_status']) => void;
  setManualStatus: (id: string, status: string) => Promise<void>;
}

export const useSubjectStore = create<SubjectState>((set) => ({
  subjects: [],
  currentId: null,
  currentDetail: null,
  loading: false,
  error: null,
  filter: 'all',
  searchQuery: '',
  pinnedSubjects: [],

  loadSubjects: async () => {
    set({ loading: true, error: null });
    try {
      const subjects = await fetchSubjects();
      set({ subjects, loading: false });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  selectSubject: async (id) => {
    set({ currentId: id, loading: true, error: null });
    try {
      const detail = await fetchSubject(id);
      set({ currentDetail: detail, loading: false });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  setFilter: (filter) => set({ filter }),
  setSearchQuery: (searchQuery) => set({ searchQuery }),

  patchStatus: (id, status) => {
    set((state) => ({
      subjects: state.subjects.map((s) =>
        s.id === id ? { ...s, labeling_status: status } : s
      ),
      currentDetail: state.currentDetail?.id === id
        ? { ...state.currentDetail, labeling_status: status }
        : state.currentDetail,
      pinnedSubjects: [id, ...state.pinnedSubjects.filter((x) => x !== id)],
    }));
  },

  setManualStatus: async (id, status) => {
    try {
      await setLabelingStatus(id, status);
      set((state) => ({
        subjects: state.subjects.map((s) =>
          s.id === id ? { ...s, labeling_status: status as SubjectInfo['labeling_status'] } : s
        ),
        currentDetail: state.currentDetail?.id === id
          ? { ...state.currentDetail, labeling_status: status as SubjectInfo['labeling_status'] }
          : state.currentDetail,
      }));
    } catch {
      // 静默失败
    }
  },
}));
