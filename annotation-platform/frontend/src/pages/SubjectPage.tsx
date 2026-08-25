// src/pages/SubjectPage.tsx
import { useEffect, useCallback, useState, useRef } from 'react';
import { Box } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import LeftPanel from '../components/panels/LeftPanel';
import RightPanel from '../components/panels/RightPanel';
import ResizeHandle from '../components/layout/ResizeHandle';
import Curvature2D from '../components/viewer/Curvature2D';
import Viewer3D from '../components/viewer/Viewer3D';
import Toolbar from '../components/toolbar/Toolbar';
import { useSubjectStore } from '../stores/subjectStore';
import { useLandmarkStore } from '../stores/landmarkStore';
import { useUIStore } from '../stores/uiStore';
import { LAST_SUBJECT_KEY } from '../App';

export default function SubjectPage() {
  const { id, viewMode: viewModeParam } = useParams<{ id: string; viewMode?: string }>();
  const navigate = useNavigate();
  const { loadSubjects, selectSubject, currentId } = useSubjectStore();
  const { loadLandmarks } = useLandmarkStore();
  const { viewMode, leftWidth, rightWidth, setLeftWidth, setRightWidth, addToast, setViewMode, clearBrushPoints, clearRestoreCloth, setBrushMode } = useUIStore();
  const prevId = useRef<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Track which view+mutation the current loading covers; only clear when match
  const loadingView = useRef<'2d' | '3d' | null>(null);

  // Sync URL viewMode param to store on mount/param change
  // Sync URL viewMode param to store on mount/param change
  useEffect(() => {
    if (viewModeParam === '2d' || viewModeParam === '3d') {
      setViewMode(viewModeParam);
    }
  }, [viewModeParam, setViewMode]);

  // Redirect to include default viewMode when URL has no viewMode param
  useEffect(() => {
    if (id && !viewModeParam) {
      navigate(`/subject/${encodeURIComponent(id)}/2d`, { replace: true });
    }
  }, [id, viewModeParam, navigate]);

  useEffect(() => {
    loadSubjects();
  }, [loadSubjects]);

  const loadSubjectData = useCallback(async (subjectId: string) => {
    if (!subjectId) return;
    try {
      await Promise.all([
        selectSubject(subjectId),
        loadLandmarks(subjectId),
      ]);
      localStorage.setItem(LAST_SUBJECT_KEY, subjectId);
    } catch (e: any) {
      addToast('加载失败: ' + (e.message || e), 'error');
    }
  }, [selectSubject, loadLandmarks, addToast]);

  const handleViewModeChange = useCallback((mode: '2d' | '3d') => {
    setViewMode(mode);
    setLoading(true);
    loadingView.current = mode;
    if (id) navigate(`/subject/${encodeURIComponent(id)}/${mode}`, { replace: true });
  }, [setViewMode, id, navigate]);

  const handleReady = useCallback((view: '2d' | '3d') => {
    console.log('SubjectPage: handleReady view=' + view + ' loadingView.current=' + loadingView.current + ' loading=' + loading);
    if (loadingView.current === view) {
      setLoading(false);
      loadingView.current = null;
    }
  }, []);

  useEffect(() => {
    if (id && id !== prevId.current) {
      prevId.current = id;
      setLoading(true);
      // Use URL param directly — store viewMode may not be synced yet (defaults to '2d')
      loadingView.current = viewModeParam === '3d' ? '3d' : '2d';
      loadSubjectData(id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const exitBrush = useCallback(() => {
    clearBrushPoints();
    clearRestoreCloth();
    setBrushMode(null);
  }, [clearBrushPoints, clearRestoreCloth, setBrushMode]);

  // 切换 subject 时自动清空擦除痕迹并退出擦除模式
  useEffect(() => { exitBrush(); }, [id, exitBrush]);

  // 切换到 2D 视图时自动清空擦除痕迹并退出擦除模式
  useEffect(() => {
    if (viewMode === '2d') exitBrush();
  }, [viewMode, exitBrush]);

  return (
    <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'background.default' }}>
      {/* Left */}
      <Box sx={{ width: leftWidth, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
        <LeftPanel />
      </Box>
      <ResizeHandle onResize={(delta) => setLeftWidth(Math.max(100, Math.min(500, leftWidth + delta)))} />

      {/* Center */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative', bgcolor: 'background.default' }}>
        {/* Title bar */}
        <Box sx={{ px: 1.5, py: 0.5, borderBottom: 1, borderColor: 'divider', bgcolor: '#0C1120', display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 3, height: 14, borderRadius: 0.5, bgcolor: 'primary.main' }} />
          <Box component="span" sx={{ fontSize: 12, fontWeight: 600, color: 'text.secondary', letterSpacing: 0.3 }}>
            {viewMode === '2d' ? '曲率热力图' : '3D 视图'}
          </Box>
          <Box sx={{ flex: 1 }} />
          <Box component="span" sx={{ fontSize: 10, color: 'rgba(148,163,184,0.4)', fontFamily: '"JetBrains Mono", monospace', letterSpacing: '0.02em' }}>
            {currentId || ''}
          </Box>
        </Box>
        {/* Canvas / 3D area */}
        <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden', bgcolor: 'background.default' }}>
          {!currentId ? (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 1.5 }}>
              <Box sx={{ width: 24, height: 24, border: '1.5px solid', borderColor: 'rgba(148,163,184,0.15)', borderTopColor: 'primary.main', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              <Box component="span" sx={{ color: 'text.secondary', fontSize: 12, letterSpacing: '0.02em' }}>选择 subject...</Box>
            </Box>
          ) : viewMode === '2d' ? (
            <Curvature2D onSwitch3D={() => handleViewModeChange('3d')} onReady={() => handleReady('2d')} />
          ) : (
            <Viewer3D onReady={() => handleReady('3d')} />
          )}
        </Box>
        {/* Toolbar */}
        <Toolbar onSwitchView={() => handleViewModeChange(viewMode === '2d' ? '3d' : '2d')} />

        {/* Loading overlay — subtle blur + spinner */}
        {loading && (
          <Box sx={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            bgcolor: 'rgba(8, 12, 24, 0.85)',
            backdropFilter: 'blur(2px)',
            zIndex: 999,
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 1.5,
          }}>
            <Box sx={{ width: 28, height: 28, border: '2px solid', borderColor: 'rgba(148,163,184,0.15)', borderTopColor: 'primary.main', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            <Box component="span" sx={{ color: 'text.secondary', fontSize: 12, letterSpacing: '0.02em' }}>加载中...</Box>
          </Box>
        )}
      </Box>

      {/* Right */}
      <ResizeHandle onResize={(delta) => setRightWidth(Math.max(150, Math.min(500, rightWidth - delta)))} />
      <Box sx={{ width: rightWidth, flexShrink: 0 }}>
        <RightPanel />
      </Box>
    </Box>
  );
}
