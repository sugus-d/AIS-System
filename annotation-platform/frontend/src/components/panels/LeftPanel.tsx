// src/components/panels/LeftPanel.tsx
import { useEffect, useState, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, TextField, ToggleButtonGroup, ToggleButton, List, ListItemButton, ListItemText, Chip, Typography, Button, CircularProgress } from '@mui/material';
import { useSubjectStore } from '../../stores/subjectStore';
import { useUIStore } from '../../stores/uiStore';
import { startDataExport, getDataExportStatus } from '../../api/metrics';
import type { SubjectInfo } from '../../types';

const FILTERS = [
  { value: 'all', label: '全部' },
  { value: 'unlabeled', label: '未标' },
  { value: 'prelabeled', label: '预标' },
  { value: 'labeled', label: '已标' },
] as const;

type FilterType = 'all' | 'unlabeled' | 'prelabeled' | 'labeled';

const STATUS_META: Record<string, { label: string; color: 'default' | 'warning' | 'success'; chipColor: 'default' | 'warning' | 'success' }> = {
  unlabeled: { label: '未标', color: 'default', chipColor: 'default' },
  prelabeled: { label: '预标', color: 'warning', chipColor: 'warning' },
  labeled: { label: '已标', color: 'success', chipColor: 'success' },
};

export default function LeftPanel() {
  const navigate = useNavigate();
  const { subjects, currentId, filter, searchQuery, loading, pinnedSubjects, loadSubjects, selectSubject, setFilter, setSearchQuery } = useSubjectStore();
  const viewMode = useUIStore((s) => s.viewMode);
  const [initialized, setInitialized] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState('');

  useEffect(() => {
    if (!initialized) {
      loadSubjects();
      setInitialized(true);
    }
  }, [initialized, loadSubjects]);

  // 当前 subject 变化时同步 filter；首次加载 subjects 后也同步
  const syncedSubject = useRef<string | null>(null);
  useEffect(() => {
    if (!currentId || subjects.length === 0) return;
    if (syncedSubject.current === currentId && subjects.length > 0) return;
    const subj = subjects.find((s) => s.id === currentId);
    if (!subj || !subj.labeling_status) return;
    setFilter(subj.labeling_status);
    syncedSubject.current = currentId;
  }, [currentId, subjects]);

  const counts = useMemo(() => {
    const total = subjects.length;
    let labeled = 0, prelabeled = 0, unlabeled = 0;
    for (const s of subjects) {
      if (s.labeling_status === 'labeled') labeled++;
      else if (s.labeling_status === 'prelabeled') prelabeled++;
      else if (s.labeling_status === 'unlabeled') unlabeled++;
    }
    return { total, labeled, prelabeled, unlabeled };
  }, [subjects]);

  const filtered = useMemo(() => {
    const pinnedSet = new Set(pinnedSubjects);
    const pinned: SubjectInfo[] = [];
    const normal: SubjectInfo[] = [];
    for (const s of subjects) {
      const matchSearch = !searchQuery || s.id.toLowerCase().includes(searchQuery.toLowerCase());
      if (!matchSearch) continue;
      const matchFilter = filter === 'all' || s.labeling_status === filter;
      const isPinned = pinnedSet.has(s.id);
      if (isPinned || matchFilter) {
        (isPinned ? pinned : normal).push(s);
      }
    }
    // 置顶的按原顺序排前面
    return [...pinned, ...normal];
  }, [subjects, filter, searchQuery, pinnedSubjects]);

  const getBadge = (s: SubjectInfo) => {
    const meta = STATUS_META[s.labeling_status] || STATUS_META.unlabeled;
    return { label: meta.label, color: meta.chipColor as 'default' | 'warning' | 'success' };
  };

  const handleFilterChange = (_: any, newFilter: FilterType | null) => {
    if (newFilter) {
      useSubjectStore.setState({ pinnedSubjects: [] });
      setFilter(newFilter);
    }
  };

  const filterButtons = [
    { value: 'all', label: `全部 ${counts.total}` },
    { value: 'unlabeled', label: `未标 ${counts.unlabeled}` },
    { value: 'prelabeled', label: `预标 ${counts.prelabeled}` },
    { value: 'labeled', label: `已标 ${counts.labeled}` },
  ];

  // Export data button - only on "labeled" filter
  const handleExportData = async () => {
    setExporting(true);
    setExportProgress('启动中...');
    try {
      const resp = await startDataExport() as any;
      if (!resp.task_id) throw new Error(resp.error || '启动失败');
      const taskId = resp.task_id;
      let polled = false;
      const poll = setInterval(async () => {
        try {
          const prog = await getDataExportStatus(taskId) as any;
          if (prog.error) { clearInterval(poll); throw new Error(prog.error); }
          setExportProgress(`⏳ ${prog.done}/${prog.total}`);
          if (prog.status === 'done') {
            clearInterval(poll); polled = true;
            setExporting(false);
            setExportProgress('');
          } else if (prog.status === 'error') {
            clearInterval(poll); polled = true;
            throw new Error(prog.error || '导出出错');
          }
        } catch (e: any) {
          if (!polled) { clearInterval(poll); polled = true; }
          setExporting(false);
          setExportProgress('');
        }
      }, 2000);
    } catch {
      setExporting(false);
      setExportProgress('');
    }
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', bgcolor: 'background.paper', borderRight: 1, borderColor: 'divider' }}>
      <Box sx={{ px: 1.5, py: 1, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', gap: 1 }}>
        <Box sx={{ width: 3, height: 14, borderRadius: 0.5, bgcolor: 'primary.main' }} />
        <Typography variant="subtitle2" sx={{ color: 'text.secondary', fontWeight: 600, letterSpacing: '0.04em', fontSize: 11 }}>
          数据列表
        </Typography>
      </Box>

      <ToggleButtonGroup size="small" exclusive value={filter} onChange={handleFilterChange} sx={{ mx: 1, my: 0.5, display: 'flex', gap: 0.5, '& .MuiToggleButton-root': { flex: 1, px: 0.5 } }}>
        {filterButtons.map((b) => (
          <ToggleButton key={b.value} value={b.value}>{b.label}</ToggleButton>
        ))}
      </ToggleButtonGroup>

      <TextField
        size="small" placeholder="搜索 Subject ID..." value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        sx={{ mx: 1, mb: 1, '& input': { fontSize: 12 } }}
      />

      {/* Export button - only on "labeled" filter */}
      {filter === 'labeled' && (
        <Button size="small" variant="outlined" onClick={handleExportData} disabled={exporting}
          sx={{ mx: 1, mb: 1, fontSize: 11 }}>
          {exporting ? (exportProgress || '导出中...') : '导出标注数据'}
        </Button>
      )}

      {loading && subjects.length === 0 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress size={20} />
        </Box>
      ) : (
        <List dense sx={{ flex: 1, overflow: 'auto', px: 1 }}>
          {filtered.map((s) => {
            const b = getBadge(s);
            return (
              <ListItemButton
                key={s.id}
                selected={s.id === currentId}
                onClick={() => navigate(`/subject/${encodeURIComponent(s.id)}/${viewMode}`)}
                sx={{ borderRadius: 1, mb: 0.5, fontSize: 12, color: s.has_cache ? 'inherit' : 'rgba(148,163,184,0.4)' }}
              >
                <ListItemText primary={s.id} primaryTypographyProps={{ fontSize: 12 }} />
                <Chip label={b.label} size="small" color={b.color} variant="outlined" sx={{ fontSize: 10 }} />
              </ListItemButton>
            );
          })}
        </List>
      )}
    </Box>
  );
}
