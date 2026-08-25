// src/App.tsx
import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, Box } from '@mui/material';
import SubjectPage from './pages/SubjectPage';
import ToastProvider from './components/common/Toast';
import { fetchSubjects } from './api/subjects';
import type { SubjectInfo } from './types';
import { apiGet } from './api/client';

const LAST_SUBJECT_KEY = 'labeling_platform_last_subject';

function AnnotationGuard({ children }: { children: React.ReactNode }) {
  const params = new URLSearchParams(window.location.search);
  const token = params.get('token');
  const reportId = params.get('reportId');
  const returnUrl = params.get('returnUrl');
  const subjectId = window.location.pathname.split('/')[2];
  const [state, setState] = useState<'checking' | 'allowed' | 'denied'>(token && reportId && subjectId ? 'checking' : 'denied');

  useEffect(() => {
    if (!token || !reportId || !subjectId) return;
    apiGet(`/annotation-session/validate?token=${encodeURIComponent(token)}&reportId=${encodeURIComponent(reportId)}&subjectId=${encodeURIComponent(subjectId)}`)
      .then(() => setState('allowed'))
      .catch(() => setState('denied'));
  }, [token, reportId, subjectId]);

  if (state === 'checking') return <Box sx={{ height: '100vh', display: 'grid', placeItems: 'center', bgcolor: 'background.default', color: 'text.secondary' }}>正在验证标注授权…</Box>;
  if (state === 'denied') return <Box sx={{ height: '100vh', display: 'grid', placeItems: 'center', bgcolor: 'background.default', color: 'error.main' }}>无有效授权，无法打开标注工具。</Box>;
  return <>
    {children}
    {returnUrl && <Box component="button" onClick={() => window.location.assign(returnUrl)} sx={{ position: 'fixed', right: 20, top: 16, zIndex: 2000, border: 1, borderColor: 'divider', borderRadius: 1, px: 1.5, py: 0.75, bgcolor: 'background.paper', color: 'text.primary', cursor: 'pointer', '&:hover': { borderColor: 'primary.main' } }}>返回报告详情</Box>}
  </>;
}

/* 精密医疗仪器深色主题 — V2 */
/* 更深邃、更克制：去掉装饰性动效，用精确的间距和层级体现品质 */
const VISUAL_THEME = {
  bg: '#080C18',        // 主背景（更深的靛蓝）
  surface: '#0E1325',   // 面板表面
  border: '#1A2237',    // 边框
  elevated: '#1C2942',  // 悬浮/高亮表面
  accent: '#38BDF8',    // 强调色（天空蓝）
  purple: '#818CF8',    // 二次强调色（淡紫）
  green: '#4ADE80',     // 成功/完成
  red: '#FB7185',       // 错误/擦除
  amber: '#FBBF24',     // 警告
  text: '#F1F5F9',      // 主文字
  textDim: '#94A3B8',   // 次要文字
  textMuted: '#475569', // 弱文字（比之前更低调）
};

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: VISUAL_THEME.accent },
    secondary: { main: VISUAL_THEME.purple },
    error: { main: VISUAL_THEME.red },
    warning: { main: VISUAL_THEME.amber },
    background: { default: VISUAL_THEME.bg, paper: VISUAL_THEME.surface },
    text: { primary: VISUAL_THEME.text, secondary: VISUAL_THEME.textDim },
    divider: VISUAL_THEME.border,
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    fontSize: 13,
    caption: { fontFamily: '"JetBrains Mono", monospace' },
  },
  shape: { borderRadius: 6 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          fontFeatureSettings: '"cv02", "cv03", "cv04", "cv11"',
          '& ::-webkit-scrollbar': { width: 4, height: 4 },
          '& ::-webkit-scrollbar-track': { background: 'transparent' },
          '& ::-webkit-scrollbar-thumb': { background: '#1E2A45', borderRadius: 4 },
          '& ::-webkit-scrollbar-thumb:hover': { background: '#2D3F60' },
          '& ::selection': { background: 'rgba(56, 189, 248, 0.25)' },
        },
        '@keyframes spin': {
          to: { transform: 'rotate(360deg)' },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          fontSize: 12,
          minHeight: 28,
          borderRadius: 5,
          borderColor: VISUAL_THEME.border,
          transition: 'all 0.12s ease',
          '&:hover': {
            borderColor: VISUAL_THEME.accent,
          },
        },
        contained: {
          boxShadow: 'none',
          '&:hover': { boxShadow: 'none' },
        },
        outlined: { borderColor: VISUAL_THEME.border },
        sizeSmall: { padding: '2px 10px' },
      },
    },
    MuiToggleButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontSize: 11,
          fontWeight: 500,
          borderColor: VISUAL_THEME.border,
          transition: 'all 0.12s ease',
          '&.Mui-selected': {
            bgcolor: `${VISUAL_THEME.accent}14`,
            color: VISUAL_THEME.accent,
            borderColor: VISUAL_THEME.accent,
          },
          '&:hover': { borderColor: VISUAL_THEME.accent },
        },
      },
    },
    MuiToggleButtonGroup: {
      styleOverrides: {
        root: { borderColor: VISUAL_THEME.border },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-notchedOutline': { borderColor: VISUAL_THEME.border },
          '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: VISUAL_THEME.accent },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: VISUAL_THEME.accent,
            borderWidth: 1,
          },
          '&.Mui-focused': {
            boxShadow: `0 0 0 2px ${VISUAL_THEME.accent}1A`,
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { fontSize: 10, height: 20 },
        outlined: { borderColor: VISUAL_THEME.border },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          bgcolor: 'transparent',
          '&.Mui-expanded': { margin: 0 },
          '&:before': { display: 'none' },
          borderTop: `1px solid ${VISUAL_THEME.border}`,
        },
      },
    },
    MuiAccordionSummary: {
      styleOverrides: { root: { minHeight: 32, '&.Mui-expanded': { minHeight: 32 } } },
    },
    MuiSlider: {
      styleOverrides: {
        rail: { backgroundColor: VISUAL_THEME.border },
        thumb: {
          boxShadow: `0 0 0 4px ${VISUAL_THEME.accent}1A`,
          '&:hover': { boxShadow: `0 0 0 6px ${VISUAL_THEME.accent}26` },
        },
      },
    },
    MuiTypography: {
      styleOverrides: {
        root: { letterSpacing: '0.01em' },
        subtitle2: { letterSpacing: '0.03em' },
        caption: { letterSpacing: '0.02em' },
      },
    },
    MuiDivider: {
      styleOverrides: { root: { borderColor: VISUAL_THEME.border } },
    },
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: 'none' },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          backgroundImage: 'none',
          bgcolor: VISUAL_THEME.surface,
          border: `1px solid ${VISUAL_THEME.border}`,
          boxShadow: '0 32px 64px rgba(0,0,0,0.6)',
          '& .MuiDialogTitle-root': { fontSize: 15, pb: 0.5, letterSpacing: '0.02em' },
          '& .MuiDialogContent-root': { pt: 1.5 },
          '& .MuiDialogActions-root': { px: 2, pb: 1.5 },
        },
      },
    },
    MuiList: {
      styleOverrides: {
        root: {
          '& .MuiListItemButton-root': {
            borderRadius: 4,
            marginBottom: 1,
            transition: 'all 0.1s ease',
          },
          '& .MuiListItemButton-root.Mui-selected': {
            borderLeft: `2px solid ${VISUAL_THEME.accent}`,
            bgcolor: `${VISUAL_THEME.accent}0A`,
            '&:hover': { bgcolor: `${VISUAL_THEME.accent}14` },
          },
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          bgcolor: VISUAL_THEME.elevated,
          color: VISUAL_THEME.text,
          fontSize: 11,
          border: `1px solid ${VISUAL_THEME.border}`,
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        icon: { color: VISUAL_THEME.textDim },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          backgroundImage: 'none',
          bgcolor: VISUAL_THEME.elevated,
          border: `1px solid ${VISUAL_THEME.border}`,
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          fontSize: 12,
          '&:hover': { bgcolor: `${VISUAL_THEME.accent}0F` },
        },
      },
    },
  },
});

function HomeRedirect() {
  const [target, setTarget] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem(LAST_SUBJECT_KEY);
    if (saved) {
      setTarget(`/subject/${encodeURIComponent(saved)}/2d`);
      return;
    }
    // No saved subject: load subjects and find first verified one
    fetchSubjects().then((subjects: SubjectInfo[]) => {
      const labeled = subjects.find((s) => s.labeling_status === 'labeled');
      if (labeled) {
        setTarget(`/subject/${encodeURIComponent(labeled.id)}/2d`);
      } else if (subjects.length > 0) {
        setTarget(`/subject/${encodeURIComponent(subjects[0].id)}/2d`);
      } else {
        // Fallback — shouldn't happen normally
        setTarget('/subject/auto');
      }
    }).catch(() => {
      setTarget('/subject/auto');
    });
  }, []);

  if (!target) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', bgcolor: 'background.default' }}>
        <Box sx={{ width: 28, height: 28, border: '2px solid', borderColor: 'divider', borderTopColor: 'primary.main', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
      </Box>
    );
  }

  return <Navigate to={target} replace />;
}

export default function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <BrowserRouter>
        <ToastProvider />
        <Routes>
          <Route path="/" element={<HomeRedirect />} />
          <Route path="/subject/:id" element={<AnnotationGuard><SubjectPage /></AnnotationGuard>} />
          <Route path="/subject/:id/:viewMode" element={<AnnotationGuard><SubjectPage /></AnnotationGuard>} />
          <Route path="*" element={<div style={{ padding: 20, color: '#888' }}>404 Not Found</div>} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export { LAST_SUBJECT_KEY };
