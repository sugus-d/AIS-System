// src/components/common/Toast.tsx
import { Snackbar, Alert } from '@mui/material';
import { useUIStore } from '../../stores/uiStore';

export default function ToastProvider() {
  const toasts = useUIStore((s) => s.toasts);
  const removeToast = useUIStore((s) => s.removeToast);
  const last = toasts[toasts.length - 1];

  if (!last) return null;

  const severityMap: Record<string, 'success' | 'info' | 'warning' | 'error'> = {
    success: 'success', info: 'info', warn: 'warning', error: 'error',
  };

  return (
    <Snackbar
      open={!!last}
      autoHideDuration={3000}
      onClose={() => removeToast(last.id)}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      sx={{ '& .MuiAlert-root': { backdropFilter: 'blur(16px)', bgcolor: 'rgba(14, 19, 37, 0.92)', border: '1px solid', borderColor: 'divider', boxShadow: '0 12px 40px rgba(0,0,0,0.6)', borderRadius: 2 } }}
    >
      <Alert severity={severityMap[last.type] || 'info'} onClose={() => removeToast(last.id)} variant="outlined" sx={{ fontSize: 12, '& .MuiAlert-icon': { fontSize: 18, mr: 1 } }}>
        {last.message}
      </Alert>
    </Snackbar>
  );
}
