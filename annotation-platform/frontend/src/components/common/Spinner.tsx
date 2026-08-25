// src/components/common/Spinner.tsx
import { Box, CircularProgress, Typography } from '@mui/material';

export default function Spinner({ text = '加载中...' }: { text?: string }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 1.5 }}>
      <CircularProgress size={20} sx={{ color: 'primary.main', opacity: 0.7 }} />
      <Typography variant="body2" color="text.secondary" sx={{ fontSize: 11, letterSpacing: '0.03em' }}>{text}</Typography>
    </Box>
  );
}
