// src/components/panels/SubjectInfo.tsx
import { Box, Typography } from '@mui/material';
import { useSubjectStore } from '../../stores/subjectStore';

export default function SubjectInfo() {
  const { currentId, currentDetail, error } = useSubjectStore();
  if (!currentId) return null;
  const d = currentDetail;
  return (
    <Box sx={{ mb: 1.5, pb: 1, borderBottom: 1, borderColor: 'divider' }}>
      <Typography variant="subtitle2" sx={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.02em', color: 'text.primary' }}>{currentId}</Typography>
      {d && (
        <Box sx={{ mt: 0.5, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {d.age && d.age !== '?' && <Typography variant="caption" sx={{ fontSize: 10.5, color: 'text.secondary', letterSpacing: '0.02em' }}>年龄 {d.age}</Typography>}
          {d.sex && d.sex !== '?' && <Typography variant="caption" sx={{ fontSize: 10.5, color: 'text.secondary', letterSpacing: '0.02em' }}>性别 {d.sex}</Typography>}
          {d.bmi && d.bmi !== '?' && <Typography variant="caption" sx={{ fontSize: 10.5, color: 'text.secondary', letterSpacing: '0.02em' }}>BMI {d.bmi}</Typography>}
        </Box>
      )}
      {error && <Typography variant="caption" color="error" sx={{ fontSize: 10.5 }}>{error}</Typography>}
    </Box>
  );
}
