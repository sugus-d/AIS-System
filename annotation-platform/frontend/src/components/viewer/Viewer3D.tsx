// src/components/viewer/Viewer3D.tsx
import { Box } from '@mui/material';
import { useLandmarkStore } from '../../stores/landmarkStore';
import { useSubjectStore } from '../../stores/subjectStore';
import MeshScene from './MeshScene';

interface Props {
  onReady?: () => void;
}

export default function Viewer3D({ onReady }: Props) {
  const currentId = useSubjectStore((s) => s.currentId);
  const landmarks = useLandmarkStore((s) => s.landmarks);
  if (!currentId) return null;
  return (
    <Box sx={{ width: '100%', height: '100%', bgcolor: 'background.default' }}>
      <MeshScene subjectId={currentId} landmarks={landmarks} onReady={onReady} />
    </Box>
  );
}
