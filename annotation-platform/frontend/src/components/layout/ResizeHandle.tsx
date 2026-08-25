// src/components/layout/ResizeHandle.tsx
import { useRef, useCallback } from 'react';
import { Box } from '@mui/material';

interface Props {
  onResize: (delta: number) => void;
}

export default function ResizeHandle({ onResize }: Props) {
  const dragging = useRef(false);
  const startX = useRef(0);
  const onResizeRef = useRef(onResize);
  onResizeRef.current = onResize;

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current) return;
    onResizeRef.current(e.clientX - startX.current);
    startX.current = e.clientX;
  }, []);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
  }, [onMouseMove]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    startX.current = e.clientX;
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [onMouseMove, onMouseUp]);

  return (
    <Box
      onMouseDown={onMouseDown}
      sx={{
        width: 4, cursor: 'col-resize', bgcolor: 'transparent',
        transition: 'all 0.12s ease', flexShrink: 0, position: 'relative',
        '&:hover': { bgcolor: 'rgba(56, 189, 248, 0.06)' },
        '&:active': { bgcolor: 'rgba(56, 189, 248, 0.12)' },
        '&::after': {
          content: '""', position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 1.5, height: 28, borderRadius: 0.5,
          bgcolor: 'rgba(148,163,184,0.12)', transition: 'all 0.12s ease',
        },
        '&:hover::after': { bgcolor: 'primary.main', height: 36 },
      }}
    />
  );
}
