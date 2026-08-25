// src/components/common/Modal.tsx
import { Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material';

interface ConfirmModalProps {
  open: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  confirmText?: string;
  cancelText?: string;
}

export function ConfirmModal({ open, title, message, onConfirm, onCancel, confirmText = '确认', cancelText = '取消' }: ConfirmModalProps) {
  return (
    <Dialog open={open} onClose={onCancel} maxWidth="xs"
      PaperProps={{ sx: { borderRadius: 1.5 } }}>
      <DialogTitle sx={{ fontSize: 14, pb: 0.5, letterSpacing: '0.02em' }}>{title}</DialogTitle>
      <DialogContent sx={{ fontSize: 12, color: 'text.secondary' }}>{message}</DialogContent>
      <DialogActions sx={{ px: 2, pb: 1.5 }}>
        <Button onClick={onCancel} size="small" sx={{ fontSize: 11, color: 'text.secondary' }}>{cancelText}</Button>
        <Button onClick={onConfirm} color="primary" variant="contained" size="small" sx={{ fontSize: 11 }}>{confirmText}</Button>
      </DialogActions>
    </Dialog>
  );
}
