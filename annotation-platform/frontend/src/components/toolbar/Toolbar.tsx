// src/components/toolbar/Toolbar.tsx
import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  Box,
  Button,
  Slider,
  Typography,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from "@mui/material";
import { useUIStore } from "../../stores/uiStore";
import { useSubjectStore } from "../../stores/subjectStore";
import { useLandmarkStore } from "../../stores/landmarkStore";
import { brushCommit, fetchCurvatureMapping } from "../../api/subjects";
import { validateLandmarks, validateLandmarksWithIssues, saveLandmarks } from "../../api/landmarks";
import type { ValidationIssue } from "../../types";

// Option descriptors — not all entries are always shown; filtered dynamically
const LANDMARK_DEFS = [
  { key: "neck_root", label: "颈根", sides: ["L", "R"], spine: false },
  {
    key: "shoulder_transition",
    label: "肩臂转点",
    sides: ["L", "R"],
    spine: false,
  },
  { key: "scapular_peaks", label: "肩胛峰", sides: ["L", "R"], spine: false },
  { key: "axilla", label: "腋窝", sides: ["L", "R"], spine: false },
  { key: "waist", label: "腰部", sides: ["L", "R"], spine: false },
  { key: "waist_lower", label: "腰下缘", sides: ["L", "R"], spine: false },
  { key: "spine_points", label: "脊柱 P", spineIdx: [0, 1, 2, 3, 4, 5], spine: true },
];

interface Props {
  onSwitchView?: () => void;
}

export default function Toolbar({ onSwitchView }: Props) {
  const {
    viewMode,
    setViewMode,
    brushMode,
    setBrushMode,
    brushSize,
    setBrushSize,
    rotationAngle,
    setRotationAngle,
    rotationMode,
    setRotationMode,
    clothOverlay,
    setClothOverlay,
    brushPoints,
    clearBrushPoints,
    restoreClothIndices,
    clearRestoreCloth,
    addToast,
    incrementMeshVersion,
    pendingLandmark,
  } = useUIStore();
  const { currentId, patchStatus, setManualStatus, subjects } = useSubjectStore();
  const { landmarks, mapping, updateLandmark2D, save, saving, isDirty } =
    useLandmarkStore();
  const storeValidationIssues = useUIStore((s) => s.setValidationIssues);
  const currentStatus = currentId ? subjects.find((s) => s.id === currentId)?.labeling_status : undefined;
  const hasBrush = brushPoints.length > 0 || restoreClothIndices.length > 0;

  const [addValue, setAddValue] = useState("");
  const [confirmAction, setConfirmAction] = useState<"clear" | "commit" | null>(
    null,
  );
  const [validationIssues, setValidationIssues] = useState<{
    errors: ValidationIssue[];
    warnings: ValidationIssue[];
  } | null>(null);

  // ── Dynamic add-landmark options: only show not-yet-placed entries ──
  const addOptions = useMemo(() => {
    const out: { value: string; label: string }[] = [];
    for (const def of LANDMARK_DEFS) {
      if (def.spine) {
        for (const idx of def.spineIdx || []) {
          const pt = landmarks[def.key]?.[idx];
          const isPlaced = pt && Array.isArray(pt);
          if (!isPlaced) {
            out.push({
              value: `${def.key}_P${idx}`,
              label: idx === 5 ? `${def.label}m` : `${def.label}${idx}`,
            });
          }
        }
      } else {
        for (let i = 0; i < def.sides.length; i++) {
          const pts = landmarks[def.key];
          if (!pts?.[i]) {
            out.push({
              value: `${def.key}_${def.sides[i]}`,
              label: `${def.label} ${def.sides[i]}`,
            });
          }
        }
      }
    }
    return out;
  }, [landmarks]);

  const handleSave = async () => {
    if (!currentId) return;
    // Step 1: 先校验坐标顺序 + mesh 吸附
    try {
      const validated = await validateLandmarksWithIssues(currentId, useLandmarkStore.getState().landmarks);
      // 使用经过 mesh 吸附的 landmarks 更新 store（吸附到最近 mesh 顶点）
      useLandmarkStore.setState({ landmarks: validated.landmarks, isDirty: true });
      const errors = validated.issues.filter((i) => i.severity === "error");
      const warnings = validated.issues.filter((i) => i.severity === "warning");
      storeValidationIssues(validated.issues); // 同步到全局 store
      if (errors.length > 0 || warnings.length > 0) {
        setValidationIssues({ errors, warnings });
        return; // 等待用户选择
      }
    } catch {
      console.error("坐标校验请求失败（如网络超时），继续保存");
    }
    // Step 2: 无问题，直接保存
    await doSave();
  };

  const doSave = async (bypass = false) => {
    if (!currentId) return;
    try {
      const result = bypass
        ? await saveLandmarks(currentId, useLandmarkStore.getState().landmarks, true)
        : await save(currentId);
      if (result?.labeling_status) {
        patchStatus(currentId, result.labeling_status as any);
      }
      if (result?.status === "error") {
        addToast("保存失败：" + (result as any)?.error || "未知错误", "error");
      } else {
        addToast("标注已保存", "success");
      }
    } catch {
      addToast("保存失败", "error");
    }
  };

  const handleForceSave = async () => {
    // 仅在存在 error 级别问题时使用 bypass（warning 不阻塞保存）
    const hasErrors = validationIssues ? validationIssues.errors.length > 0 : false;
    setValidationIssues(null);
    storeValidationIssues([]);
    await doSave(hasErrors);
  };

  const handleSwitchView = () => {
    setViewMode(viewMode === "2d" ? "3d" : "2d");
    onSwitchView?.();
  };

  // ── Brush actions ──
  const handleBrushErase = () => {
    setBrushMode(brushMode === "erase" ? null : "erase");
  };
  const handleBrushRestore = () => {
    setBrushMode(brushMode === "restore" ? null : "restore");
  };
  const handleBrushClear = () => {
    if (hasBrush) {
      setConfirmAction("clear");
      return;
    }
    clearBrushPoints();
    clearRestoreCloth();
    addToast("已清空所有标记区域", "info");
  };
  const handleConfirmClear = () => {
    clearBrushPoints();
    clearRestoreCloth();
    addToast("已清空所有标记区域", "info");
    setConfirmAction(null);
  };
  const handleBrushCommit = async () => {
    if (!currentId) return;
    setConfirmAction("commit");
  };
  const handleConfirmCommit = async () => {
    if (!currentId) return;
    setConfirmAction(null);
    try {
      await brushCommit(currentId, {
        points: brushPoints.length > 0 ? brushPoints : undefined,
        cloth_indices:
          restoreClothIndices.length > 0 ? restoreClothIndices : undefined,
      });
      incrementMeshVersion();
      setBrushMode(null);
      clearBrushPoints();
      clearRestoreCloth();
      // commit 后重新获取 PCA mapping（edited mesh 的 PCA 参数可能变了）
      fetchCurvatureMapping(currentId)
        .then((m) => useLandmarkStore.setState({ mapping: m }))
        .catch((e) => console.error("获取 curvature mapping 失败", e));
      // commit 后校验 landmark：吸附到最新 mesh + 更新脊椎联动
      validateLandmarks(currentId, useLandmarkStore.getState().landmarks)
        .then((validated) => useLandmarkStore.setState({ landmarks: validated, isDirty: true }))
        .catch((e) => console.error("commit 后 landmark 校验失败", e));
      addToast("操作已确认并保存", "success");
    } catch {
      addToast("操作失败", "error");
    }
  };

  const setPendingLandmark = useUIStore((s) => s.setPendingLandmark);
  // 新增 landmark 被放置/取消后清空下拉框
  const prevPending = useRef(pendingLandmark);
  useEffect(() => {
    if (prevPending.current && !pendingLandmark) {
      setAddValue("");
    }
    prevPending.current = pendingLandmark;
  }, [pendingLandmark]);

  // 下拉选项变更后，若当前选中值已不存在于选项列表中（如该点已放置），则清空
  useEffect(() => {
    if (addValue) {
      const validValues = new Set(addOptions.map((o) => o.value));
      if (!validValues.has(addValue)) {
        setAddValue("");
      }
    }
  }, [addOptions, addValue]);

  const parseAddValue = useCallback((val: string) => {
    if (!val) return null;
    if (val.startsWith("spine_points_P")) {
      const idx = parseInt(val.replace("spine_points_P", ""), 10);
      return { name: "spine_points", index: idx };
    } else {
      const under = val.lastIndexOf("_");
      return {
        name: val.slice(0, under),
        index: val.endsWith("_R") ? 1 : 0,
      };
    }
  }, []);
  const handleAddChange = (val: string) => {
    setAddValue(val);
    // 已激活状态下切换下拉项 → 立即更新 pending landmark
    if (pendingLandmark && val) {
      const parsed = parseAddValue(val);
      if (parsed) setPendingLandmark(parsed);
    }
  };
  const handleAddToggle = () => {
    if (pendingLandmark) {
      // 取消激活
      setPendingLandmark(null);
    } else if (addValue) {
      // 激活：用当前下拉选中值
      const parsed = parseAddValue(addValue);
      if (parsed) setPendingLandmark(parsed);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        gap: 1,
        p: 1,
        alignItems: "center",
        flexWrap: "wrap",
        borderTop: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
      }}
    >
      <Button size="small" variant="outlined" onClick={handleSwitchView}>
        {viewMode === "2d" ? "3D 视图" : "返回 2D"}
      </Button>

      <Button
        size="small"
        variant={isDirty ? "contained" : "outlined"}
        color={isDirty ? "success" : "inherit"}
        onClick={handleSave}
        disabled={saving}
      >
        {saving ? "保存中..." : "完成标注"}
      </Button>

      <Select
        size="small"
        value={currentStatus || 'prelabeled'}
        onChange={(e) => { if (currentId) setManualStatus(currentId, e.target.value); }}
        sx={{ fontSize: 11, height: 28, minWidth: 80 }}
      >
        <MenuItem value="unlabeled" sx={{ fontSize: 11 }}>未标</MenuItem>
        <MenuItem value="prelabeled" sx={{ fontSize: 11 }}>预标</MenuItem>
        <MenuItem value="labeled" sx={{ fontSize: 11 }}>已标</MenuItem>
      </Select>

      {/* ── Brush controls + cloth toggle — 3D mode always visible ── */}
      {viewMode === "3d" && (
        <>
          <Box
            sx={{ borderLeft: 1, borderColor: "divider", mx: 0.5, height: 20 }}
          />
          <Button
            size="small"
            variant={brushMode === "erase" ? "contained" : "outlined"}
            color={brushMode === "erase" ? "error" : "inherit"}
            onClick={handleBrushErase}
            sx={{ fontSize: 11, minWidth: 60 }}
          >
            擦除
          </Button>
          <Button
            size="small"
            variant={brushMode === "restore" ? "contained" : "outlined"}
            color={brushMode === "restore" ? "secondary" : "inherit"}
            onClick={handleBrushRestore}
            sx={{ fontSize: 11, minWidth: 56 }}
          >
            恢复
          </Button>
          <Slider
            size="small"
            value={brushSize}
            onChange={(_, v) => setBrushSize(v as number)}
            min={5}
            max={80}
            sx={{ width: 80 }}
          />
          <Typography variant="caption" sx={{ minWidth: 20 }}>
            {brushSize}
          </Typography>
          <Button
            size="small"
            variant={hasBrush ? "contained" : "outlined"}
            color={hasBrush ? "error" : "inherit"}
            onClick={handleBrushClear}
            sx={{
              fontSize: 11,
              minWidth: 48,
              border: hasBrush ? "1px solid transparent" : undefined,
            }}
          >
            清空
          </Button>
          <Button
            size="small"
            variant={hasBrush ? "contained" : "outlined"}
            color={hasBrush ? "primary" : "inherit"}
            onClick={handleBrushCommit}
            sx={{
              fontSize: 11,
              minWidth: 48,
              border: hasBrush ? "1px solid transparent" : undefined,
            }}
          >
            确认
          </Button>
          <Box
            sx={{ borderLeft: 1, borderColor: "divider", mx: 0.5, height: 20 }}
          />
          <Button
            size="small"
            variant={clothOverlay ? "contained" : "outlined"}
            color={clothOverlay ? "info" : "inherit"}
            onClick={() => setClothOverlay(!clothOverlay)}
            sx={{ fontSize: 11, minWidth: 52 }}
          >
            衣物
          </Button>
        </>
      )}

      {/* ── Rotation controls — 2D mode only ── */}
      {viewMode === "2d" && (
        <>
          <Box
            sx={{ borderLeft: 1, borderColor: "divider", mx: 0.5, height: 20 }}
          />
          <Button
            size="small"
            variant={rotationMode ? "contained" : "outlined"}
            color={rotationMode ? "info" : "inherit"}
            onClick={() => setRotationMode(!rotationMode)}
            sx={{ fontSize: 11, minWidth: 72 }}
          >
            旋转模式
          </Button>
          <Typography variant="caption" sx={{ minWidth: 32 }}>
            {rotationAngle}°
          </Typography>
          <Button
            size="small"
            variant="outlined"
            onClick={() => {
              setRotationAngle(0);
              setRotationMode(false);
            }}
            sx={{ fontSize: 11, minWidth: 48 }}
          >
            复位
          </Button>
        </>
      )}

      {/* ── Add landmark dropdown — 2D mode only ── */}
      {viewMode === "2d" && (
        <>
          <Box
            sx={{ borderLeft: 1, borderColor: "divider", mx: 0.5, height: 20 }}
          />
          <Button
            size="small"
            variant={pendingLandmark ? "contained" : "outlined"}
            color={pendingLandmark ? "primary" : "inherit"}
            onClick={handleAddToggle}
            disabled={!addValue}
            sx={{ fontSize: 11, minWidth: 84 }}
          >
            添加标注点
          </Button>
          <Select
            size="small"
            value={addValue}
            displayEmpty
            onChange={(e) => handleAddChange(e.target.value)}
            sx={{ fontSize: 11, height: 28, minWidth: 120 }}
          >
            <MenuItem value="" disabled>
              {addOptions.length === 0 ? "已添加全部" : "选择类型..."}
            </MenuItem>
            {addOptions.map((opt) => (
              <MenuItem key={opt.value} value={opt.value} sx={{ fontSize: 11 }}>
                {opt.label}
              </MenuItem>
            ))}
          </Select>
        </>
      )}

      {/* Confirm dialog */}
      <Dialog
        open={confirmAction !== null}
        onClose={() => setConfirmAction(null)}
        PaperProps={{ sx: { borderRadius: 2 } }}
      >
        <DialogTitle sx={{ fontSize: 15, pb: 0.5 }}>
          {confirmAction === "clear" ? "清空标记" : "提交修改"}
        </DialogTitle>
        <DialogContent sx={{ pt: 1.5 }}>
          <DialogContentText sx={{ fontSize: 13 }}>
            {confirmAction === "clear"
              ? "确定清空所有笔刷标记区域？擦除和恢复的痕迹都将被移除。"
              : "确定提交笔刷修改？模型将被修改并保存为新版本。"}
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 2, pb: 1.5 }}>
          <Button
            size="small"
            onClick={() => setConfirmAction(null)}
            sx={{ fontSize: 12, color: "text.secondary" }}
          >
            取消
          </Button>
          <Button
            size="small"
            variant="contained"
            color={confirmAction === "clear" ? "error" : "primary"}
            onClick={
              confirmAction === "clear" ? handleConfirmClear : handleConfirmCommit
            }
            sx={{ fontSize: 12 }}
          >
            确定
          </Button>
        </DialogActions>
      </Dialog>

      {/* Validation issues dialog */}
      <Dialog
        open={validationIssues !== null}
        onClose={() => setValidationIssues(null)}
        PaperProps={{ sx: { borderRadius: 2, minWidth: 360, maxWidth: 480 } }}
      >
        <DialogTitle sx={{ fontSize: 14, pb: 0.5, letterSpacing: '0.02em' }}>
          坐标校验
        </DialogTitle>
        <DialogContent sx={{ pt: 1.5 }}>
          {validationIssues?.errors.length ? (
            <Box sx={{ mb: 1.5 }}>
              <Typography variant="caption" sx={{ fontWeight: "bold", color: "error.main", fontSize: 12, display: "block", mb: 0.5, letterSpacing: "0.02em" }}>
                错误
              </Typography>
              {validationIssues.errors.map((issue, idx) => (
                <Typography key={idx} variant="body2" sx={{ fontSize: 12, color: "error.main", mb: 0.5, pl: 1 }}>
                  {issue.message}
                </Typography>
              ))}
            </Box>
          ) : null}
          {validationIssues?.warnings.length ? (
            <Box sx={{ mb: 1 }}>
              <Typography variant="caption" sx={{ fontWeight: "bold", color: "warning.main", fontSize: 12, display: "block", mb: 0.5, letterSpacing: "0.02em" }}>
                警告
              </Typography>
              {validationIssues.warnings.map((issue, idx) => (
                <Typography key={idx} variant="body2" sx={{ fontSize: 12, color: "warning.dark", mb: 0.5, pl: 1 }}>
                  {issue.message}
                </Typography>
              ))}
            </Box>
          ) : null}
        </DialogContent>
        <DialogActions sx={{ px: 2, pb: 1.5 }}>
          <Button
            size="small"
            onClick={() => setValidationIssues(null)}
            sx={{ fontSize: 12, color: "text.secondary" }}
          >
            返回修改
          </Button>
          {validationIssues?.errors.length ? (
            <Button
              size="small"
              variant="contained"
              color="warning"
              onClick={handleForceSave}
              sx={{ fontSize: 12 }}
            >
              忽略并保存
            </Button>
          ) : (
            <Button
              size="small"
              variant="contained"
              onClick={handleForceSave}
              sx={{ fontSize: 12 }}
            >
              确认保存
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}
