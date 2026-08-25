// src/components/viewer/Curvature2D.tsx
import { useEffect, useRef, useState, useCallback } from 'react';
import { Box, Typography } from '@mui/material';
import { useLandmarkStore } from '../../stores/landmarkStore';
import { useUIStore } from '../../stores/uiStore';
import { useSubjectStore } from '../../stores/subjectStore';
import { getCurvatureImageUrl } from '../../api/subjects';
import { useCanvasRenderer, dataToPixel, pixelToData, landmarkToDisplay } from '../../hooks/useCanvasRenderer';

const IMAGE_TIMEOUT_MS = 30_000;

/** Promise 化的图片加载，含超时 */
function loadImageWithTimeout(src: string, timeoutMs = IMAGE_TIMEOUT_MS): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    const timer = setTimeout(() => {
      img.onload = null;
      img.onerror = null;
      reject(new Error('图片加载超时'));
    }, timeoutMs);
    img.onload = () => {
      clearTimeout(timer);
      resolve(img);
    };
    img.onerror = () => {
      clearTimeout(timer);
      reject(new Error('图片加载失败'));
    };
    img.src = src;
  });
}

const HIT_RADIUS = 20;
const SPINE_CONSTRAINT_PAIRS = ['neck_root', 'scapular_peaks', 'axilla', 'waist'];
const SPINE_HIDDEN: Record<number, boolean> = {};
const MIDBACK_PAIR = ['axilla', 'waist'];  // mid_back P5: A=mid(axilla_L+waist_L), B=mid(axilla_R+waist_R)

interface Props {
  onSwitch3D?: () => void;
  onReady?: () => void;
}

export default function Curvature2D({ onSwitch3D, onReady }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [imgReady, setImgReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const loadedForSubject = useRef<string | null>(null);
  const { mapping, landmarks, loading, error: landmarkError, updateLandmark2D, fetchLift } = useLandmarkStore();
  const rotationAngle = useUIStore((s) => s.rotationAngle);
  const rotationMode = useUIStore((s) => s.rotationMode);
  const setRotationMode = useUIStore((s) => s.setRotationMode);
  const setRotationAngle = useUIStore((s) => s.setRotationAngle);
  const leftWidth = useUIStore((s) => s.leftWidth);
  const rightWidth = useUIStore((s) => s.rightWidth);
  const meshVersion = useUIStore((s) => s.meshVersion);
  const currentId = useSubjectStore((s) => s.currentId);
  const pendingLandmark = useUIStore((s) => s.pendingLandmark);
  const setPendingLandmark = useUIStore((s) => s.setPendingLandmark);
  const [dragTarget, setDragTarget] = useState<{ name: string; index: number } | null>(null);
  const mousePos = useRef<{ x: number; y: number } | null>(null);

  // 加载曲率图像 — currentId 或 meshVersion 变化时刷新
  const loadKey = `${currentId}-v${meshVersion}`;
  const prevKey = useRef<string | null>(null);
  useEffect(() => {
    if (!currentId) {
      setImg(null);
      setImgReady(false);
      loadedForSubject.current = null;
      prevKey.current = null;
      return;
    }
    const key = `${currentId}-v${meshVersion}`;
    if (prevKey.current === key && imgReady) return;
    prevKey.current = key;

    setImg(null);
    setImgReady(false);
    loadedForSubject.current = null;

    const subjectAtStart = currentId;
    let cancelled = false;
    loadImageWithTimeout(getCurvatureImageUrl(currentId, meshVersion))
      .then((imgEl) => {
        if (!cancelled && loadedForSubject.current === null) {
          setImg(imgEl);
          setImgReady(true);
          loadedForSubject.current = subjectAtStart;
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          useUIStore.getState().addToast(`曲率图加载失败：${err.message}，请重试`, 'error');
          setImgReady(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [currentId, meshVersion, imgReady]);

  const { draw } = useCanvasRenderer(canvasRef, img, mapping, landmarks, rotationAngle, dragTarget, pendingLandmark, mousePos);
  useEffect(() => { draw(); }, [draw]);

  // Escape 键取消新增 landmark
  useEffect(() => {
    if (!pendingLandmark) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setPendingLandmark(null);
        mousePos.current = null;
        draw();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [pendingLandmark, setPendingLandmark, draw]);

  // Resize canvas to container — also triggers redraw on panel resize
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const resize = () => {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
      drawRef.current();
    };
    resize();
    const observer = new ResizeObserver(() => resize());
    observer.observe(container);
    return () => observer.disconnect();
  }, [leftWidth, rightWidth]);

  // Find nearby landmark point (project original 3D → PCA space → pixel)
  const findNearby = useCallback((px: number, py: number) => {
    if (!mapping) return null;
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const w = canvas.width, h = canvas.height;
    let best: { name: string; index: number } | null = null;
    let bestDist = HIT_RADIUS;
    for (const [name, pts] of Object.entries(landmarks)) {
      for (let i = 0; i < pts.length; i++) {
        const raw = pts[i];
        if (!raw || !Array.isArray(raw)) continue;
        if (raw[0] == null) continue;
        const dp = landmarkToDisplay(raw, mapping);
        if (!dp) continue;
        const p = dataToPixel(dp.x, dp.y, mapping, w, h);
        const d = Math.hypot(p.x - px, p.y - py);
        if (d < bestDist) { bestDist = d; best = { name, index: i }; }
      }
    }
    return best;
  }, [landmarks, mapping]);

  // Project point p onto line segment ab, clamped to the segment
  const projectOnLine = useCallback((p: number[], a: number[], b: number[]) => {
    const ax = b[0] - a[0], ay = b[1] - a[1];
    const bx = p[0] - a[0], by = p[1] - a[1];
    const dot = ax * bx + ay * by;
    const len2 = ax * ax + ay * ay;
    if (len2 < 1e-8) return [a[0], a[1]];
    let t = dot / len2;
    t = Math.max(0, Math.min(1, t));
    return [a[0] + t * ax, a[1] + t * ay];
  }, []);

  // 计算 A(mid axilla_L+waist_L) 和 B(mid axilla_R+waist_R)
  const computeMidBackAB = useCallback(() => {
    const getMid = (name1: string, name2: string, side: number) => {
      const p1 = landmarks[name1]?.[side];
      const p2 = landmarks[name2]?.[side];
      if (!p1 || !p2) return null;
      const d1 = landmarkToDisplay(p1, mapping!);
      const d2 = landmarkToDisplay(p2, mapping!);
      if (!d1 || !d2) return null;
      return { x: (d1.x + d2.x) / 2, y: (d1.y + d2.y) / 2 };
    };
    const A = getMid(MIDBACK_PAIR[0], MIDBACK_PAIR[1], 0);
    const B = getMid(MIDBACK_PAIR[0], MIDBACK_PAIR[1], 1);
    return A && B ? { A, B } : null;
  }, [landmarks, mapping]);

  // 更新 mid_back ：投影当前 P6 到 AB 线上
  const updateMidBack = useCallback(() => {
    const ab = computeMidBackAB();
    if (!ab) return;
    const curP = landmarks['spine_points']?.[5];
    if (curP) {
      const curPPCA = landmarkToDisplay(curP, mapping!);
      if (curPPCA) {
        const proj = projectOnLine([curPPCA.x, curPPCA.y], [ab.A.x, ab.A.y], [ab.B.x, ab.B.y]);
        useLandmarkStore.getState().updateLandmark2D('spine_points', 5, parseFloat(proj[0].toFixed(1)), parseFloat(proj[1].toFixed(1)));
        return;
      }
    }
    // 回退：中点
    const mx = (ab.A.x + ab.B.x) / 2;
    const my = (ab.A.y + ab.B.y) / 2;
    useLandmarkStore.getState().updateLandmark2D('spine_points', 5, parseFloat(mx.toFixed(1)), parseFloat(my.toFixed(1)));
  }, [landmarks, mapping, computeMidBackAB, projectOnLine]);

  // 双边 landmark → PCA 空间将当前 P 投影到 LR 连线（不跑到中点）
  const updateSpineMidpoint = useCallback((bilateralName: string) => {
    const spineIdx = SPINE_CONSTRAINT_PAIRS.indexOf(bilateralName);
    if (spineIdx < 0 || SPINE_HIDDEN[spineIdx]) return;
    const pair = landmarks[bilateralName];
    if (!pair || !pair[0] || !pair[1]) return;
    const pL = landmarkToDisplay(pair[0], mapping!);
    const pR = landmarkToDisplay(pair[1], mapping!);
    if (!pL || !pR) return;
    // 获取当前 P 点，投影到新的 LR 线段上
    const curP = landmarks['spine_points']?.[spineIdx];
    if (curP) {
      const curPPCA = landmarkToDisplay(curP, mapping!);
      if (curPPCA) {
        const proj = projectOnLine([curPPCA.x, curPPCA.y], [pL.x, pL.y], [pR.x, pR.y]);
        useLandmarkStore.getState().updateLandmark2D('spine_points', spineIdx, parseFloat(proj[0].toFixed(1)), parseFloat(proj[1].toFixed(1)));
      }
    } else {
      // 回退：无当前 P 时用中点（首次加载）
      const midX_pca = (pL.x + pR.x) / 2;
      const midY_pca = (pL.y + pR.y) / 2;
      useLandmarkStore.getState().updateLandmark2D('spine_points', spineIdx, parseFloat(midX_pca.toFixed(1)), parseFloat(midY_pca.toFixed(1)));
    }
    // axilla 或 waist 变化时同时更新 mid_back 
    if (bilateralName === MIDBACK_PAIR[0] || bilateralName === MIDBACK_PAIR[1]) {
      updateMidBack();
    }
  }, [landmarks, mapping, updateMidBack]);

  // 拖拽时记录最后一次的 PCA 坐标，用于 lift 反向搜索
  const lastDragPCA = useRef<{ x: number; y: number } | null>(null);
  // Rotation drag state
  const rotDrag = useRef<{ startAngle: number; startA: number } | null>(null);

  // Mouse handlers
  const isDragging = useRef(false);
  const dragPoint = useRef<{ name: string; index: number } | null>(null);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (busy || !mapping || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const pxRaw = e.clientX - rect.left;
    const pyRaw = e.clientY - rect.top;
    const w = canvasRef.current.width || rect.width;
    const h = canvasRef.current.height || rect.height;

    // 反旋转像素坐标（用于数据定位，不用于旋转角度计算）
    const _angRad = -rotationAngle * Math.PI / 180;
    const _dcx = pxRaw - w/2, _dcy = pyRaw - h/2;
    const px = rotationAngle !== 0 ? _dcx * Math.cos(_angRad) - _dcy * Math.sin(_angRad) + w/2 : pxRaw;
    const py = rotationAngle !== 0 ? _dcx * Math.sin(_angRad) + _dcy * Math.cos(_angRad) + h/2 : pyRaw;

    // Pending landmark placement
    if (pendingLandmark) {
      const data = pixelToData(px, py, mapping, w, h);
      const x = parseFloat(data.x.toFixed(1));
      const y = parseFloat(data.y.toFixed(1));
      updateLandmark2D(pendingLandmark.name, pendingLandmark.index, x, y);
      setPendingLandmark(null);
      mousePos.current = null;
      // lift 吸附到最近 mesh 顶点
      if (currentId) {
        fetchLift(currentId, x, y).then((res) => {
          useLandmarkStore.getState().updateLandmark3D(pendingLandmark.name, pendingLandmark.index, res);
        }).catch(() => {});
      }
      return;
    }

    // Rotation mode drag
    if (rotationMode) {
      const cx = w / 2;
      const cy = h / 2;
      rotDrag.current = {
        startAngle: rotationAngle,
        startA: Math.atan2(pyRaw - cy, pxRaw - cx),
      };
      canvasRef.current.style.cursor = 'grabbing';
      return;
    }

    // Landmark drag
    const near = findNearby(px, py);
    if (near) {
      isDragging.current = true;
      dragPoint.current = near;
      setDragTarget(near);
      canvasRef.current.style.cursor = 'grabbing';
    }
  }, [mapping, findNearby, busy, rotationMode, rotationAngle, pendingLandmark, updateLandmark2D, setPendingLandmark, currentId, fetchLift]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (busy || !mapping || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const pxRaw = e.clientX - rect.left;
    const pyRaw = e.clientY - rect.top;
    const w = canvasRef.current.width;
    const h = canvasRef.current.height;

    // 反旋转像素坐标（用于数据定位）
    const _angRad = -rotationAngle * Math.PI / 180;
    const _dcx = pxRaw - w/2, _dcy = pyRaw - h/2;
    const px = rotationAngle !== 0 ? _dcx * Math.cos(_angRad) - _dcy * Math.sin(_angRad) + w/2 : pxRaw;
    const py = rotationAngle !== 0 ? _dcx * Math.sin(_angRad) + _dcy * Math.cos(_angRad) + h/2 : pyRaw;

    // Track mouse for pending placement ghost marker（用原始坐标，ghost 绘制在旋转层外）
    mousePos.current = { x: pxRaw, y: pyRaw };
    if (pendingLandmark) { draw(); return; }

    // Rotation drag
    if (rotDrag.current) {
      const delta = Math.atan2(pyRaw - h/2, pxRaw - w/2) - rotDrag.current.startA;
      let newAngle = (rotDrag.current.startAngle + delta * 180 / Math.PI) % 360;
      if (newAngle < 0) newAngle += 360;
      setRotationAngle(Math.round(newAngle));
      return;
    }

    if (isDragging.current && dragPoint.current) {
      const data = pixelToData(px, py, mapping, w, h);
      const { name, index } = dragPoint.current;
      let x = parseFloat(data.x.toFixed(1));
      let y = parseFloat(data.y.toFixed(1));
      lastDragPCA.current = { x, y };
      // Spine points: 双边 landmark 投影到 PCA 空间后再做约束
      if (name === 'spine_points') {
        let pairName: string | null = null;
        if (index === 5) {
          // mid_back : 约束在 AB 线上（A=mid(axilla+waist)L, B=mid(axilla+waist)R）
          const ab = computeMidBackAB();
          if (ab) {
            const proj = projectOnLine([x, y], [ab.A.x, ab.A.y], [ab.B.x, ab.B.y]);
            x = parseFloat(proj[0].toFixed(1));
            y = parseFloat(proj[1].toFixed(1));
          }
        } else {
          pairName = SPINE_CONSTRAINT_PAIRS[index];
        }
        if (pairName && mapping) {
          const pair = landmarks[pairName];
          if (pair && pair[0] && pair[1]) {
            const a = landmarkToDisplay(pair[0], mapping);
            const b = landmarkToDisplay(pair[1], mapping);
            if (a && b) {
              const proj = projectOnLine([x, y], [a.x, a.y], [b.x, b.y]);
              x = parseFloat(proj[0].toFixed(1));
              y = parseFloat(proj[1].toFixed(1));
            }
          }
        }
      }
      updateLandmark2D(name, index, x, y);
      // Sync spine midpoint when dragging a bilateral point
      if (name !== 'spine_points') {
        updateSpineMidpoint(name);
      }
      draw();
    } else if (rotationMode) {
      canvasRef.current.style.cursor = 'move';
    } else {
      const near = findNearby(px, py);
      canvasRef.current.style.cursor = near ? 'grab' : 'default';
    }
  }, [mapping, findNearby, updateLandmark2D, updateSpineMidpoint, draw, busy, rotationMode, rotationAngle, setRotationAngle, pendingLandmark]);

  // 右键取消新增 landmark
  const onContextMenu = useCallback((e: React.MouseEvent) => {
    if (pendingLandmark) {
      e.preventDefault();
      setPendingLandmark(null);
      mousePos.current = null;
      draw();
    }
  }, [pendingLandmark, setPendingLandmark, draw]);

  const onMouseUp = useCallback(async () => {
    // End rotation drag
    if (rotDrag.current) {
      rotDrag.current = null;
      if (canvasRef.current) canvasRef.current.style.cursor = rotationMode ? 'move' : 'default';
    }

    if (!isDragging.current || !dragPoint.current || !currentId) {
      isDragging.current = false;
      dragPoint.current = null;
      setDragTarget(null);
      return;
    }
    const { name, index } = dragPoint.current;
    isDragging.current = false;
    dragPoint.current = null;
    setDragTarget(null);
    if (canvasRef.current) canvasRef.current.style.cursor = 'wait';
    setBusy(true);

    try {
      // 用拖拽时记录的 PCA 坐标做 lift（store 存的是逆变换后的原始 3D，不能用）
      const liftPt = lastDragPCA.current || (() => {
        const pt = landmarks[name]?.[index] as any;
        return { x: Array.isArray(pt) ? pt[0] : pt?.x, y: Array.isArray(pt) ? pt[1] : pt?.y };
      })();
      if (liftPt.x != null && liftPt.y != null) {
        if (name === 'spine_points') {
          await new Promise((r) => setTimeout(r, 50));
          // Spine: project raw drag coords onto constraint line, then use constrained coords for lift
          let constrainedX = liftPt.x;
          let constrainedY = liftPt.y;
          if (index === 5) {
            // mid_back : constrain to AB line
            const ab = computeMidBackAB();
            if (ab) {
              const proj = projectOnLine([liftPt.x, liftPt.y], [ab.A.x, ab.A.y], [ab.B.x, ab.B.y]);
              constrainedX = parseFloat(proj[0].toFixed(1));
              constrainedY = parseFloat(proj[1].toFixed(1));
              useLandmarkStore.getState().updateLandmark2D('spine_points', index, constrainedX, constrainedY);
            }
          } else {
            const pairName = SPINE_CONSTRAINT_PAIRS[index];
            if (pairName && mapping) {
              const pair = landmarks[pairName];
              if (pair && pair[0] && pair[1]) {
                const a = landmarkToDisplay(pair[0], mapping);
                const b = landmarkToDisplay(pair[1], mapping);
                if (a && b) {
                  const proj = projectOnLine([liftPt.x, liftPt.y], [a.x, a.y], [b.x, b.y]);
                  constrainedX = parseFloat(proj[0].toFixed(1));
                  constrainedY = parseFloat(proj[1].toFixed(1));
                  useLandmarkStore.getState().updateLandmark2D('spine_points', index, constrainedX, constrainedY);
                }
              }
            }
          }
          // Spine points also need fetchLift to snap to mesh surface vertex
          // use constrainedX/constrainedY so 3D lift position matches constrained 2D display
          try {
            const res = await fetchLift(currentId, constrainedX, constrainedY);
            useLandmarkStore.getState().updateLandmark3D('spine_points', index, res);
          } catch {}
          draw();
        } else {
          const res = await fetchLift(currentId, liftPt.x, liftPt.y);
          useLandmarkStore.getState().updateLandmark3D(name, index, res);
          const spineIdx = SPINE_CONSTRAINT_PAIRS.indexOf(name);
          if (spineIdx >= 0 && !SPINE_HIDDEN[spineIdx]) {
            const pair = useLandmarkStore.getState().landmarks[name];
            if (pair && pair[0] && pair[1] && mapping) {
              const a = landmarkToDisplay(pair[0], mapping);
              const b = landmarkToDisplay(pair[1], mapping);
              if (a && b) {
                const mx = parseFloat(((a.x + b.x) / 2).toFixed(1));
                const my = parseFloat(((a.y + b.y) / 2).toFixed(1));
                useLandmarkStore.getState().updateLandmark2D('spine_points', spineIdx, mx, my);
                draw();
              }
            }
          }
        }
      }
    } catch (err) {
      console.error('lift failed:', err);
    } finally {
      setBusy(false);
      if (canvasRef.current) canvasRef.current.style.cursor = 'default';
    }
  }, [currentId, landmarks, fetchLift, draw]);

  // All three must be ready AND belong to the same subject
  const dataConsistent = imgReady && mapping && loadedForSubject.current === currentId;
  const showSpinner = loading || !dataConsistent;

  // Report readiness to parent for opaque overlay coordination
  const wasReady = useRef(false);
  useEffect(() => {
    if (dataConsistent && !wasReady.current) {
      wasReady.current = true;
      onReady?.();
    } else if (!dataConsistent) {
      wasReady.current = false;
    }
  }, [dataConsistent, onReady]);

  return (
    <Box ref={containerRef} sx={{ width: '100%', height: '100%', position: 'relative', bgcolor: 'background.default', overflow: 'hidden' }}>
      <canvas
        ref={canvasRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onContextMenu={onContextMenu}
        style={{ width: '100%', height: '100%', display: 'block', cursor: busy ? 'wait' : 'default' }}
      />
      {/* Busy overlay during lift — blocks all interaction */}
      {busy && (
        <Box sx={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
          bgcolor: 'rgba(13,17,25,0.6)', zIndex: 50,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 1,
        }}>
          <Box sx={{ width: 24, height: 24, border: '2px solid', borderColor: 'divider', borderTopColor: 'primary.main', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
          <Typography sx={{ color: 'text.secondary', fontSize: 12 }}>更新中...</Typography>
        </Box>
      )}
      {showSpinner && !busy && (
        <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, pointerEvents: 'none' }}>
          <Box sx={{ width: 28, height: 28, border: '2px solid', borderColor: 'divider', borderTopColor: 'primary.main', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
          <Typography sx={{ color: 'text.secondary', fontSize: 13 }}>加载中...</Typography>
        </Box>
      )}
      {landmarkError && (
        <Box sx={{ position: 'absolute', bottom: 8, left: 8, bgcolor: 'rgba(255,0,0,0.7)', color: '#fff', px: 1, py: 0.5, borderRadius: 1, fontSize: 11 }}>
          {landmarkError}
        </Box>
      )}
    </Box>
  );
}
