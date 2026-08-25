// src/hooks/useCanvasRenderer.ts
import { useCallback } from 'react';
import type { Landmarks, Mapping } from '../types';

const COLORS: Record<string, string> = {
  neck_root: '#00FFFF', shoulder_transition: '#FF4444',
  scapular_peaks: '#44FF44', axilla: '#FF44FF',
  waist: '#FFFF44', waist_lower: '#FF8C00', spine_points: '#FFFFFF',
};
const ZH: Record<string, string> = {
  neck_root: '颈根', shoulder_transition: '肩臂转点', scapular_peaks: '肩胛峰',
  axilla: '腋窝', waist: '腰部', waist_lower: '腰下缘', spine_points: '脊柱',
};
const SPINE_HIDDEN: Record<number, boolean> = {};
const MARKER_RADIUS = 8;

function getPt(pt: number[] | null | undefined): { x: number; y: number } | null {
  if (!pt || !Array.isArray(pt)) return null;
  if (pt[0] == null) return null;
  return { x: pt[0], y: pt[1] };
}

/** 将原始 3D landmark 通过 PCA 参数投影到 2D 显示空间（PC2→X, PC1→Y）。 */
export function landmarkToDisplay(pt: number[], mapping: Mapping): { x: number; y: number } | null {
  if (!pt) return null;
  if (!mapping.pca_mean || !mapping.pca_Vt) return getPt(pt);
  const mean = mapping.pca_mean as [number, number, number];
  const Vt = mapping.pca_Vt as [[number, number, number], [number, number, number], [number, number, number]];
  const cx = pt[0] - mean[0], cy = pt[1] - mean[1], cz = (pt[2] ?? 0) - mean[2];
  const pc1 = cx * Vt[0][0] + cy * Vt[0][1] + cz * Vt[0][2];
  const pc2 = cx * Vt[1][0] + cy * Vt[1][1] + cz * Vt[1][2];
  return { x: pc2, y: pc1 };
}

export function dataToPixel(x: number, y: number, mapping: Mapping, w: number, h: number) {
  return {
    x: (x - mapping.x_data_range[0]) / (mapping.x_data_range[1] - mapping.x_data_range[0]) * w,
    y: (1 - (y - mapping.y_data_range[0]) / (mapping.y_data_range[1] - mapping.y_data_range[0])) * h,
  };
}

export function pixelToData(px: number, py: number, mapping: Mapping, w: number, h: number) {
  return {
    x: mapping.x_data_range[0] + (px / w) * (mapping.x_data_range[1] - mapping.x_data_range[0]),
    y: mapping.y_data_range[0] + (1 - py / h) * (mapping.y_data_range[1] - mapping.y_data_range[0]),
  };
}

export function useCanvasRenderer(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  img: HTMLImageElement | null,
  mapping: Mapping | null,
  landmarks: Landmarks,
  rotationAngle: number,
  dragTarget: { name: string; index: number } | null,
  pendingLandmark?: { name: string; index: number } | null,
  mousePosRef?: React.MutableRefObject<{ x: number; y: number } | null>,
) {
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mapping) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    if (w === 0 || h === 0) return;

    // Background — matches curvature image facecolor (#1a1a1a)
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, w, h);

    // Don't render landmarks without an image — avoids ghost artifacts
    if (!img) return;

    // Rotation
    ctx.save();
    const ang = rotationAngle * Math.PI / 180;
    if (ang !== 0) {
      ctx.translate(w / 2, h / 2);
      ctx.rotate(ang);
      ctx.translate(-w / 2, -h / 2);
    }

    // Curvature image
    ctx.drawImage(img, 0, 0, w, h);

    // Spine connecting lines
    const spinePts = landmarks['spine_points'];
    if (spinePts) {
      // 解剖顺序：0(颈根)→1(肩胛峰)→2(腋窝)→m(中背)→3(腰部)→4(腰下缘)
      const LINE_INDICES = [0, 1, 2, 5, 3, 4];
      const visible: { pt: ReturnType<typeof getPt>; i: number }[] = [];
      for (const i of LINE_INDICES) {
        if (i >= spinePts.length || SPINE_HIDDEN[i]) continue;
        const raw = spinePts[i];
        if (raw == null) continue;
        const pt = landmarkToDisplay(raw, mapping);
        if (!pt) continue;
        visible.push({ pt, i });
      }
      if (visible.length >= 2) {
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = 'rgba(255,255,0,0.8)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        const p0 = dataToPixel(visible[0].pt.x, visible[0].pt.y, mapping, w, h);
        ctx.moveTo(p0.x, p0.y);
        for (let j = 1; j < visible.length; j++) {
          const p = dataToPixel(visible[j].pt.x, visible[j].pt.y, mapping, w, h);
          ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Draw L/R connection line during drag
    const SPINE_CONSTRAINT_PAIRS = ['neck_root', 'scapular_peaks', 'axilla', 'waist'];
    if (dragTarget) {
      const { name, index } = dragTarget;
      if (name === "spine_points" && index === 5) {
        // mid_back : draw AB line (mid axilla/waist L ↔ mid axilla/waist R)
        const getMid = (name1: string, name2: string, side: number) => {
          const p1 = landmarks[name1]?.[side];
          const p2 = landmarks[name2]?.[side];
          if (!p1 || !p2) return null;
          const d1 = landmarkToDisplay(p1, mapping);
          const d2 = landmarkToDisplay(p2, mapping);
          if (!d1 || !d2) return null;
          return dataToPixel((d1.x + d2.x) / 2, (d1.y + d2.y) / 2, mapping, w, h);
        };
        const A = getMid('axilla', 'waist', 0);
        const B = getMid('axilla', 'waist', 1);
        if (A && B) {
          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = 'rgba(255,255,255,0.7)';
          ctx.lineWidth = 4;
          ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke();
          ctx.setLineDash([]);
        }
      } else if (name === 'spine_points' && index === 4) {
        // P4 free point: draw waist_lower L → P4 → waist_lower R polyline
        const wp = landmarks['waist_lower'];
        if (wp && wp.length >= 2 && wp[0] && wp[1]) {
          const p4pts = landmarks['spine_points']?.[4];
          if (p4pts) {
            const pLd = landmarkToDisplay(wp[0], mapping);
            const pRd = landmarkToDisplay(wp[1], mapping);
            const p4d = landmarkToDisplay(p4pts, mapping);
            if (pLd && pRd && p4d) {
              const a = dataToPixel(pLd.x, pLd.y, mapping, w, h);
              const b = dataToPixel(p4d.x, p4d.y, mapping, w, h);
              const c = dataToPixel(pRd.x, pRd.y, mapping, w, h);
              ctx.setLineDash([6, 4]);
              ctx.strokeStyle = 'rgba(255,255,255,0.7)';
              ctx.lineWidth = 4;
              ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.lineTo(c.x, c.y); ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        }
      } else if (name === 'waist_lower') {
        // waist_lower L → P4 → waist_lower R polyline (no direct L-R line)
        const wp = landmarks['waist_lower'];
        const p4pts = landmarks['spine_points']?.[4];
        if (wp && wp.length >= 2 && wp[0] && wp[1] && p4pts) {
          const pLd = landmarkToDisplay(wp[0], mapping);
          const pRd = landmarkToDisplay(wp[1], mapping);
          const p4d = landmarkToDisplay(p4pts, mapping);
          if (pLd && pRd && p4d) {
            const a = dataToPixel(pLd.x, pLd.y, mapping, w, h);
            const b = dataToPixel(p4d.x, p4d.y, mapping, w, h);
            const c = dataToPixel(pRd.x, pRd.y, mapping, w, h);
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = 'rgba(255,255,255,0.7)';
            ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.lineTo(c.x, c.y); ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      } else {
        const pairName = name === 'spine_points' ? SPINE_CONSTRAINT_PAIRS[index] : name;
        if (pairName) {
          const pts = landmarks[pairName];
          if (pts && pts.length >= 2) {
            const pL = landmarkToDisplay(pts[0], mapping);
            const pR = landmarkToDisplay(pts[1], mapping);
            if (pL && pR) {
              const a = dataToPixel(pL.x, pL.y, mapping, w, h);
              const b = dataToPixel(pR.x, pR.y, mapping, w, h);
              ctx.setLineDash([6, 4]);
              ctx.strokeStyle = 'rgba(255,255,255,0.7)';
              ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
    }
    }

    // Spine points
    if (spinePts) {
      ctx.font = '11px sans-serif';
      for (let i = 0; i < spinePts.length; i++) {
        if (SPINE_HIDDEN[i]) continue;
        // 跳过未放置的点（全零坐标）
        const raw = spinePts[i];
        if (raw == null) continue;
        const pt = landmarkToDisplay(raw, mapping);
        if (!pt) continue;
        const p = dataToPixel(pt.x, pt.y, mapping, w, h);
        ctx.beginPath(); ctx.arc(p.x, p.y, MARKER_RADIUS + 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, MARKER_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = '#FFFFFF'; ctx.fill();
        ctx.strokeStyle = '#000000'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#FFFFFF';
        ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 4;
        ctx.fillText(i === 5 ? `脊柱 Pm` : `脊柱 P${i}`, p.x + MARKER_RADIUS + 5, p.y - MARKER_RADIUS - 3);
        ctx.shadowBlur = 0;
      }
    }

    // Bilateral landmarks
    for (const [name, pts] of Object.entries(landmarks)) {
      if (name === 'spine_points') continue;
      const color = COLORS[name] || '#FFFFFF';
      ctx.font = 'bold 11px sans-serif';
      const labels = ['L', 'R'];
      for (let i = 0; i < pts.length; i++) {
        const pt = landmarkToDisplay(pts[i], mapping);
        if (!pt) continue;
        const p = dataToPixel(pt.x, pt.y, mapping, w, h);
        ctx.beginPath(); ctx.arc(p.x, p.y, MARKER_RADIUS + 3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, MARKER_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
        ctx.strokeStyle = '#000'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#FFF';
        ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 4;
        ctx.fillText(`${ZH[name] || name} ${labels[i]}`, p.x + MARKER_RADIUS + 5, p.y - MARKER_RADIUS - 3);
        ctx.shadowBlur = 0;
      }
    }

    ctx.restore();

    // Tip text (unrotated)
    if (dragTarget) {
      ctx.fillStyle = 'rgba(255,255,100,0.8)';
      ctx.font = '12px sans-serif';
      ctx.fillText('正在拖拽 — 松手后自动更新', 12, h - 12);
    }

    // Ghost marker for pending placement
    const mp = mousePosRef?.current;
    if (pendingLandmark && mp) {
      // Draw constraint line during pending placement
      if (pendingLandmark.name === 'spine_points') {
        const idx = pendingLandmark.index;
        if (idx < SPINE_CONSTRAINT_PAIRS.length) {
          // P0-P3: L-R line of corresponding bilateral pair
          const pair = landmarks[SPINE_CONSTRAINT_PAIRS[idx]];
          if (pair && pair[0] && pair[1]) {
            const pL = landmarkToDisplay(pair[0], mapping);
            const pR = landmarkToDisplay(pair[1], mapping);
            if (pL && pR) {
              const a = dataToPixel(pL.x, pL.y, mapping, w, h);
              const b = dataToPixel(pR.x, pR.y, mapping, w, h);
              ctx.setLineDash([6, 4]); ctx.strokeStyle = 'rgba(255,255,255,0.7)';
              ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        } else if (idx === 4) {
          // P4: waist_lower L → cursor → waist_lower R polyline
          const wp = landmarks['waist_lower'];
          if (wp && wp.length >= 2 && wp[0] && wp[1]) {
            const pL = landmarkToDisplay(wp[0], mapping);
            const pR = landmarkToDisplay(wp[1], mapping);
            if (pL && pR) {
              const a = dataToPixel(pL.x, pL.y, mapping, w, h);
              const b = dataToPixel(pR.x, pR.y, mapping, w, h);
              ctx.setLineDash([6, 4]); ctx.strokeStyle = 'rgba(255,255,255,0.7)';
              ctx.lineWidth = 4;
              ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mp.x, mp.y); ctx.lineTo(b.x, b.y); ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        } else if (idx === 5) {
          // Pm: AB line (mid axilla+waist L ↔ R)
          const getMid = (n1: string, n2: string, side: number) => {
            const p1 = landmarks[n1]?.[side], p2 = landmarks[n2]?.[side];
            if (!p1 || !p2) return null;
            const d1 = landmarkToDisplay(p1, mapping), d2 = landmarkToDisplay(p2, mapping);
            return d1 && d2 ? dataToPixel((d1.x+d2.x)/2, (d1.y+d2.y)/2, mapping, w, h) : null;
          };
          const A = getMid('axilla','waist',0), B = getMid('axilla','waist',1);
          if (A && B) {
            ctx.setLineDash([6, 4]); ctx.strokeStyle = 'rgba(255,255,255,0.7)';
            ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      } else if (pendingLandmark.name === 'waist_lower') {
        // waist_lower L → cursor → waist_lower R polyline (no direct L-R line)
        const wp = landmarks['waist_lower'];
        if (wp && wp.length >= 2 && wp[0] && wp[1]) {
          const pL = landmarkToDisplay(wp[0], mapping);
          const pR = landmarkToDisplay(wp[1], mapping);
          if (pL && pR) {
            const a = dataToPixel(pL.x, pL.y, mapping, w, h);
            const b = dataToPixel(pR.x, pR.y, mapping, w, h);
            ctx.setLineDash([6, 4]); ctx.strokeStyle = 'rgba(255,255,255,0.7)';
            ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mp.x, mp.y); ctx.lineTo(b.x, b.y); ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      } else {
        // Bilateral landmark: L-R line
        const pair = landmarks[pendingLandmark.name];
        if (pair && pair.length >= 2 && pair[0] && pair[1]) {
          const pL = landmarkToDisplay(pair[0], mapping);
          const pR = landmarkToDisplay(pair[1], mapping);
          if (pL && pR) {
            const a = dataToPixel(pL.x, pL.y, mapping, w, h);
            const b = dataToPixel(pR.x, pR.y, mapping, w, h);
            ctx.setLineDash([6, 4]); ctx.strokeStyle = 'rgba(255,255,255,0.7)';
            ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }

      // Ghost marker circle
      ctx.beginPath();
      ctx.arc(mp.x, mp.y, MARKER_RADIUS + 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '11px sans-serif';
      const label = pendingLandmark.name === 'spine_points'
        ? `脊柱 P${pendingLandmark.index === 5 ? 'm' : pendingLandmark.index}`
        : ZH[pendingLandmark.name] || pendingLandmark.name;
      ctx.fillText(label, mp.x + MARKER_RADIUS + 8, mp.y - MARKER_RADIUS - 3);
    }
  }, [canvasRef, img, mapping, landmarks, rotationAngle, dragTarget, pendingLandmark, mousePosRef]);

  return { draw };
}
