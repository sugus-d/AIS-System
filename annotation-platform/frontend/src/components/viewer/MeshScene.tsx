// src/components/viewer/MeshScene.tsx
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as React from "react";
import { Box } from "@mui/material";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import type { Landmarks } from "../../types";
import { SPINE_CONSTRAINT_PAIRS } from "../../constants";
import { landmarkToDisplay } from "../../hooks/useCanvasRenderer";
import { useSubjectStore } from "../../stores/subjectStore";
import * as THREE from "three";
import { getMeshUrl } from "../../api/subjects";
import { useUIStore } from "../../stores/uiStore";
import { useLandmarkStore } from "../../stores/landmarkStore";

let GLTFLoader: any = null;
async function ensureGLTFLoader() {
  if (!GLTFLoader) {
    const mod = await import("three/examples/jsm/loaders/GLTFLoader.js");
    GLTFLoader = mod.GLTFLoader;
  }
}

// 共享 overlay 请求缓存：按 subjectId-version 去重，避免重复请求
const _overlayCache = new Map<string, Promise<any>>();
function fetchClothOverlay(subjectId: string, version: number): Promise<any> {
  const key = `${subjectId}-v${version}`;
  if (!_overlayCache.has(key)) {
    _overlayCache.set(
      key,
      fetch(`/api/subjects/${subjectId}/mesh/overlay-cloth`).then((r) =>
        r.json(),
      ),
    );
  }
  return _overlayCache.get(key)!;
}

function getPt(
  pt: number[] | null | undefined,
): { x: number; y: number; z: number } | null {
  if (!pt || !Array.isArray(pt)) return null;
  if (pt[0] == null) return null;
  return { x: pt[0], y: pt[1], z: pt[2] };
}

// ── 3D 标注约束：脊柱白点 = 两点连线 × 默认投影视角（PC3）张成的平面 ──
// 白点在该平面内运动 ⇔ 从默认视角（2D 视图）正交投影始终落在对应连线上 → 行为与 2D 一致。
type ConstraintSegment = { A: THREE.Vector3; B: THREE.Vector3 };

/** 脊柱约束点对应的 3D 线段：P0–P3 取对应双边组 L–R；Pm(index 5) 取 A=mid(axilla,waist)ₗ、B=mid(axilla,waist)ᵣ；P4 自由 → null */
function getConstraintSegment(
  index: number,
  landmarks: Landmarks,
): ConstraintSegment | null {
  let A3: number[] | null = null;
  let B3: number[] | null = null;
  if (index >= 0 && index < SPINE_CONSTRAINT_PAIRS.length) {
    const pair = landmarks[SPINE_CONSTRAINT_PAIRS[index]];
    if (pair?.[0] && pair?.[1]) {
      A3 = pair[0];
      B3 = pair[1];
    }
  } else if (index === 5) {
    // mid_back：腋下连线与腰部连线的中线（无两侧对应点）
    const ax = landmarks["axilla"];
    const wa = landmarks["waist"];
    if (ax?.[0] && ax?.[1] && wa?.[0] && wa?.[1]) {
      A3 = [(ax[0][0] + wa[0][0]) / 2, (ax[0][1] + wa[0][1]) / 2, (ax[0][2] + wa[0][2]) / 2];
      B3 = [(ax[1][0] + wa[1][0]) / 2, (ax[1][1] + wa[1][1]) / 2, (ax[1][2] + wa[1][2]) / 2];
    }
  }
  if (!A3 || !B3) return null;
  return {
    A: new THREE.Vector3(A3[0], A3[1], A3[2]),
    B: new THREE.Vector3(B3[0], B3[1], B3[2]),
  };
}

/** 把点投影到线段上（与 2D 视图 projectOnLine 相同的连线约束，PCA 2D 空间） */
function projectOnLine2D(
  p: { x: number; y: number },
  a: { x: number; y: number },
  b: { x: number; y: number },
): { x: number; y: number } {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const len2 = abx * abx + aby * aby;
  if (len2 < 1e-8) return { x: a.x, y: a.y };
  const t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2;
  return { x: a.x + t * abx, y: a.y + t * aby };
}

const COLORS: Record<string, number> = {
  neck_root: 0x00ffff,
  shoulder_transition: 0xff4444,
  scapular_peaks: 0x44ff44,
  axilla: 0xff44ff,
  waist: 0xffff44,
  waist_lower: 0xff8c00,
  spine_points: 0xffffff,
};

// 脊柱轴线：按解剖顺序连接脊柱点（0 颈根 → 1 肩胛 → 2 腋窝 → 5 中背 → 3 腰部 → 4 腰下缘）
const SPINE_LINE_ORDER = [0, 1, 2, 5, 3, 4];

function LandmarkMarkers({ landmarks }: { landmarks: Landmarks }) {
  const group = useMemo(() => {
    const g = new THREE.Group();
    g.renderOrder = 1000; // 标注层置顶
    for (const [name, pts] of Object.entries(landmarks)) {
      if (!Array.isArray(pts)) continue;
      const color = COLORS[name] || 0xffffff;
      for (let i = 0; i < pts.length; i++) {
        const pt = getPt(pts[i]);
        if (!pt) continue;
        // 与 2D marker 样式对标：黑色半透明外圈（模拟 2D 的黑色晕圈+描边）+ 彩色/白色实心圆
        const halo = new THREE.Mesh(
          new THREE.SphereGeometry(6.5, 20, 20),
          new THREE.MeshBasicMaterial({
            color: 0x000000,
            transparent: true,
            opacity: 0.45,
            depthTest: false,
            depthWrite: false,
          }),
        );
        halo.position.set(pt.x, pt.y, pt.z);
        g.add(halo);
        const core = new THREE.Mesh(
          new THREE.SphereGeometry(4.5, 20, 20),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            depthTest: false,
            depthWrite: false,
          }),
        );
        core.position.set(pt.x, pt.y, pt.z);
        g.add(core);
      }
    }
    return g;
  }, [landmarks]);
  return <primitive object={group} />;
}

// ── 脊柱轴线：按解剖顺序连接脊柱点（0颈根→1肩胛→2腋窝→5中背→3腰部→4腰下缘），
// 与 2D 的脊柱连线一致（白色实线），始终置顶显示
function SpineAxisLine({ landmarks }: { landmarks: Landmarks }) {
  const pts = useMemo(() => {
    const spine = landmarks["spine_points"];
    if (!spine) return [];
    const out: [number, number, number][] = [];
    for (const i of SPINE_LINE_ORDER) {
      const p = getPt(spine[i]);
      if (p) out.push([p.x, p.y, p.z]);
    }
    return out;
  }, [landmarks]);
  if (pts.length < 2) return null;
  return (
    <Line
      points={pts}
      color="#ffffff"
      lineWidth={2}
      transparent
      opacity={0.85}
      depthTest={false}
      depthWrite={false}
      renderOrder={1000}
    />
  );
}

function computeLandmarkCenter(landmarks: Landmarks): THREE.Vector3 | null {
  const c = new THREE.Vector3();
  let count = 0;
  for (const pts of Object.values(landmarks)) {
    if (!Array.isArray(pts)) continue;
    for (const pt of pts) {
      const p = getPt(pt);
      if (!p) continue;
      c.x += p.x;
      c.y += p.y;
      c.z += p.z;
      count++;
    }
  }
  if (count === 0) return null;
  c.x /= count;
  c.y /= count;
  c.z /= count;
  return c;
}

// Camera auto-framing — uses mesh bounding box when available, falls back to landmarks
function CameraFramer({
  meshScene,
  landmarks,
  controlsRef,
}: {
  meshScene: THREE.Group | null;
  landmarks: Landmarks;
  controlsRef: React.MutableRefObject<any>;
}) {
  const framed = useRef<string | null>(null);

  useEffect(() => {
    if (!meshScene) return;
    if (framed.current === meshScene.uuid) return; // already framed this mesh
    try {
      const box = new THREE.Box3().setFromObject(meshScene);
      const center = box.getCenter(new THREE.Vector3());
      if (controlsRef.current) {
        controlsRef.current.target.copy(center);
        controlsRef.current.update();
      }
      framed.current = meshScene.uuid;
    } catch (e) {
      /* ignore */
    }
  }, [meshScene]);

  // Fallback: use landmarks center if no mesh yet
  useEffect(() => {
    if (meshScene || framed.current) return;
    const center = computeLandmarkCenter(landmarks);
    if (!center) return;
    if (controlsRef.current) {
      controlsRef.current.target.copy(center);
      controlsRef.current.update();
    }
  }, [meshScene, landmarks]);

  return <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.15} />;
}

// Mesh loader — immediately unmounts old mesh when url changes,
// then loads and renders the new one.
function MeshModel({
  url,
  onMeshReady,
}: {
  url: string;
  onMeshReady: (scene: THREE.Group | null) => void;
}) {
  const sceneRef = useRef<THREE.Group | null>(null);
  const [loaded, setLoaded] = useState(false);
  const notified = useRef(false);

  useEffect(() => {
    // IMMEDIATELY destroy old state — prevents old mesh from persisting
    setLoaded(false);
    sceneRef.current = null;
    notified.current = false;

    let cancelled = false;
    (async () => {
      await ensureGLTFLoader();
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 30_000);
        const res = await fetch(url, { signal: controller.signal });
        clearTimeout(timer);
        const glb = await res.arrayBuffer();
        const loader = new GLTFLoader();
        const gltf = await new Promise<any>((resolve, reject) => {
          loader.parse(glb, "", resolve, reject);
        });
        if (!cancelled) {
          sceneRef.current = gltf.scene;
          setLoaded(true);
        }
      } catch (e: any) {
        const msg = e?.name === "AbortError" ? "模型加载超时" : "模型加载失败";
        console.error("mesh load failed:", e);
        if (!cancelled && !notified.current) {
          notified.current = true;
          onMeshReady(null);
          useUIStore.getState().addToast(`${msg}，请重试`, "error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [url]); // Intentionally only depend on url — onMeshReady is stable from parent

  useEffect(() => {
    if (loaded && !notified.current) {
      notified.current = true;
      onMeshReady(sceneRef.current);
    }
  }, [loaded, onMeshReady]);

  if (!loaded || !sceneRef.current) return null;
  return <primitive object={sceneRef.current} />;
}

// Clothing overlay — 重载 mesh 时（commit 后 meshVersion 自增）自动刷新
function ClothOverlay({ subjectId }: { subjectId: string }) {
  const visible = useUIStore((s) => s.clothOverlay);
  const meshVersion = useUIStore((s) => s.meshVersion);
  const [points, setPoints] = useState<THREE.Points | null>(null);
  const loadKey = `${subjectId}-v${meshVersion}`;
  const loadedFor = useRef<string | null>(null);

  useEffect(() => {
    if (!visible || loadedFor.current === loadKey) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchClothOverlay(subjectId, meshVersion);
        if (!cancelled && data.extra_points?.length) {
          const geo = new THREE.BufferGeometry();
          geo.setFromPoints(
            data.extra_points.map(
              (p: number[]) => new THREE.Vector3(p[0], p[1], p[2]),
            ),
          );
          const mat = new THREE.PointsMaterial({
            color: 0xff6666,
            size: 4,
            transparent: true,
            opacity: 0.6,
            depthTest: true,
            sizeAttenuation: true,
          });
          setPoints(new THREE.Points(geo, mat));
          loadedFor.current = loadKey;
        }
      } catch (e) {
        console.error("cloth overlay:", e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [visible, loadKey, subjectId]);

  if (!visible || !points) return null;
  return <primitive object={points} />;
}

// ── Brush handler — real-time: cached mesh vertices, paints on mouse move ──
function BrushHandler({
  subjectId,
  controlsRef,
  cursorRef,
}: {
  subjectId: string;
  controlsRef: React.MutableRefObject<any>;
  cursorRef: React.RefObject<HTMLDivElement | null>;
}) {
  const brushMode = useUIStore((s) => s.brushMode);
  const brushSize = useUIStore((s) => s.brushSize);
  const clothOverlay = useUIStore((s) => s.clothOverlay);
  const brushPoints = useUIStore((s) => s.brushPoints);
  const addBrushPoints = useUIStore((s) => s.addBrushPoints);
  const { camera, gl, scene } = useThree();
  const drawing = useRef(false);
  const tmpV = useRef(new THREE.Vector3());
  // Cached mesh vertices in world space
  const meshVerts = useRef<number[][]>([]);
  // Cached cloth overlay vertices
  const clothVerts = useRef<{ pos: number[]; idx: number }[]>([]);
  // Green cloth restore particles ref
  const greenClothRef = useRef<THREE.Points | null>(null);
  const restoreClothPainted = useRef<Set<number>>(new Set()); // 累加 cloth 索引
  // Set of already-painted vertex keys (dedup)
  const painted = useRef<Set<string>>(new Set());
  // Last paint time — time-based throttle
  const lastPaintTime = useRef(0);
  // Restore mode: cursor positions collected during drag
  const restoreStrokePts = useRef<number[][]>([]);

  // Build vertex cache when entering brush mode (mesh is loaded by then)
  useEffect(() => {
    if (!brushMode) return;
    const arr: number[][] = [];
    scene.updateMatrixWorld(true);
    scene.traverse((obj) => {
      const m = obj as THREE.Mesh;
      if (!m.isMesh || !m.geometry) return;
      const pos = m.geometry.getAttribute("position");
      if (!pos) return;
      const fa = pos.array as Float32Array;
      const v = new THREE.Vector3();
      for (let i = 0; i < fa.length; i += 3) {
        v.set(fa[i], fa[i + 1], fa[i + 2]);
        v.applyMatrix4(m.matrixWorld);
        arr.push([v.x, v.y, v.z]);
      }
    });
    meshVerts.current = arr;
    painted.current.clear();
  }, [brushMode, scene]);

  // Fetch cloth overlay data when needed
  const brushMeshVersion = useUIStore((s) => s.meshVersion);
  const clothLoadedForRef = useRef<string | null>(null);
  useEffect(() => {
    const loadKey = `${subjectId}-v${brushMeshVersion}`;
    if (!clothOverlay || !subjectId || clothLoadedForRef.current === loadKey)
      return;
    clothLoadedForRef.current = loadKey;
    (async () => {
      try {
        const data = await fetchClothOverlay(subjectId, brushMeshVersion);
        if (data.extra_points && data.extra_indices) {
          clothVerts.current = data.extra_points.map(
            (p: number[], i: number) => ({
              pos: p,
              idx: data.extra_indices[i],
            }),
          );
        }
      } catch {
        clothVerts.current = [];
      }
    })();
  }, [clothOverlay, subjectId, brushMeshVersion]);

  // Cleanup green particles when cloth overlay turned off
  useEffect(() => {
    if (!clothOverlay) {
      if (greenClothRef.current) {
        scene.remove(greenClothRef.current);
        greenClothRef.current = null;
      }
      restoreClothPainted.current.clear();
    }
  }, [clothOverlay, scene]);

  // Watch store for external clearRestoreCloth calls (from Toolbar clear/commit/exitBrush)
  const storeLen = useUIStore((s) => s.restoreClothIndices.length);
  useEffect(() => {
    if (storeLen === 0 && restoreClothPainted.current.size > 0) {
      restoreClothPainted.current.clear();
      if (greenClothRef.current) {
        scene.remove(greenClothRef.current);
        greenClothRef.current = null;
      }
    }
  }, [storeLen, scene]);

  // 清空按钮将 brushPoints 置空后同步清除 painted 集合，允许重新绘制
  const bpLen = useUIStore((s) => s.brushPoints.length);
  useEffect(() => {
    if (brushMode && bpLen === 0 && painted.current.size > 0) {
      painted.current.clear();
    }
  }, [bpLen, brushMode]);

  // Screen-space cursor — hidden when circle edge touches canvas boundary
  useEffect(() => {
    const canvas = gl.domElement;
    const cam = camera as THREE.PerspectiveCamera;
    const fovRad = (cam.fov * Math.PI) / 180;
    const tanHalfFov = Math.tan(fovRad / 2);
    const onMouse = (e: MouseEvent) => {
      if (!cursorRef.current) return;
      const depth = controlsRef.current?.target
        ? camera.position.distanceTo(controlsRef.current.target)
        : camera.position.length();
      const pixelsPerUnit = canvas.clientHeight / (2 * depth * tanHalfFov);
      const diameter = Math.max(6, brushSize * 2 * pixelsPerUnit * 0.5);
      const r = diameter / 2;
      const rect = canvas.getBoundingClientRect();
      const inside =
        e.clientX - r >= rect.left &&
        e.clientX + r <= rect.right &&
        e.clientY - r >= rect.top &&
        e.clientY + r <= rect.bottom;
      cursorRef.current.style.display = inside ? "" : "none";
      if (!inside) return;
      cursorRef.current.style.left = `${e.clientX - r}px`;
      cursorRef.current.style.top = `${e.clientY - r}px`;
      cursorRef.current.style.width = `${diameter}px`;
      cursorRef.current.style.height = `${diameter}px`;
    };
    canvas.addEventListener("mousemove", onMouse);
    return () => canvas.removeEventListener("mousemove", onMouse);
  }, [gl, brushSize, camera, cursorRef]);

  // Disable orbit controls when brush mode active
  useEffect(() => {
    if (controlsRef.current) controlsRef.current.enabled = !brushMode;
  }, [brushMode, controlsRef]);

  // Paint helper — screen-space projection: mark verts within cursor disk
  const paintAt = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = gl.domElement;
      const cw = canvas.clientWidth,
        ch = canvas.clientHeight;
      if (!cw || !ch) return;

      // Convert viewport coords → canvas-relative coords (canvas may be offset by panels/title bar)
      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;

      const depth = controlsRef.current?.target
        ? camera.position.distanceTo(controlsRef.current.target)
        : camera.position.length();
      const fovRad = ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180;
      const cr = ((brushSize * ch) / (2 * depth * Math.tan(fovRad / 2))) * 0.5;
      const crSq = cr * cr;

      // ── 遮挡计算：第一遍找光标内最近的顶点（纯深度比较，不依赖 mesh 对象）──
      const camPos = camera.position;
      let nearestCamDistSq = Infinity;
      for (const v of meshVerts.current) {
        tmpV.current.set(v[0], v[1], v[2]).project(camera);
        const sx = ((tmpV.current.x + 1) / 2) * cw;
        const sy = ((1 - tmpV.current.y) / 2) * ch;
        if ((sx - px) * (sx - px) + (sy - py) * (sy - py) > crSq) continue;
        const dx = v[0] - camPos.x,
          dy = v[1] - camPos.y,
          dz = v[2] - camPos.z;
        const dSq = dx * dx + dy * dy + dz * dz;
        if (dSq < nearestCamDistSq) nearestCamDistSq = dSq;
      }

      let occLimitSq = Infinity;
      if (nearestCamDistSq !== Infinity) {
        const nearestDist = Math.sqrt(nearestCamDistSq);
        // 笔刷在 mesh 表面的世界空间半径，≈ brushSize/2
        // 作为遮挡容差：同表面上 nearestDist ± 这范围内的顶点不被遮挡
        const worldR = brushSize * 0.5;
        occLimitSq = (nearestDist + worldR) ** 2;
      }

      if (brushMode === "erase") {
        const found: number[][] = [];
        for (const v of meshVerts.current) {
          const key = `${v[0].toFixed(1)},${v[1].toFixed(1)},${v[2].toFixed(1)}`;
          if (painted.current.has(key)) continue;
          tmpV.current.set(v[0], v[1], v[2]).project(camera);
          const sx = ((tmpV.current.x + 1) / 2) * cw;
          const sy = ((1 - tmpV.current.y) / 2) * ch;
          if ((sx - px) * (sx - px) + (sy - py) * (sy - py) <= crSq) {
            // 遮挡检查：比表面更深的顶点跳过（不依赖 raycaster）
            const dx = v[0] - camPos.x,
              dy = v[1] - camPos.y,
              dz = v[2] - camPos.z;
            if (dx * dx + dy * dy + dz * dz > occLimitSq) continue;
            painted.current.add(key);
            found.push(v);
          }
        }
        if (found.length > 0) addBrushPoints(found);
        // 擦除时也移除重叠的 cloth 恢复标记
        if (
          clothOverlay &&
          restoreClothPainted.current.size > 0 &&
          clothVerts.current.length > 0
        ) {
          let removed = false;
          for (const cv of clothVerts.current) {
            if (!restoreClothPainted.current.has(cv.idx)) continue;
            tmpV.current.set(cv.pos[0], cv.pos[1], cv.pos[2]).project(camera);
            const sx = ((tmpV.current.x + 1) / 2) * cw;
            const sy = ((1 - tmpV.current.y) / 2) * ch;
            if ((sx - px) * (sx - px) + (sy - py) * (sy - py) <= crSq) {
              restoreClothPainted.current.delete(cv.idx);
              removed = true;
            }
          }
          if (removed) {
            const allPts: number[][] = [];
            const allIdx: number[] = [];
            for (const cv of clothVerts.current) {
              if (restoreClothPainted.current.has(cv.idx)) {
                allPts.push(cv.pos);
                allIdx.push(cv.idx);
              }
            }
            useUIStore.getState().setRestoreCloth(allPts, allIdx);
            if (greenClothRef.current) {
              scene.remove(greenClothRef.current);
              greenClothRef.current = null;
            }
            if (allPts.length > 0) {
              const geo = new THREE.BufferGeometry();
              geo.setFromPoints(
                allPts.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
              );
              const mat = new THREE.PointsMaterial({
                color: 0x44ff44,
                size: 5,
                depthTest: true,
                transparent: true,
                opacity: 0.7,
              });
              const pts = new THREE.Points(geo, mat);
              pts.renderOrder = 1;
              scene.add(pts);
              greenClothRef.current = pts;
            }
          }
        }
      } else if (brushMode === "restore") {
        // ── Cloth restore: 布料顶点也用自身深度做遮挡 ──
        if (clothOverlay && clothVerts.current.length > 0) {
          // 第一遍：找光标内最近的布料顶点
          let nearestClothDistSq = Infinity;
          for (const cv of clothVerts.current) {
            if (restoreClothPainted.current.has(cv.idx)) continue;
            tmpV.current.set(cv.pos[0], cv.pos[1], cv.pos[2]).project(camera);
            const sx = ((tmpV.current.x + 1) / 2) * cw;
            const sy = ((1 - tmpV.current.y) / 2) * ch;
            if ((sx - px) * (sx - px) + (sy - py) * (sy - py) > crSq) continue;
            const dx = cv.pos[0] - camPos.x,
              dy = cv.pos[1] - camPos.y,
              dz = cv.pos[2] - camPos.z;
            const dSq = dx * dx + dy * dy + dz * dz;
            if (dSq < nearestClothDistSq) nearestClothDistSq = dSq;
          }
          let clothOccLimitSq = Infinity;
          if (nearestClothDistSq !== Infinity) {
            const nearestClothDist = Math.sqrt(nearestClothDistSq);
            clothOccLimitSq = (nearestClothDist + brushSize * 0.5) ** 2;
          }
          // 第二遍：标记未遮挡的布料顶点
          for (const cv of clothVerts.current) {
            if (restoreClothPainted.current.has(cv.idx)) continue;
            tmpV.current.set(cv.pos[0], cv.pos[1], cv.pos[2]).project(camera);
            const sx = ((tmpV.current.x + 1) / 2) * cw;
            const sy = ((1 - tmpV.current.y) / 2) * ch;
            if ((sx - px) * (sx - px) + (sy - py) * (sy - py) <= crSq) {
              const dx = cv.pos[0] - camPos.x,
                dy = cv.pos[1] - camPos.y,
                dz = cv.pos[2] - camPos.z;
              if (dx * dx + dy * dy + dz * dz > clothOccLimitSq) continue;
              restoreClothPainted.current.add(cv.idx);
            }
          }
          // 从累加集合重建绿色粒子
          const allPts: number[][] = [];
          const allIdx: number[] = [];
          for (const cv of clothVerts.current) {
            if (restoreClothPainted.current.has(cv.idx)) {
              allPts.push(cv.pos);
              allIdx.push(cv.idx);
            }
          }
          useUIStore.getState().setRestoreCloth(allPts, allIdx);
          if (greenClothRef.current) {
            scene.remove(greenClothRef.current);
            greenClothRef.current = null;
          }
          if (allPts.length > 0) {
            const geo = new THREE.BufferGeometry();
            geo.setFromPoints(
              allPts.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
            );
            const mat = new THREE.PointsMaterial({
              color: 0x44ff44,
              size: 5,
              depthTest: true,
              transparent: true,
              opacity: 0.7,
            });
            const pts = new THREE.Points(geo, mat);
            pts.renderOrder = 1;
            scene.add(pts);
            greenClothRef.current = pts;
          }
        }
        // ── Cancel erase marks + occlusion ──
        restoreStrokePts.current.push([px, py, cr, occLimitSq]);
        const currentPts = useUIStore.getState().brushPoints;
        if (!currentPts.length) return;
        const keep: number[][] = [];
        for (const bp of currentPts) {
          const key = `${bp[0].toFixed(1)},${bp[1].toFixed(1)},${bp[2].toFixed(1)}`;
          tmpV.current.set(bp[0], bp[1], bp[2]).project(camera);
          const sx = ((tmpV.current.x + 1) / 2) * cw;
          const sy = ((1 - tmpV.current.y) / 2) * ch;
          if ((sx - px) * (sx - px) + (sy - py) * (sy - py) <= crSq) {
            // 遮挡检查：被挡住的擦除标记不恢复
            const dx = bp[0] - camPos.x,
              dy = bp[1] - camPos.y,
              dz = bp[2] - camPos.z;
            if (dx * dx + dy * dy + dz * dz > occLimitSq) {
              keep.push(bp);
              continue;
            }
            painted.current.delete(key);
          } else {
            keep.push(bp);
          }
        }
        if (keep.length !== currentPts.length) {
          useUIStore.getState().clearBrushPoints();
          useUIStore.getState().addBrushPoints(keep);
        }
      }
    },
    [brushMode, brushSize, camera, gl, addBrushPoints, subjectId, scene, clothOverlay],
  );

  // Pointer events — use screen coords directly, no raycast
  useEffect(() => {
    if (!brushMode) return;
    const canvas = gl.domElement;

    const onDown = (e: PointerEvent) => {
      drawing.current = true;
      if (brushMode === "restore") restoreStrokePts.current = [];
      paintAt(e.clientX, e.clientY);
    };
    const onMove = (e: PointerEvent) => {
      if (!drawing.current) return;
      const now = Date.now();
      if (now - lastPaintTime.current < 50) return;
      lastPaintTime.current = now;
      paintAt(e.clientX, e.clientY);
    };
    const onUp = () => {
      drawing.current = false;
      if (brushMode === "restore" && restoreStrokePts.current.length > 0) {
        const currentPts = useUIStore.getState().brushPoints;
        const keep: number[][] = [];
        for (const bp of currentPts) {
          const key = `${bp[0].toFixed(1)},${bp[1].toFixed(1)},${bp[2].toFixed(1)}`;
          if (!painted.current.has(key)) {
            keep.push(bp);
            continue;
          }
          let near = false;
          for (const sp of restoreStrokePts.current) {
            const occLimitSq = sp[3] as number;
            // 遮挡检查：比 stroke 时刻的表面更深的顶点不恢复
            const camDx = bp[0] - camera.position.x,
              camDy = bp[1] - camera.position.y,
              camDz = bp[2] - camera.position.z;
            if (camDx * camDx + camDy * camDy + camDz * camDz > occLimitSq)
              continue;
            tmpV.current.set(bp[0], bp[1], bp[2]).project(camera);
            const sx = ((tmpV.current.x + 1) / 2) * canvas.clientWidth;
            const sy = ((1 - tmpV.current.y) / 2) * canvas.clientHeight;
            const dx = sx - sp[0],
              dy = sy - sp[1];
            if (dx * dx + dy * dy <= sp[2] * sp[2]) {
              near = true;
              break;
            }
          }
          if (!near) keep.push(bp);
          else painted.current.delete(key);
        }
        if (keep.length !== currentPts.length) {
          useUIStore.getState().clearBrushPoints();
          useUIStore.getState().addBrushPoints(keep);
        }
        restoreStrokePts.current = [];
      }
    };

    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      drawing.current = false;
    };
  }, [brushMode, camera, gl, paintAt]);

  // Red particle overlay
  const particlesRef = useRef<THREE.Points | null>(null);
  useEffect(() => {
    if (!brushPoints || brushPoints.length === 0) {
      if (particlesRef.current) {
        scene.remove(particlesRef.current);
        particlesRef.current = null;
      }
      return;
    }
    const geo = new THREE.BufferGeometry();
    geo.setFromPoints(
      brushPoints.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
    );
    const mat = new THREE.PointsMaterial({
      color: 0xff4444,
      size: 4,
      transparent: true,
      opacity: 0.6,
      depthTest: true,
    });
    const np = new THREE.Points(geo, mat);
    if (particlesRef.current) scene.remove(particlesRef.current);
    scene.add(np);
    particlesRef.current = np;
    return () => {
      if (particlesRef.current) scene.remove(particlesRef.current);
    };
  }, [brushPoints, scene]);

  return null;
}

// ── 标注辅助线：与 2D 一致——拖脊柱点/放置时显示约束线，拖双边点时显示该组 L–R 线，
// P4 / waist_lower 显示 waist_lower L → P4 → waist_lower R 折线（白色虚线 [6,4]、4px、0.7 透明度）
function SpineConstraintGuide({
  active,
}: {
  active: { name: string; index: number } | null;
}) {
  const pendingLandmark = useUIStore((s) => s.pendingLandmark);
  const landmarks = useLandmarkStore((s) => s.landmarks);
  const activePoint = active ?? pendingLandmark;
  const pts = useMemo(() => {
    if (!activePoint) return null;
    if (activePoint.name === "spine_points") {
      if (activePoint.index === 4) {
        // P4 自由点：waist_lower L → P4 → waist_lower R（与 2D polyline 一致）
        const wp = landmarks["waist_lower"];
        const p4 = landmarks["spine_points"]?.[4];
        if (wp?.[0] && wp?.[1] && p4) {
          return [
            new THREE.Vector3(wp[0][0], wp[0][1], wp[0][2]),
            new THREE.Vector3(p4[0], p4[1], p4[2]),
            new THREE.Vector3(wp[1][0], wp[1][1], wp[1][2]),
          ];
        }
        return null;
      }
      // P0–P3：对应双边组 L–R 连线；Pm(5)：腋下-腰部中线
      const seg = getConstraintSegment(activePoint.index, landmarks);
      if (!seg) return null;
      return [seg.A, seg.B];
    }
    // 双边点：该组 L–R 3D 线段（与 2D 拖拽时画的连线一致）
    const pair = landmarks[activePoint.name];
    if (!pair?.[0] || !pair?.[1]) return null;
    if (activePoint.name === "waist_lower") {
      // waist_lower 拖拽：waist_lower L → P4 → waist_lower R 折线（与 2D 一致，无直接 L–R 线）
      const p4 = landmarks["spine_points"]?.[4];
      if (!p4) return null;
      return [
        new THREE.Vector3(pair[0][0], pair[0][1], pair[0][2]),
        new THREE.Vector3(p4[0], p4[1], p4[2]),
        new THREE.Vector3(pair[1][0], pair[1][1], pair[1][2]),
      ];
    }
    return [
      new THREE.Vector3(pair[0][0], pair[0][1], pair[0][2]),
      new THREE.Vector3(pair[1][0], pair[1][1], pair[1][2]),
    ];
  }, [activePoint, landmarks]);
  if (!pts || pts.length < 2) return null;
  return (
    <Line
      points={pts.map((p) => p.toArray())}
      color="#ffffff"
      lineWidth={4}
      dashed
      dashSize={6}
      gapSize={4}
      transparent
      opacity={0.7}
      depthTest={false}
      depthWrite={false}
      renderOrder={1000}
    />
  );
}

// ── Landmark annotation handler — 3D 点击放置 / 拖拽移动 landmark ──
function LandmarkAnnotationHandler({
  meshScene,
  controlsRef,
  onConstraint,
}: {
  meshScene: THREE.Group | null;
  controlsRef: React.MutableRefObject<any>;
  onConstraint?: (c: { name: string; index: number } | null) => void;
}) {
  const { camera, gl } = useThree();
  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const dragTarget = useRef<{ name: string; index: number } | null>(null);
  // 拖拽期间记录的最终约束 PCA 坐标，松手时用于 lift 贴到 mesh
  const lastDrag = useRef<{ name: string; index: number; pca: { x: number; y: number } } | null>(null);
  const tmpV = useRef(new THREE.Vector3());
  const pendingLandmark = useUIStore((s) => s.pendingLandmark);

  // 放置期间禁用 orbit controls，避免旋转干扰点击
  useEffect(() => {
    if (controlsRef.current) controlsRef.current.enabled = !pendingLandmark;
    if (gl.domElement) gl.domElement.style.cursor = pendingLandmark ? "crosshair" : "default";
  }, [pendingLandmark, controlsRef, gl]);

  // 屏幕空间找最近 marker（拖拽命中）
  const findNearbyMarker = useCallback(
    (clientX: number, clientY: number) => {
      const rect = gl.domElement.getBoundingClientRect();
      const landmarks = useLandmarkStore.getState().landmarks;
      let best: { name: string; index: number } | null = null;
      let bestDist = 18;
      for (const [name, pts] of Object.entries(landmarks)) {
        if (!Array.isArray(pts)) continue;
        for (let i = 0; i < pts.length; i++) {
          const pt = pts[i];
          if (!pt || !Array.isArray(pt) || pt[0] == null) continue;
          tmpV.current.set(pt[0], pt[1], pt[2]).project(camera);
          const sx = ((tmpV.current.x + 1) / 2) * rect.width + rect.left;
          const sy = ((1 - tmpV.current.y) / 2) * rect.height + rect.top;
          const d = Math.hypot(sx - clientX, sy - clientY);
          if (d < bestDist) {
            bestDist = d;
            best = { name, index: i };
          }
        }
      }
      return best;
    },
    [camera, gl],
  );

  const mapping = useLandmarkStore((s) => s.mapping);
  // PCA 投影参数（与 2D 视图 / lift 同一套）：pca 原点 mean + 三个主轴。
  // 2D 显示 PC1→Y、PC2→X；PC3 为默认投影视角方向（2D→3D lift 的 ray-cast 方向）。
  const pc = useMemo(() => {
    if (!mapping?.pca_mean || !mapping?.pca_Vt) return null;
    const mean = mapping.pca_mean;
    const Vt = mapping.pca_Vt;
    return {
      mean: new THREE.Vector3(mean[0], mean[1], mean[2]),
      v0: new THREE.Vector3(Vt[0][0], Vt[0][1], Vt[0][2]).normalize(), // PC1 → Y
      v1: new THREE.Vector3(Vt[1][0], Vt[1][1], Vt[1][2]).normalize(), // PC2 → X
      v2: new THREE.Vector3(Vt[2][0], Vt[2][1], Vt[2][2]).normalize(), // PC3：默认视角方向
    };
  }, [mapping]);

  // 屏幕鼠标射线 ∩ 固定 PCA 投影平面（法向量 PC3，过 pca 原点）→ PCA 2D 坐标（与 2D 视图同一坐标空间）。
  // 视角几乎与投影平面平行时返回 null（保持当前位置，避免落点跑到无穷远）。
  const screenToPCA = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } | null => {
      if (!pc) return null;
      const rect = gl.domElement.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
      const ny = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(new THREE.Vector2(nx, ny), camera);
      const ray = raycaster.ray;
      const denom = ray.direction.dot(pc.v2);
      if (Math.abs(denom) < 1e-4) return null;
      const t = pc.mean.clone().sub(ray.origin).dot(pc.v2) / denom;
      if (t <= 0) return null;
      const d = ray.origin
        .clone()
        .add(ray.direction.clone().multiplyScalar(t))
        .sub(pc.mean);
      return { x: d.dot(pc.v1), y: d.dot(pc.v0) };
    },
    [pc, camera, gl, raycaster],
  );

  // PCA 2D → 平面 3D（PC3 分量为 0）：拖拽期间的即时预览位置
  const pcaToPlane3D = useCallback(
    (pca: { x: number; y: number }): THREE.Vector3 | null => {
      if (!pc) return null;
      return pc.mean
        .clone()
        .add(pc.v0.clone().multiplyScalar(pca.y))
        .add(pc.v1.clone().multiplyScalar(pca.x));
    },
    [pc],
  );

  // 2D 连线约束（与 2D 视图 projectOnLine 同一逻辑）：P0–P3 → 对应双边组 L–R 连线，Pm(5) → 腋下-腰部中线，P4 自由
  const apply2DSpineConstraint = useCallback(
    (index: number, pca: { x: number; y: number }, lm: Landmarks): { x: number; y: number } => {
      if (!mapping) return pca;
      if (index >= 0 && index < SPINE_CONSTRAINT_PAIRS.length) {
        const pair = lm[SPINE_CONSTRAINT_PAIRS[index]];
        if (pair?.[0] && pair?.[1]) {
          const a = landmarkToDisplay(pair[0], mapping);
          const b = landmarkToDisplay(pair[1], mapping);
          if (a && b) return projectOnLine2D(pca, a, b);
        }
      } else if (index === 5) {
        const ax = lm["axilla"];
        const wa = lm["waist"];
        if (ax?.[0] && ax?.[1] && wa?.[0] && wa?.[1]) {
          const aL = landmarkToDisplay(ax[0], mapping);
          const aR = landmarkToDisplay(ax[1], mapping);
          const wL = landmarkToDisplay(wa[0], mapping);
          const wR = landmarkToDisplay(wa[1], mapping);
          if (aL && aR && wL && wR) {
            const A = { x: (aL.x + wL.x) / 2, y: (aL.y + wL.y) / 2 };
            const B = { x: (aR.x + wR.x) / 2, y: (aR.y + wR.y) / 2 };
            return projectOnLine2D(pca, A, B);
          }
        }
      }
      return pca; // P4 自由点 / 端点缺失时按原始坐标
    },
    [mapping],
  );

  // 与 2D 完全一致：约束后的 PCA 坐标 → 后端 lift（沿 PC3 ray-cast mesh）→ 3D 落点
  const liftPoint = useCallback(
    async (name: string, index: number, pca: { x: number; y: number }): Promise<void> => {
      const currentId = useSubjectStore.getState().currentId;
      if (!currentId) return;
      try {
        const res = await useLandmarkStore.getState().fetchLift(
          currentId,
          parseFloat(pca.x.toFixed(1)),
          parseFloat(pca.y.toFixed(1)),
        );
        useLandmarkStore.getState().updateLandmark3D(name, index, res);
      } catch {
        /* 失败则保留当前预览位置 */
      }
    },
    [],
  );

  // 落位：脊柱点先做 2D 连线约束再 lift；其余点直接 lift
  const commitPoint = useCallback(
    (name: string, index: number, pca: { x: number; y: number }): Promise<void> => {
      const constrained =
        name === "spine_points"
          ? apply2DSpineConstraint(index, pca, useLandmarkStore.getState().landmarks)
          : pca;
      return liftPoint(name, index, constrained);
    },
    [apply2DSpineConstraint, liftPoint],
  );

  // 拖双边点时实时同步对应脊柱中点（与 2D onMouseMove 的 updateSpineMidpoint 行为一致；平面预览）
  const syncSpineLive = useCallback(
    (bilateralName: string) => {
      const lm = useLandmarkStore.getState().landmarks;
      const syncIndex = (idx: number) => {
        const cur = lm["spine_points"]?.[idx];
        if (!cur || !mapping) return;
        const curPCA = landmarkToDisplay(cur, mapping);
        if (!curPCA) return;
        const constrained = apply2DSpineConstraint(idx, curPCA, lm);
        const tmp = pcaToPlane3D(constrained);
        if (tmp) {
          useLandmarkStore.getState().updateLandmark3D("spine_points", idx, {
            x: tmp.x,
            y: tmp.y,
            z: tmp.z,
          });
        }
      };
      const idx = SPINE_CONSTRAINT_PAIRS.indexOf(bilateralName);
      if (idx >= 0) syncIndex(idx);
      if (bilateralName === "axilla" || bilateralName === "waist") syncIndex(5);
    },
    [apply2DSpineConstraint, pcaToPlane3D, mapping],
  );

  // 拖动双边点 / axilla / waist 结束后，把对应脊柱中点重新约束到新连线（2D updateSpineMidpoint / updateMidBack 同源逻辑）
  const syncSpineAfterMove = useCallback(
    (bilateralName: string) => {
      const lm = useLandmarkStore.getState().landmarks;
      const syncIndex = (idx: number) => {
        const cur = lm["spine_points"]?.[idx];
        if (!cur || !mapping) return;
        const curPCA = landmarkToDisplay(cur, mapping);
        if (curPCA) void commitPoint("spine_points", idx, curPCA);
      };
      const idx = SPINE_CONSTRAINT_PAIRS.indexOf(bilateralName);
      if (idx >= 0) syncIndex(idx);
      if (bilateralName === "axilla" || bilateralName === "waist") syncIndex(5);
    },
    [commitPoint, mapping],
  );

  useEffect(() => {
    const canvas = gl.domElement;
    const onDown = (e: PointerEvent) => {
      if (useUIStore.getState().brushMode) return; // 笔刷优先
      const pending = useUIStore.getState().pendingLandmark;
      if (pending) return; // 放置由 pointerup 处理
      const near = findNearbyMarker(e.clientX, e.clientY);
      if (near) {
        dragTarget.current = near;
        onConstraint?.(near);
        if (controlsRef.current) controlsRef.current.enabled = false;
        canvas.style.cursor = "grabbing";
      }
    };
    const onMove = (e: PointerEvent) => {
      if (useUIStore.getState().brushMode) return;
      if (dragTarget.current) {
        const pca = screenToPCA(e.clientX, e.clientY);
        if (!pca) return; // 视角与标注平面几乎平行：保持当前位置，避免落点跳远
        const constrained =
          dragTarget.current.name === "spine_points"
            ? apply2DSpineConstraint(dragTarget.current.index, pca, useLandmarkStore.getState().landmarks)
            : pca;
        const tmp = pcaToPlane3D(constrained);
        if (tmp) {
          // 拖拽期间先在 PCA 平面上即时预览，松手后再 lift 贴到 mesh（与 2D 交互节奏一致）
          lastDrag.current = {
            name: dragTarget.current.name,
            index: dragTarget.current.index,
            pca: constrained,
          };
          useLandmarkStore.getState().updateLandmark3D(dragTarget.current.name, dragTarget.current.index, {
            x: tmp.x,
            y: tmp.y,
            z: tmp.z,
          });
          // 与 2D 一致：拖动双边点时脊柱中点实时跟随（平面预览）
          if (dragTarget.current.name !== "spine_points") {
            syncSpineLive(dragTarget.current.name);
          }
        }
      } else if (!useUIStore.getState().pendingLandmark) {
        canvas.style.cursor = findNearbyMarker(e.clientX, e.clientY)
          ? "grab"
          : "default";
      }
    };
    const onUp = (e: PointerEvent) => {
      if (useUIStore.getState().brushMode) return;
      const pending = useUIStore.getState().pendingLandmark;
      if (dragTarget.current) {
        const drag = lastDrag.current;
        dragTarget.current = null;
        lastDrag.current = null;
        onConstraint?.(null);
        if (drag) {
          void commitPoint(drag.name, drag.index, drag.pca);
          // 双侧对称点 / axilla / waist 拖动结束后，把脊柱中点重新约束到新连线
          if (drag.name !== "spine_points") void syncSpineAfterMove(drag.name);
        }
        if (controlsRef.current) controlsRef.current.enabled = !pending;
        canvas.style.cursor = pending ? "crosshair" : "default";
        return;
      }
      if (pending) {
        const pca = screenToPCA(e.clientX, e.clientY);
        if (pca) {
          void commitPoint(pending.name, pending.index, pca);
          if (pending.name !== "spine_points") void syncSpineAfterMove(pending.name);
          useUIStore.getState().setPendingLandmark(null);
          canvas.style.cursor = "default";
        }
      }
    };
    canvas.style.cursor = useUIStore.getState().pendingLandmark
      ? "crosshair"
      : "default";
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      if (controlsRef.current) controlsRef.current.enabled = true;
    };
  }, [gl, controlsRef, meshScene, findNearbyMarker, screenToPCA, apply2DSpineConstraint, pcaToPlane3D, commitPoint, syncSpineAfterMove, syncSpineLive, onConstraint]);

  return null;
}

interface Props {
  subjectId: string;
  landmarks: Landmarks;
  onReady?: () => void;
}

export default function MeshScene({ subjectId, landmarks, onReady }: Props) {
  const meshVersion = useUIStore((s) => s.meshVersion);
  const [activeConstraint, setActiveConstraint] = useState<{ name: string; index: number } | null>(null);
  const meshUrl = getMeshUrl(subjectId) + "?v=" + meshVersion;
  const [meshScene, setMeshScene] = useState<THREE.Group | null>(null);
  const [meshLoaded, setMeshLoaded] = useState(false);
  // Which subject the loaded mesh belongs to — prevents cross-subject rendering
  const meshForSubject = useRef<string | null>(null);
  const signalled = useRef(false);
  // Shared ref for orbit controls — allows BrushHandler to disable during brush mode
  const controlsRef = useRef<any>(null);
  const brushMode = useUIStore((s) => s.brushMode);
  const brushSize = useUIStore((s) => s.brushSize);
  const cursorRef = useRef<HTMLDivElement>(null);
  // Syncs immediately on useLayoutEffect — used by stable handleMeshReady callback
  const subjectRef = useRef(subjectId);

  const hasLandmarks = Object.values(landmarks).some(
    (pts) => Array.isArray(pts) && pts.length > 0,
  );

  // ─── Reset ALL state synchronously on subject change (useLayoutEffect runs before useEffects) ───
  useLayoutEffect(() => {
    subjectRef.current = subjectId;
    setMeshScene(null);
    setMeshLoaded(false);
    signalled.current = false;
    meshForSubject.current = null;
  }, [subjectId]);

  // ─── Mesh loaded callback — stable reference, does NOT cause MeshModel to re-fetch ───
  const handleMeshReady = useCallback((scene: THREE.Group | null) => {
    if (scene) {
      setMeshScene(scene);
      meshForSubject.current = subjectRef.current;
    }
    setMeshLoaded(true);
  }, []);

  // ─── All data MUST belong to the CURRENT subject ───
  const dataConsistent =
    meshLoaded && hasLandmarks && meshForSubject.current === subjectId;

  // ─── Single combined readiness check: data present → wait GPU → signal parent ───
  useEffect(() => {
    if (!dataConsistent) return;
    if (signalled.current) return;
    signalled.current = true;

    // Double rAF: first schedules before render, second fires after GPU completes
    const h1 = requestAnimationFrame(() => {
      const h2 = requestAnimationFrame(() => {
        onReady?.();
      });
    });
    // Safety: force dismiss after 12s for very large meshes
    const t = setTimeout(() => {
      onReady?.();
    }, 12000);
    return () => {
      cancelAnimationFrame(h1);
      clearTimeout(t);
    };
  }, [dataConsistent, onReady]);

  const cursorColor = brushMode === "restore" ? "#888888" : "#ff6666";
  return (
    <Box sx={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas
        key={subjectId}
        style={{ width: "100%", height: "100%", display: "block" }}
        camera={{ position: [0, -150, 600], fov: 40, far: 5000 }}
        onCreated={({ gl }) => {
          gl.setClearColor("#0d1117");
        }}
      >
        <ambientLight intensity={1.2} />
        <directionalLight position={[300, -200, 600]} intensity={1.0} />
        <directionalLight position={[-300, -100, -100]} intensity={0.5} />
        <hemisphereLight args={["#606080", "#202040", 0.6]} />
        <CameraFramer
          meshScene={meshScene}
          landmarks={landmarks}
          controlsRef={controlsRef}
        />
        <MeshModel url={meshUrl} onMeshReady={handleMeshReady} />
        <LandmarkMarkers landmarks={landmarks} />
        <SpineAxisLine landmarks={landmarks} />
        <ClothOverlay subjectId={subjectId} />
        <BrushHandler
          subjectId={subjectId}
          controlsRef={controlsRef}
          cursorRef={cursorRef}
        />
        <LandmarkAnnotationHandler meshScene={meshScene} controlsRef={controlsRef} onConstraint={setActiveConstraint} />
        <SpineConstraintGuide active={activeConstraint} />
      </Canvas>
      {/* Screen-space brush cursor overlay — ref-based DOM, no re-render */}
      <Box
        ref={cursorRef}
        sx={{
          position: "fixed",
          pointerEvents: "none",
          zIndex: 9999,
          borderRadius: "50%",
          bgcolor: cursorColor,
          opacity: 0.35,
          display: brushMode ? "block" : "none",
        }}
      />
    </Box>
  );
}
