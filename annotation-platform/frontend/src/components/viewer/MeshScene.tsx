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
import { OrbitControls } from "@react-three/drei";
import type { Landmarks } from "../../types";
import * as THREE from "three";
import { getMeshUrl } from "../../api/subjects";
import { useUIStore } from "../../stores/uiStore";

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

const COLORS: Record<string, number> = {
  neck_root: 0x00ffff,
  shoulder_transition: 0xff4444,
  scapular_peaks: 0x44ff44,
  axilla: 0xff44ff,
  waist: 0xffff44,
  waist_lower: 0xff8c00,
  spine_points: 0xffffff,
};

function LandmarkMarkers({ landmarks }: { landmarks: Landmarks }) {
  const group = useMemo(() => {
    const g = new THREE.Group();
    for (const [name, pts] of Object.entries(landmarks)) {
      if (!Array.isArray(pts)) continue;
      const color = COLORS[name] || 0xffffff;
      for (let i = 0; i < pts.length; i++) {
        const pt = getPt(pts[i]);
        if (!pt) continue;
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(4, 16, 16),
          new THREE.MeshBasicMaterial({ color }),
        );
        sphere.position.set(pt.x, pt.y, pt.z);
        g.add(sphere);
      }
    }
    return g;
  }, [landmarks]);
  return <primitive object={group} />;
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

interface Props {
  subjectId: string;
  landmarks: Landmarks;
  onReady?: () => void;
}

export default function MeshScene({ subjectId, landmarks, onReady }: Props) {
  const meshVersion = useUIStore((s) => s.meshVersion);
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
        <ClothOverlay subjectId={subjectId} />
        <BrushHandler
          subjectId={subjectId}
          controlsRef={controlsRef}
          cursorRef={cursorRef}
        />
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
