import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { useTranslation } from "react-i18next";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import api from "@/lib/api";

// 受检者页面「扫描结果」3D 查看器：
// 仿标注平台显示样式（暗色背景 + 灯光），仅查看/旋转/缩放/平移，无任何标注功能。
// 加载中/失败都会给出明确提示，避免出现“一块空白”而无从判断。
export default function PlyViewer({
  fileId,
  onLoadError,
}: {
  fileId: string;
  onLoadError?: (message: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const { t } = useTranslation();
  const [state, setState] = useState<{ status: "loading" | "ready" | "error"; message?: string }>({ status: "loading" });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    setState({ status: "loading" });

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      const message = t("caseDetail.view3dUnsupported");
      setState({ status: "error", message });
      onLoadError?.(message);
      return;
    }
    renderer.setClearColor(0x0d1117); // 与标注平台一致的暗色背景
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 1, 20000);
    camera.position.set(0, -150, 600);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // 灯光（与标注平台一致）
    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    const dir1 = new THREE.DirectionalLight(0xffffff, 1.0);
    dir1.position.set(300, -200, 600);
    scene.add(dir1);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dir2.position.set(-300, -100, -100);
    scene.add(dir2);
    scene.add(new THREE.HemisphereLight(0x606080, 0x202040, 0.6));

    const fitObject = (object: THREE.Object3D) => {
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const distance = maxDim / 2 / Math.tan((camera.fov * Math.PI) / 360);
      camera.position.copy(center).add(new THREE.Vector3(0, -maxDim * 0.25, distance * 1.4));
      controls.target.copy(center);
      controls.update();
    };

    const resize = () => {
      const w = container.clientWidth || 320;
      const h = container.clientHeight || 320;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    let disposed = false;
    let raf = 0;
    const loop = () => {
      if (disposed) return;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    let cancelled = false;
    const fail = (message: string) => {
      if (disposed) return;
      setState({ status: "error", message });
      onLoadError?.(message);
    };

    api
      .downloadMesh(fileId)
      .then((buffer) => {
        if (cancelled || disposed) return;
        if (!buffer || buffer.byteLength === 0) {
          fail(t("caseDetail.view3dEmpty"));
          return;
        }
        let geometry: THREE.BufferGeometry;
        try {
          geometry = new PLYLoader().parse(buffer);
        } catch {
          fail(t("caseDetail.view3dInvalid"));
          return;
        }
        const position = geometry.attributes.position as THREE.BufferAttribute | undefined;
        if (!position || position.count === 0) {
          fail(t("caseDetail.view3dEmpty"));
          return;
        }
        const hasFaces = Boolean(geometry.index && geometry.index.count > 0);
        // 没有三角面的 PLY（点云）用点渲染，避免“解析成功但一片空白”
        const object: THREE.Object3D = hasFaces
          ? new THREE.Mesh(
              geometry,
              new THREE.MeshStandardMaterial({
                color: 0x9fb6d4,
                roughness: 0.7,
                metalness: 0.05,
                flatShading: true,
                side: THREE.DoubleSide,
              }),
            )
          : new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0x9fb6d4, size: 2, sizeAttenuation: true }));
        scene.add(object);
        fitObject(object);
        setState({ status: "ready" });
      })
      .catch((err) => {
        fail(err instanceof Error ? err.message : t("common.load3dFailed"));
      });

    return () => {
      cancelled = true;
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      // 只 dispose 不会释放 WebGL 上下文：反复展开 3D 会耗尽上下文导致后续空白
      try {
        renderer.forceContextLoss();
      } catch { /* 某些环境不支持，忽略 */ }
      if (renderer.domElement.parentElement === container) container.removeChild(renderer.domElement);
    };
  }, [fileId, onLoadError]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      {state.status !== "ready" && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-4 text-center">
          <span className={`text-sm ${state.status === "error" ? "text-[color:var(--color-error)]" : "text-white/75"}`}>
            {state.status === "loading" ? t("caseDetail.view3dLoading") : state.message || t("caseDetail.view3dFailed")}
          </span>
        </div>
      )}
    </div>
  );
}
