// src/constants.ts
// 标注约束与显示的共享常量 —— 2D（Curvature2D / useCanvasRenderer）与 3D（MeshScene）标注共用，避免规则漂移。

// 脊柱点 P0–P3 分别锁定在对应双边 landmark 的 L–R 连线上（PCA 2D 的连线约束）
export const SPINE_CONSTRAINT_PAIRS = ['neck_root', 'scapular_peaks', 'axilla', 'waist'];

// mid_back（Pm，spine_points index 5）：无两侧对应点，约束在 A–B 连线上；
// A = mid(axilla_L, waist_L)，B = mid(axilla_R, waist_R)（"腋下连线与腰部连线的中线"）
export const MIDBACK_PAIR = ['axilla', 'waist'];

// 暂不需要隐藏的脊柱点（保留结构，与 2D 原有逻辑一致）
export const SPINE_HIDDEN: Record<number, boolean> = {};

// 默认投影视角方向 = PCA 第三主轴（PC3）在 pca_Vt 矩阵中的行号 —— 2D 视图的正交投影方向，
// 也是 2D→3D lift 的 ray-cast 方向。3D 标注的约束面 = 两点连线 × PC3 张成的平面。
export const DEFAULT_VIEW_DIR_INDEX = 2;