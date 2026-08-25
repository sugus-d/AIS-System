// Subject（列表轻量版）
export interface SubjectInfo {
  id: string;
  labeling_status: 'unlabeled' | 'prelabeled' | 'labeled';
  has_cache: boolean;
}

// Subject 详情（点击后延迟加载）
export interface SubjectDetail {
  id: string;
  has_gt: boolean;
  has_cache: boolean;
  labeling_status: string;
  mesh_file?: string;
  age?: string;
  sex?: string;
  bmi?: string;
  arms?: string;
  body_asymmetry?: boolean;
  notes?: string;
}

// Landmarks
export type Point2D = { x: number; y: number };
export type Point3D = { x: number; y: number; z: number };
// API returns landmarks as number[][] (arrays of [x, y, z])
export type Landmarks = Record<string, number[][]>;

export interface Mapping {
  x_data_range: [number, number];
  y_data_range: [number, number];
  pca_mean?: [number, number, number];
  pca_Vt?: [[number, number, number], [number, number, number], [number, number, number]];
}

export interface Contours {
  left: number[][];
  right: number[][];
}

// Metrics
export interface MetricsResults {
  subject_id: string;
  is_relaxed: boolean;
  metrics: Record<string, any>;
}

export interface ValidationResult {
  is_relaxed: boolean;
  checks: Record<string, { value: number; threshold: number; pass: boolean }>;
}

// Coordinate validation issues
export interface ValidationIssue {
  type: 'x_order' | 'spine_x_range' | 'y_order';
  landmark: string;
  severity: 'error' | 'warning';
  message: string;
  index?: number; // present for spine_x_range issues
}

// Clinical data
export interface ClinicalData {
  [subjectId: string]: {
    age?: string;
    sex?: string;
    height_cm?: number;
    weight_kg?: number;
    arm_span_cm?: number;
    seating_height_cm?: number;
    xray_date?: string;
    recruit_date?: string;
    has_brace?: string;
    max_cobb?: number;
    atr?: number;
    remarks?: string;
    curve1_apex?: string;
    curve1_cobb?: number;
    curve1_direction?: string;
    curve1_level?: string;
    curve2_apex?: string;
    curve2_cobb?: number;
    curve2_direction?: string;
    curve2_level?: string;
    curve3_apex?: string;
    curve3_cobb?: number;
    curve3_direction?: string;
    curve3_level?: string;
    curve4_apex?: string;
    curve4_cobb?: number;
    curve4_direction?: string;
    curve4_level?: string;
  };
}

// Brush
export interface BrushMarkResult {
  marked_count: number;
  marked_points: number[][];
  total: number;
}

export interface BrushCommitResult {
  glb?: number[];
  vert_count?: number;
  error?: string;
}

// Export
export interface ExportTaskStatus {
  task_id?: string;
  status?: string;
  done?: number;
  total?: number;
  error?: string;
}

// Toast
export type ToastType = 'info' | 'success' | 'warn' | 'error';
export interface Toast {
  id: string;
  message: string;
  type: ToastType;
}
