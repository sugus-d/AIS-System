import { readFile } from "node:fs/promises";
import path from "node:path";
const coreRoot = process.env.AIS_CORE_ROOT || path.join(process.cwd(), "AIS_core_algo");
const algorithmUrl = process.env.AIS_ALGORITHM_URL || "http://127.0.0.1:8000";
export const algorithmPaths = { coreRoot, reportRoot: path.join(coreRoot, "prediction", "outputs"), meshRoot: path.join(coreRoot, "data", "mesh") };
export async function predict(filePath: string, subjectId: string, clinical: Record<string, unknown>, landmarks?: Record<string, unknown>) {
  const bytes = await readFile(filePath); const body = new FormData(); body.append("file", new Blob([bytes]), path.basename(filePath)); body.append("subject_id", subjectId); body.append("clinical", JSON.stringify(clinical)); body.append("model", process.env.AIS_MODEL_VERSION || "v1.0.0"); if (landmarks) body.append("landmarks", JSON.stringify(landmarks));
  const serviceToken = process.env.AIS_SERVICE_TOKEN;
  let response: globalThis.Response;
  try {
    response = await fetch(`${algorithmUrl}/api/predict`, {
      method: "POST",
      body,
      headers: serviceToken ? { "x-ais-service-token": serviceToken } : undefined,
      signal: AbortSignal.timeout(Number(process.env.AIS_ALGORITHM_TIMEOUT_MS || 300000)),
    });
  } catch {
    throw new Error("AIS 核心算法服务不可用");
  }
  const payload = await response.json().catch(() => ({})); if (!response.ok) throw new Error(payload.detail || "AIS 核心算法分析失败"); return payload;
}
export function severityZh(severity: string) { return ({ Normal: "正常", Mild: "轻度", Moderate: "中度", Severe: "重度" } as Record<string, string>)[severity] || severity; }
