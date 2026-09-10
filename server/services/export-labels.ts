// 导出文件（CSV / HTML）的文案：跟随请求头 Accept-Language，默认简体中文。
// 客户端 api.ts 会在每个请求上带当前界面语言，因此导出内容与界面语言保持一致。
export type ExportLang = "zh-CN" | "zh-TW";

export const resolveExportLang = (req: any): ExportLang => {
  // 优先 ?lang=（便于 <a href> / <img> 这类无法带请求头的场景），其次 Accept-Language
  const query = String(req?.query?.lang || "");
  const source = query || String(req?.headers?.["accept-language"] || "");
  return /tw|hant|繁體|繁体/i.test(source) ? "zh-TW" : "zh-CN";
};

const SEVERITY: Record<ExportLang, Record<string, string>> = {
  "zh-CN": { Normal: "正常", Mild: "轻度", Moderate: "中度", Severe: "重度" },
  "zh-TW": { Normal: "正常", Mild: "輕度", Moderate: "中度", Severe: "重度" },
};

export const severityLabel = (severity: string, lang: ExportLang) => SEVERITY[lang][severity] || severity;

const LABELS: Record<ExportLang, Record<string, string>> = {
  "zh-CN": {
    reportTitle: "AIS 报告",
    generatedAt: "生成时间",
    caseNumber: "病例编号",
    name: "姓名",
    cobbAngle: "Cobb 角",
    aisLevel: "AIS 等级",
    algorithmLevel: "算法等级",
    modelVersion: "模型版本",
    scanTime: "扫描时间",
    clinicalData: "临床数据",
    metric: "指标",
    value: "数值",
    caseCount: "病例数",
    reportCount: "分析报告数",
    csvReportId: "报告ID",
    csvCaseNumber: "病例编号",
    csvName: "姓名",
    csvCobbAngle: "Cobb角",
    csvAisLevel: "AIS等级",
    csvModelVersion: "模型版本",
    csvGeneratedAt: "生成时间",
  },
  "zh-TW": {
    reportTitle: "AIS 報告",
    generatedAt: "生成時間",
    caseNumber: "病例編號",
    name: "姓名",
    cobbAngle: "Cobb 角",
    aisLevel: "AIS 等級",
    algorithmLevel: "演算法等級",
    modelVersion: "模型版本",
    scanTime: "掃描時間",
    clinicalData: "臨床數據",
    metric: "指標",
    value: "數值",
    caseCount: "病例數",
    reportCount: "分析報告數",
    csvReportId: "報告ID",
    csvCaseNumber: "病例編號",
    csvName: "姓名",
    csvCobbAngle: "Cobb角",
    csvAisLevel: "AIS等級",
    csvModelVersion: "模型版本",
    csvGeneratedAt: "生成時間",
  },
};

export const exportLabels = (lang: ExportLang) => LABELS[lang];
