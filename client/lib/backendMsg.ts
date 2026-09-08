import i18next from "@/i18n";

// 后端返回的中文/英文消息 → 当前语言提示。
// 字典：locales/zh-CN.json 与 zh-TW.json 的 backend 命名空间，key = 后端消息原文。
// 命中则返回当前语言翻译；未命中（新消息/本地消息）原样返回。
export function translateBackendMessage(message: unknown): string {
  const text = typeof message === "string" ? message : "";
  if (!text) return text;
  return i18next.t(`backend.${text}`, { defaultValue: text });
}
