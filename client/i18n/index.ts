import i18next from "i18next";
import { initReactI18next } from "react-i18next";
import zhCN from "../locales/zh-CN.json";
import zhTW from "../locales/zh-TW.json";

export type Locale = "zh-CN" | "zh-TW";
export const LOCALE_KEY = "ais_locale";

// 语言检测：优先用户偏好（localStorage，跨重启保留）→ 跟随系统语言 → 默认简体
export function detectLocale(): Locale {
  const saved = localStorage.getItem(LOCALE_KEY);
  if (saved === "zh-CN" || saved === "zh-TW") return saved;
  const lang = (navigator.language || "").toLowerCase();
  if (lang.startsWith("zh")) {
    if (/tw|hk|mo|hant/.test(lang)) return "zh-TW";
    return "zh-CN";
  }
  return "zh-CN";
}

export function setLocale(locale: Locale) {
  localStorage.setItem(LOCALE_KEY, locale);
  document.documentElement.lang = locale;
  void i18next.changeLanguage(locale);
}

void i18next.use(initReactI18next).init({
  resources: {
    "zh-CN": { translation: zhCN },
    "zh-TW": { translation: zhTW },
  },
  lng: detectLocale(),
  fallbackLng: "zh-CN",
  interpolation: { escapeValue: false },
});

export default i18next;
