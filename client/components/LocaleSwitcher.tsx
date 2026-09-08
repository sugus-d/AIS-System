import { useTranslation } from "react-i18next";
import { setLocale } from "@/i18n";

// 语言切换器：显示各语言自身的名称（简体 / 繁體），三处复用（登录页、Header、设置页）
export default function LocaleSwitcher() {
  const { i18n } = useTranslation();
  const current: "zh-CN" | "zh-TW" = i18n.language?.toLowerCase().startsWith("zh-tw")
    ? "zh-TW"
    : "zh-CN";
  const base =
    "rounded-md px-2 py-1 text-xs font-semibold transition-colors cursor-pointer";
  return (
    <div className="inline-flex items-center gap-0.5 rounded-lg border border-slate-300 bg-white p-0.5 shadow-sm">
      <button
        type="button"
        className={`${base} ${current === "zh-CN" ? "bg-[color:var(--color-primary)] text-white" : "text-slate-500 hover:text-slate-800"}`}
        onClick={() => setLocale("zh-CN")}
      >
        简体
      </button>
      <button
        type="button"
        className={`${base} ${current === "zh-TW" ? "bg-[color:var(--color-primary)] text-white" : "text-slate-500 hover:text-slate-800"}`}
        onClick={() => setLocale("zh-TW")}
      >
        繁體
      </button>
    </div>
  );
}
