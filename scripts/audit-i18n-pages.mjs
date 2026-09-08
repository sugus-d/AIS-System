// 逐文件审计：确认每个页面/组件是否做了多语言。
// 判定：① 是否使用 useTranslation/i18next；② 是否有残留"运行时中文"（排除注释、数据值比较、console、lang 属性）。
// 运行：node scripts/audit-i18n-pages.mjs
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

const walk = (dir, files = []) => {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) {
      if (name === "node_modules" || name === "dist" || name === "locales" || name === "ui") continue;
      walk(p, files);
    } else if (/\.(ts|tsx)$/.test(name)) files.push(p);
  }
  return files;
};

// 数据值比较（中文作为后端数据匹配，不可翻译）
const dataCompare = /(?:===?|!==?)\s*['"][\u4e00-\u9fff]|['"][\u4e00-\u9fff]['"]\s*(?:===?|!==?)|label:\s*['"][\u4e00-\u9fff]/;

const audit = (file) => {
  const rel = file.replace(root + path.sep, "").replace(/\\/g, "/");
  const text = readFileSync(file, "utf8");
  const hasI18n = /\buseTranslation\b|\bi18next\.t\b|\bfrom ["']@\/i18n["']/.test(text);
  const tCount = (text.match(/\bt\(["']/g) || []).length;
  const lines = text.split("\n");
  const residual = [];
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    const trimmed = line.trim();
    // 跳过纯注释行
    if (trimmed.startsWith("//") || trimmed.startsWith("/*") || trimmed.startsWith("*") || trimmed.startsWith("*/")) continue;
    if (!/[\u4e00-\u9fff]/.test(line)) continue;
    // 去掉行尾注释后判断
    const code = line.replace(/\/\/.*$/, "").replace(/\/\*[\s\S]*?\*\//g, "");
    if (!/[\u4e00-\u9fff]/.test(code)) continue;
    if (/console\./.test(code)) continue;          // 控制台日志
    if (/lang=["']zh-CN["']/.test(code)) continue; // PDF lang 属性（已动态化，此处兼容旧写法）
    if (dataCompare.test(code)) continue;          // 后端数据值比较
    if (code.includes("t(") && !dataCompare.test(code)) {
      // 该行已含 t() 翻译调用，仅当其中还有游离中文才报
      const stripped = code.replace(/t\(["'][^"']*["'](?:,[^)]*)?\)/g, "");
      if (!/[\u4e00-\u9fff]/.test(stripped)) continue;
    }
    residual.push(`${i + 1}: ${trimmed.slice(0, 120)}`);
  }
  return { rel, hasI18n, tCount, residual };
};

const files = walk(path.join(root, "client/pages")).concat(walk(path.join(root, "client/components")).filter((f) => !f.includes("\\ui\\") && !f.includes("/ui/")));
const rows = files.map(audit).filter((r) => /\.(tsx)$/.test(r.rel));

console.log("页面/组件多语言审计结果\n" + "=".repeat(70));
for (const r of rows.sort((a, b) => a.rel.localeCompare(b.rel))) {
  const status = !r.hasI18n && r.residual.length > 0 ? "❌ 未做" : r.residual.length > 0 ? `⚠️ 残留 ${r.residual.length}` : r.hasI18n ? "✅ 完成" : "⚪ 无文案";
  console.log(`${status.padEnd(12)} ${r.rel}  (t×${r.tCount})`);
  for (const res of r.residual.slice(0, 6)) console.log(`        ${res}`);
  if (r.residual.length > 6) console.log(`        … 另有 ${r.residual.length - 6} 处`);
}
