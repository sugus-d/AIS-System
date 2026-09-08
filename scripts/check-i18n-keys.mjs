// 校验脚本：扫描 client 中所有 t("...") / i18next.t("...") 调用，
// 与 zh-CN.json 字典比对，输出缺失的翻译 key（防止页面显示 key 名）。
// 运行：node scripts/check-i18n-keys.mjs
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const dict = JSON.parse(readFileSync(path.join(root, "client/locales/zh-CN.json"), "utf8"));

// 展平字典 key（a.b.c → "a.b.c"）
const flatten = (obj, prefix = "", out = []) => {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object") flatten(v, key, out);
    else out.push(key);
  }
  return out;
};
const dictKeys = new Set(flatten(dict));

const walk = (dir, files = []) => {
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) {
      if (name === "node_modules" || name === "dist" || name === "locales") continue;
      walk(p, files);
    } else if (/\.(ts|tsx)$/.test(name)) files.push(p);
  }
  return files;
};

const used = new Map(); // key -> files
const keyRe = /(?:t|i18next\.t)\(\s*["']([^"']+)["']/g;
for (const file of walk(path.join(root, "client"))) {
  const text = readFileSync(file, "utf8");
  let m;
  while ((m = keyRe.exec(text)) !== null) {
    const key = m[1];
    if (!key.includes(".")) continue; // 动态 key（变量）跳过
    if (!used.has(key)) used.set(key, []);
    used.get(key).push(file.replace(root + path.sep, ""));
  }
}

const missing = [...used.entries()].filter(([key]) => !dictKeys.has(key));
if (missing.length === 0) {
  console.log(`✓ 全部 ${used.size} 个静态翻译 key 均已存在于字典`);
} else {
  console.log(`✗ 缺失 ${missing.length} 个 key（页面将显示 key 名）：`);
  for (const [key, files] of missing) {
    console.log(`  - ${key}  (${[...new Set(files)].join(", ")})`);
  }
}
