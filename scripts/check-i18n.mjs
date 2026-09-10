// i18n 一致性检查：客户端用到的 key 是否两套语言包都有、插值占位符是否匹配、
// 后端返回给用户的中文提示是否有 backend.* 翻译、简繁两份语言包是否同步。
// 用法：pnpm i18n:check
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const localeDir = path.join(root, "client", "locales");
const LOCALES = ["zh-CN.json", "zh-TW.json"];

const flatten = (obj, prefix = "", out = {}) => {
  for (const [key, value] of Object.entries(obj)) {
    const next = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) flatten(value, next, out);
    else out[next] = value;
  }
  return out;
};

const raw = Object.fromEntries(LOCALES.map((file) => [file, JSON.parse(fs.readFileSync(path.join(localeDir, file), "utf8"))]));
const flat = Object.fromEntries(LOCALES.map((file) => [file, flatten(raw[file])]));
const sections = new Set(Object.keys(raw[LOCALES[0]]));

const walk = (dir, out = []) => {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (["node_modules", "generated", "locales", "dist", ".release"].includes(entry.name)) continue;
      walk(full, out);
    } else if (/\.(ts|tsx)$/.test(entry.name)) out.push(full);
  }
  return out;
};

const problems = [];
const keyPattern = /^[a-z][A-Za-z0-9]*(?:\.[A-Za-z0-9_]+)+$/;
const isLocaleKey = (key) => keyPattern.test(key) && sections.has(key.split(".")[0]) && !/\.(html|json|md|csv|pdf|png|jpe?g|webp|ply|stl)$/i.test(key);

// 1) 客户端静态 key 覆盖 + 插值占位符
for (const file of [...walk(path.join(root, "client")), ...walk(path.join(root, "desktop"))]) {
  const rel = path.relative(root, file);
  const text = fs.readFileSync(file, "utf8");
  const seen = new Set();
  const check = (key) => {
    if (!isLocaleKey(key) || seen.has(key)) return;
    seen.add(key);
    for (const locale of LOCALES) {
      if (!(key in flat[locale])) problems.push(`[缺少 key] ${key}  （${rel}，${locale} 中没有）`);
    }
  };
  for (const match of text.matchAll(/(?:\bt|\bi18n\.t)\(\s*["'`]([^"'`$]+)["'`]/g)) check(match[1]);
  for (const match of text.matchAll(/["'`]([a-z][A-Za-z0-9]*(?:\.[A-Za-z0-9_]+)+)["'`]/g)) check(match[1]);
  for (const match of text.matchAll(/\bt\(\s*["'`]([^"'`$]+)["'`]\s*,\s*\{([^}]*)\}/g)) {
    const key = match[1];
    if (!isLocaleKey(key)) continue;
    const vars = [...match[2].matchAll(/([A-Za-z_$][\w$]*)\s*:/g)].map((m) => m[1]);
    for (const locale of LOCALES) {
      const value = flat[locale][key];
      if (typeof value !== "string") continue;
      for (const name of vars) {
        if (!value.includes(`{{${name}}}`)) problems.push(`[插值缺失] ${key} 缺少 {{${name}}}（${locale}），实际值：${value}`);
      }
    }
  }
}

// 2) 后端返回给用户的中文提示是否已加入 backend 命名空间
for (const file of walk(path.join(root, "server"))) {
  const rel = path.relative(root, file);
  const text = fs.readFileSync(file, "utf8");
  const consts = new Map();
  for (const match of text.matchAll(/\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"((?:[^"\\]|\\.)*)"/g)) consts.set(match[1], match[2]);
  const messages = new Set();
  for (const match of text.matchAll(/message:\s*"((?:[^"\\]|\\.)*)"/g)) messages.add(match[1]);
  for (const match of text.matchAll(/message:\s*([A-Za-z_$][\w$]*)/g)) {
    const value = consts.get(match[1]);
    if (value) messages.add(value);
  }
  for (const message of messages) {
    if (!/[\u4e00-\u9fff]/.test(message)) continue;
    for (const locale of LOCALES) {
      if (!(`backend.${message}` in flat[locale])) problems.push(`[后端提示未翻译] “${message}” 缺少 backend.${message}（${locale}，${rel}）`);
    }
  }
}

// 3) 简繁语言包同步性
for (const key of Object.keys(flat["zh-CN.json"])) {
  if (!(key in flat["zh-TW.json"])) problems.push(`[简繁不同步] ${key} 只在 zh-CN 中存在`);
}
for (const key of Object.keys(flat["zh-TW.json"])) {
  if (!(key in flat["zh-CN.json"])) problems.push(`[简繁不同步] ${key} 只在 zh-TW 中存在`);
}

if (problems.length) {
  console.error(`i18n 检查未通过，共 ${problems.length} 处问题：`);
  for (const problem of problems) console.error("  -", problem);
  process.exit(1);
}
console.log(`i18n 检查通过：zh-CN ${Object.keys(flat["zh-CN.json"]).length} 个 key，zh-TW ${Object.keys(flat["zh-TW.json"]).length} 个 key，客户端/后端引用全部命中。`);
