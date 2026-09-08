// 一次性脚本：用 OpenCC 将 zh-CN.json 转换为 zh-TW.json（繁体初稿）。
// 生成后需人工校对医学/流程术语（受检者/筛查/报告/脊柱/肩胛峰/腋窝/腰部/审核等）。
// 运行：node scripts/generate-zh-tw.mjs
import { Converter } from "opencc-js";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const converter = Converter({ from: "cn", to: "tw" });

const convert = (obj) => {
  const out = {};
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === "string") out[key] = converter(value);
    else if (value && typeof value === "object") out[key] = convert(value);
    else out[key] = value;
  }
  return out;
};

const input = JSON.parse(readFileSync(path.join(root, "client/locales/zh-CN.json"), "utf8"));
writeFileSync(
  path.join(root, "client/locales/zh-TW.json"),
  JSON.stringify(convert(input), null, 2) + "\n",
  "utf8",
);
console.log("zh-TW.json generated");
