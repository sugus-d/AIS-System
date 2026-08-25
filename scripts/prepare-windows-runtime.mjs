import { existsSync } from "node:fs";
import path from "node:path";

const root = process.cwd();
const python = path.join(root, "runtime", "python", "python.exe");
const model = path.join(root, "AIS_core_algo", "prediction", "models", "v1.0.0.joblib");
if (!existsSync(python)) throw new Error(`缺少内置 Python 运行时：${python}`);
if (!existsSync(model)) throw new Error(`缺少 AIS 生产模型：${model}`);
console.log("Windows 便携运行时校验通过。");
