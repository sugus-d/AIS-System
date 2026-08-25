import { Router } from "express";
import { db, audit } from "../services/database";
import { requireRoles } from "../middleware/access";

const router = Router(); const defaults = { name: "AIS", logo: "", timezone: "Asia/Shanghai", version: "1.0.0" };
router.get("/", async (_req, res) => { const setting = await db.appSetting.findUnique({ where: { key: "system" } }); let system = defaults; try { if (setting) system = { ...defaults, ...JSON.parse(setting.value) }; } catch { /* retain defaults */ } res.json({ success: true, data: { system } }); });
router.put("/system", requireRoles("system_admin"), async (req: any, res) => { const system = { ...defaults, name: req.body?.name || defaults.name, logo: req.body?.logo || "", timezone: req.body?.timezone || defaults.timezone }; await db.appSetting.upsert({ where: { key: "system" }, create: { key: "system", value: JSON.stringify(system) }, update: { value: JSON.stringify(system) } }); await audit(req.user.id, "update_settings", "AppSetting", "system"); res.json({ success: true, data: system }); });
export default router;
