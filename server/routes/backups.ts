import { Router } from "express";
import { audit } from "../services/database";
import { createBackup, listBackups, scheduleRestore } from "../services/backup";
import { requireRoles } from "../middleware/access";
const router = Router();
router.get("/", requireRoles("system_admin"), async (_req, res) => res.json({ success: true, data: { list: await listBackups() } }));
router.post("/", requireRoles("system_admin"), async (req: any, res) => { try { const backup = await createBackup(); await audit(req.user.id, "create_backup", "Backup", backup.name); res.status(201).json({ success: true, data: backup }); } catch (error) { res.status(500).json({ success: false, message: error instanceof Error ? error.message : "Backup failed." }); } });
router.post("/:name/restore", requireRoles("system_admin"), async (req: any, res) => { try { const data = await scheduleRestore(req.params.name); await audit(req.user.id, "schedule_restore", "Backup", req.params.name); res.json({ success: true, data }); } catch (error) { res.status(422).json({ success: false, message: error instanceof Error ? error.message : "Restore failed." }); } });
export default router;
