import bcrypt from "bcryptjs";
import { Router } from "express";
import { db, audit } from "../services/database";

const router = Router(); const present = (user: any) => ({ id: user.id, username: user.username, name: user.displayName, role: user.role, department: user.department, email: user.email, phone: user.phone, createdAt: user.createdAt });
router.get("/", async (req: any, res) => { const user = await db.user.findUnique({ where: { id: req.user.id } }); if (!user) return res.status(401).json({ success: false }); res.json({ success: true, data: present(user) }); });
router.put("/", async (req: any, res) => { const user = await db.user.update({ where: { id: req.user.id }, data: { displayName: req.body?.name, department: req.body?.department } }); await audit(req.user.id, "update_profile", "User", user.id); res.json({ success: true, data: present(user) }); });
router.put("/password", async (req: any, res) => { const user = await db.user.findUnique({ where: { id: req.user.id } }); if (!user || !(await bcrypt.compare(String(req.body?.oldPassword || ""), user.passwordHash))) return res.status(400).json({ success: false, message: "Current password is invalid." }); if (req.body?.newPassword !== req.body?.confirmPassword || String(req.body?.newPassword || "").length < 12) return res.status(400).json({ success: false, message: "New password must match and contain at least 12 characters." }); await db.user.update({ where: { id: user.id }, data: { passwordHash: await bcrypt.hash(req.body.newPassword, 12) } }); await audit(user.id, "change_password", "User", user.id); res.json({ success: true }); });
router.post("/bind-email", async (req: any, res) => { const user = await db.user.update({ where: { id: req.user.id }, data: { email: req.body?.email } }); res.json({ success: true, data: present(user) }); });
router.post("/bind-phone", async (req: any, res) => { const user = await db.user.update({ where: { id: req.user.id }, data: { phone: req.body?.phone } }); res.json({ success: true, data: present(user) }); });
export default router;
