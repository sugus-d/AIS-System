import { Router, Request, Response } from "express";
import bcrypt from "bcryptjs";
import { audit, createToken, db, parseToken } from "../services/database";

const router = Router();
const publicUser = (user: any) => ({ id: user.id, username: user.username, name: user.displayName, role: user.role, department: user.department, institution: user.institution?.name ?? null, email: user.email, phone: user.phone });
router.post("/login", async (req: Request, res: Response) => {
  const { username, password } = req.body || {}; const user = await db.user.findUnique({ where: { username }, include: { institution: true } });
  if (!user || !user.active || !(await bcrypt.compare(password || "", user.passwordHash))) return res.status(401).json({ success: false, message: "用户名或密码错误" });
  await audit(user.id, "login", "User", user.id); return res.json({ success: true, data: { token: createToken(user.id), user: publicUser(user) } });
});
router.post("/logout", (_req, res) => res.json({ success: true }));
router.get("/me", async (req: Request, res: Response) => { const userId = parseToken(req.headers.authorization?.replace("Bearer ", "")); const user = userId ? await db.user.findUnique({ where: { id: userId }, include: { institution: true } }) : null; if (!user || !user.active) return res.status(401).json({ success: false, message: "登录已失效" }); return res.json({ success: true, data: publicUser(user) }); });
router.post("/password", async (req: Request, res: Response) => { const userId = parseToken(req.headers.authorization?.replace("Bearer ", "")); const user = userId ? await db.user.findUnique({ where: { id: userId } }) : null; if (!user || !(await bcrypt.compare(req.body?.oldPassword || "", user.passwordHash))) return res.status(400).json({ success: false, message: "原密码错误" }); if (!req.body?.newPassword || req.body.newPassword.length < 12) return res.status(400).json({ success: false, message: "新密码至少需要 12 位" }); await db.user.update({ where: { id: user.id }, data: { passwordHash: await bcrypt.hash(req.body.newPassword, 12) } }); await audit(user.id, "change_password", "User", user.id); return res.json({ success: true }); });
export default router;
