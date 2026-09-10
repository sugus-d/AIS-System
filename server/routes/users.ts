import bcrypt from "bcryptjs";
import { Router } from "express";
import { db, audit } from "../services/database";
import { requireRoles } from "../middleware/access";

const router = Router();

const ROLES = ["system_admin", "institution_admin", "operator"] as const;
type Role = (typeof ROLES)[number];
const isRole = (value: unknown): value is Role => typeof value === "string" && (ROLES as readonly string[]).includes(value);

const present = (user: any) => ({ id: user.id, username: user.username, name: user.displayName, role: user.role, department: user.department, institutionId: user.institutionId, email: user.email, phone: user.phone, status: user.active ? "active" : "disabled", createdAt: user.createdAt });

// 三级权限模型：系统管理员不归属机构；机构管理员 / 临床操作员归属机构。
// 机构管理员只能看到自己机构内、且非系统管理员的账户。
const visible = (actor: any, user: any) => actor.role === "system_admin" || (user.role !== "system_admin" && Boolean(actor.institutionId) && actor.institutionId === user.institutionId);

const normalizeUsername = (value: unknown) => { const username = String(value ?? "").trim(); return username.length >= 2 && username.length <= 32 && /^[\p{L}\p{N}._-]+$/u.test(username) ? username : null; };
const usernameError = "用户名需为 2-32 位字母、数字或 . _ -，且不能包含空格。";
const passwordError = "密码至少需要 12 位。";
const institutionRequiredError = "请选择所属机构。";
const institutionNameRequiredError = "请填写机构名称。";
const institutionDuplicateError = "机构名称已存在。";
const invalidRoleError = "无效的角色。";
const adminOnlyError = "仅系统管理员可管理管理员账户。";

const loadInstitutionMap = async () => {
  const institutions = await db.institution.findMany({ include: { users: { select: { displayName: true, role: true } } } });
  return new Map(institutions.map((i) => [i.id, { name: i.name, admin: i.users.filter((u) => u.role === "institution_admin").map((u) => u.displayName).join("、") || null }]));
};

const normalizeInstitutionName = (value: unknown) => String(value ?? "").trim();
const nextInstitutionCode = () => `INST-${Date.now().toString(36).toUpperCase()}`;

// 账号只能归属到“已存在”的机构：机构由系统管理员在机构管理里创建，创建账号时必须选择一个机构
const findInstitutionId = async (value: unknown): Promise<string | null> => {
  if (typeof value !== "string" || !value) return null;
  const existing = await db.institution.findUnique({ where: { id: value }, select: { id: true } });
  return existing?.id ?? null;
};

router.get("/", requireRoles("system_admin", "institution_admin"), async (req: any, res) => {
  const instMap = await loadInstitutionMap();
  let users = await db.user.findMany({ orderBy: { createdAt: "desc" } });
  users = users.filter((user) => visible(req.user, user));
  if (typeof req.query.keyword === "string") users = users.filter((user) => user.username.includes(req.query.keyword) || user.displayName.includes(req.query.keyword));
  if (typeof req.query.role === "string") users = users.filter((user) => user.role === req.query.role);
  const page = Math.max(1, Number(req.query.page || 1));
  const pageSize = Math.min(100, Math.max(1, Number(req.query.pageSize || 20)));
  res.json({
    success: true,
    data: {
      list: users.slice((page - 1) * pageSize, page * pageSize).map((user) => {
        const inst = user.role !== "system_admin" && user.institutionId ? instMap.get(user.institutionId) : undefined;
        // 三级模型：系统管理员不属于任何机构；机构管理员与临床操作员都显示所属机构名称
        const superior = user.role === "system_admin" ? null : inst?.name || null;
        return { ...present(user), institutionName: inst?.name || null, institutionAdmin: inst?.admin || null, superior };
      }),
      total: users.length,
      page,
      pageSize,
    },
  });
});

router.get("/institutions", requireRoles("system_admin", "institution_admin"), async (req: any, res) => {
  const institutions = await db.institution.findMany({ include: { users: { select: { displayName: true, role: true } } } });
  const list = req.user.role === "system_admin" ? institutions : institutions.filter((i) => i.id === req.user.institutionId);
  const data = list.map((i) => ({ id: i.id, name: i.name, code: i.code, admin: i.users.filter((u) => u.role === "institution_admin").map((u) => u.displayName).join("、") || null }));
  res.json({ success: true, data });
});

// 机构维护（仅系统管理员）：新增 / 改名 / 启停
router.post("/institutions", requireRoles("system_admin"), async (req: any, res) => {
  const name = normalizeInstitutionName(req.body?.name);
  if (!name) return res.status(400).json({ success: false, message: institutionNameRequiredError });
  if (await db.institution.findUnique({ where: { name } })) return res.status(409).json({ success: false, message: institutionDuplicateError });
  const code = (typeof req.body?.code === "string" && req.body.code.trim()) || nextInstitutionCode();
  try {
    const created = await db.institution.create({ data: { name, code } });
    await audit(req.user.id, "create", "Institution", created.id);
    res.status(201).json({ success: true, data: { id: created.id, name: created.name, code: created.code } });
  } catch {
    res.status(409).json({ success: false, message: institutionDuplicateError });
  }
});

router.put("/institutions/:id", requireRoles("system_admin"), async (req: any, res) => {
  const existing = await db.institution.findUnique({ where: { id: req.params.id } });
  if (!existing) return res.status(404).json({ success: false, message: "Institution not found." });
  const name = normalizeInstitutionName(req.body?.name) || existing.name;
  const clash = await db.institution.findUnique({ where: { name } });
  if (clash && clash.id !== existing.id) return res.status(409).json({ success: false, message: institutionDuplicateError });
  const updated = await db.institution.update({ where: { id: existing.id }, data: { name, active: typeof req.body?.active === "boolean" ? req.body.active : undefined } });
  await audit(req.user.id, "update", "Institution", updated.id);
  res.json({ success: true, data: { id: updated.id, name: updated.name, code: updated.code, active: updated.active } });
});

router.get("/:id", requireRoles("system_admin", "institution_admin"), async (req: any, res) => {
  const user = await db.user.findUnique({ where: { id: req.params.id } });
  if (!user || !visible(req.user, user)) return res.status(404).json({ success: false, message: "User not found." });
  res.json({ success: true, data: present(user) });
});

router.post("/", requireRoles("system_admin", "institution_admin"), async (req: any, res) => {
  const rawUsername = req.body?.username ? normalizeUsername(req.body.username) : null;
  if (req.body?.username && !rawUsername) return res.status(400).json({ success: false, message: usernameError });
  const username = rawUsername || `operator_${Date.now()}`;
  const password = String(req.body?.password || "");
  if (password.length < 12) return res.status(400).json({ success: false, message: passwordError });

  let role: Role;
  let institutionId: string | null;
  if (req.user.role === "system_admin") {
    role = isRole(req.body?.role) ? req.body.role : "operator";
    if (role === "system_admin") {
      institutionId = null; // 系统管理员不归属机构
    } else {
      // 机构管理员 / 临床操作员：必须归属一个已存在的机构
      institutionId = await findInstitutionId(req.body?.institutionId);
      if (!institutionId) return res.status(400).json({ success: false, message: institutionRequiredError });
    }
  } else {
    role = "operator"; // 机构管理员只能创建本机构的临床操作员
    institutionId = req.user.institutionId || null;
    if (!institutionId) return res.status(400).json({ success: false, message: institutionRequiredError });
  }
  try {
    const user = await db.user.create({ data: { username, passwordHash: await bcrypt.hash(password, 12), displayName: String(req.body?.name || username), role, institutionId, department: req.body?.department, email: req.body?.email, phone: req.body?.phone } });
    await audit(req.user.id, "create", "User", user.id);
    res.status(201).json({ success: true, data: present(user) });
  } catch {
    res.status(409).json({ success: false, message: "用户名已存在，请换一个。" });
  }
});

router.put("/:id", requireRoles("system_admin", "institution_admin"), async (req: any, res) => {
  const existing = await db.user.findUnique({ where: { id: req.params.id } });
  if (!existing || !visible(req.user, existing)) return res.status(404).json({ success: false, message: "User not found." });

  const data: any = { displayName: req.body?.name, department: req.body?.department, email: req.body?.email, phone: req.body?.phone, active: req.body?.status ? req.body.status === "active" : undefined };
  if (req.body?.username !== undefined) {
    const username = normalizeUsername(req.body.username);
    if (!username) return res.status(400).json({ success: false, message: usernameError });
    data.username = username;
  }
  if (req.user.role === "system_admin") {
    let targetRole: Role = isRole(existing.role) ? existing.role : "operator";
    if (req.body?.role !== undefined) {
      if (!isRole(req.body.role)) return res.status(400).json({ success: false, message: invalidRoleError });
      targetRole = req.body.role;
    }
    data.role = targetRole;
    if (targetRole === "system_admin") {
      data.institutionId = null; // 系统管理员不归属机构
    } else {
      const nextInstitutionId = (await findInstitutionId(req.body?.institutionId)) || existing.institutionId;
      if (!nextInstitutionId) return res.status(400).json({ success: false, message: institutionRequiredError });
      data.institutionId = nextInstitutionId;
    }
  } else if (existing.role !== "operator") {
    // 机构管理员只能维护本机构内临床操作员的资料，不能改动管理员账户
    return res.status(403).json({ success: false, message: adminOnlyError });
  }
  if (typeof req.body?.password === "string" && req.body.password !== "") {
    if (req.body.password.length < 12) return res.status(400).json({ success: false, message: passwordError });
    data.passwordHash = await bcrypt.hash(req.body.password, 12);
  }
  // 账号首次归入某个机构时，把它名下尚未归属机构的档案一并归入该机构
  const adoptInstitutionId = typeof data.institutionId === "string" && data.institutionId && data.institutionId !== existing.institutionId ? data.institutionId : null;
  try {
    const user = await db.user.update({ where: { id: existing.id }, data });
    if (adoptInstitutionId) await db.case.updateMany({ where: { ownerId: existing.id, institutionId: null }, data: { institutionId: adoptInstitutionId } });
    await audit(req.user.id, "update", "User", user.id);
    res.json({ success: true, data: present(user) });
  } catch {
    res.status(409).json({ success: false, message: "用户名已存在，请换一个。" });
  }
});

router.delete("/:id", requireRoles("system_admin"), async (req: any, res) => {
  if (req.params.id === req.user.id) return res.status(400).json({ success: false, message: "Cannot delete the current user." });
  const user = await db.user.findUnique({ where: { id: req.params.id } });
  if (!user) return res.status(404).json({ success: false, message: "User not found." });
  const ownedCases = await db.case.count({ where: { ownerId: user.id } });
  if (ownedCases > 0) return res.status(400).json({ success: false, message: "该用户名下还有受检者档案，请先转移或删除后再删除用户。" });
  await db.$transaction(async (tx) => {
    await tx.analysisTask.deleteMany({ where: { submittedById: user.id } });
    await tx.user.delete({ where: { id: user.id } });
  });
  await audit(req.user.id, "delete", "User", user.id);
  res.json({ success: true });
});

router.post("/:id/toggle-status", requireRoles("system_admin"), async (req: any, res) => {
  const user = await db.user.findUnique({ where: { id: req.params.id } });
  if (!user) return res.status(404).json({ success: false, message: "User not found." });
  const updated = await db.user.update({ where: { id: user.id }, data: { active: !user.active } });
  res.json({ success: true, data: { status: updated.active ? "active" : "disabled" } });
});

export default router;
