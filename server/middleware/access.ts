import { NextFunction, Request, Response } from 'express';
import { db, parseToken } from '../services/database';

export type Role = 'system_admin' | 'institution_admin' | 'operator';

export async function authenticate(req: Request & { user?: any }, res: Response, next: NextFunction) {
  // Query-string token fallback: plain <img> requests cannot carry the
  // Authorization header, so report image URLs append ?token= instead.
  const token = req.headers.authorization?.replace('Bearer ', '') || (typeof req.query.token === 'string' ? req.query.token : '');
  if (!token) return res.status(401).json({ success: false, message: '请先登录' });
  try {
    const userId = parseToken(token); const user = userId ? await db.user.findUnique({ where: { id: userId } }) : null;
    if (!user || !user.active) return res.status(401).json({ success: false, message: '登录已失效' });
    const actor: any = { ...user, name: user.displayName, status: user.active ? 'active' : 'disabled' };
    // 数据可见性需要识别“系统管理员录入的档案”，这里按请求带上系统管理员 id 集合（无权限者不需要）
    if (actor.role !== 'system_admin') {
      actor.systemAdminIds = new Set((await db.user.findMany({ where: { role: 'system_admin' }, select: { id: true } })).map((row: any) => row.id));
    }
    req.user = actor;
    next();
  } catch { return res.status(401).json({ success: false, message: '无效的登录凭证' }); }
}

export function requireRoles(...roles: Role[]) {
  return (req: Request & { user?: any }, res: Response, next: NextFunction) => {
    if (!req.user || !roles.includes(req.user.role)) return res.status(403).json({ success: false, message: '无权限执行此操作' });
    next();
  };
}

// 档案 / 上传 / 报告 的可见性（上级可见下级，下级看不到上级）：
// - 系统管理员：可查看所有下级机构与操作员录入的数据
// - 机构管理员：可见本机构的数据（本人 + 本机构操作员录入）；系统管理员录入的数据不可见
// - 临床操作员：只可见本人录入的数据
export function canAccessCase(user: any, item: any) {
  if (user.role === 'system_admin') return true;
  // 自己录入的档案始终可见（包括尚未归属机构的档案）
  if (item?.ownerId && item.ownerId === user.id) return true;
  if (user.role !== 'institution_admin') return false;
  if (!user.institutionId || user.institutionId !== item?.institutionId) return false;
  const systemAdminIds: Set<string> | undefined = user.systemAdminIds;
  if (systemAdminIds?.has(item.ownerId)) return false;
  return true;
}
