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
    req.user = { ...user, name: user.displayName, status: user.active ? 'active' : 'disabled' };
    next();
  } catch { return res.status(401).json({ success: false, message: '无效的登录凭证' }); }
}

export function requireRoles(...roles: Role[]) {
  return (req: Request & { user?: any }, res: Response, next: NextFunction) => {
    if (!req.user || !roles.includes(req.user.role)) return res.status(403).json({ success: false, message: '无权限执行此操作' });
    next();
  };
}

export function canAccessCase(user: any, item: any) {
  if (user.role === 'system_admin') return true;
  if (!user.institutionId || user.institutionId !== item.institutionId) return false;
  return user.role === 'institution_admin' || item.ownerId === user.id;
}
