import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import api from "@/lib/api";
import { UserRound } from "lucide-react";

const items = [{ path: "/dashboard", label: "工作台" }, { path: "/cases", label: "受检者管理" }, { path: "/statistics", label: "数据统计" }];

const roleLabels: Record<string, string> = {
  system_admin: "系统管理员",
  institution_admin: "机构管理员",
  operator: "临床操作员",
  admin: "管理员",
};

type CurrentUser = { id?: string; username?: string; name?: string; role?: string; department?: string | null; institution?: string | null };

export default function Header({ isAdmin }: { isAdmin: boolean }) {
  void isAdmin; // 保留向后兼容；真实身份以 /auth/me 从数据库读取为准
  const navigate = useNavigate();
  const location = useLocation();

  const [user, setUser] = useState<CurrentUser>(() => ({
    id: localStorage.getItem("user_id") || undefined,
    name: localStorage.getItem("user_name") || undefined,
    role: localStorage.getItem("user_role") || undefined,
    department: localStorage.getItem("user_department") || undefined,
    institution: localStorage.getItem("user_institution") || undefined,
  }));

  useEffect(() => {
    let cancelled = false;
    api.getCurrentUser()
      .then((fresh) => {
        if (cancelled || !fresh) return;
        setUser(fresh);
        if (fresh.role) localStorage.setItem("user_role", fresh.role);
        if (fresh.name) localStorage.setItem("user_name", fresh.name);
        if (fresh.department) localStorage.setItem("user_department", fresh.department);
        if (fresh.institution) localStorage.setItem("user_institution", fresh.institution);
      })
      .catch(() => { /* 保留 localStorage 快照 */ });
    return () => { cancelled = true; };
  }, []);

  const role = user.role || "operator";
  const roleLabel = roleLabels[role] || role;
  const isSystemAdmin = role === "system_admin";
  const canManageUsers = role === "system_admin" || role === "institution_admin";
  const displayName = user.name || user.username || "未登录";

  const logout = () => {
    ["user_role", "user_name", "user_department", "user_institution", "user_id", "user_token", "auth_token"].forEach((k) => localStorage.removeItem(k));
    navigate("/login");
  };

  return (
    <div className="w-full h-full px-4 md:px-6 flex items-center gap-3 md:gap-6">
      <div className="shrink-0 flex items-center gap-2">
        <div className="w-8 h-8 bg-[color:var(--color-primary)] rounded-md flex items-center justify-center text-white font-bold text-sm">AIS</div>
        <div className="text-card-title text-[color:var(--color-text-primary)] font-semibold whitespace-nowrap">AIS 筛查系统</div>
      </div>
      <nav className="min-w-0 flex-1 overflow-x-auto">
        <div className="flex items-center gap-2 w-full min-w-[520px] pr-3">
          {items.map((item) => (
            <div key={item.path} className="flex-1 px-2">
              <button
                onClick={() => navigate(item.path)}
                className={`block w-1/2 min-w-[90px] mx-auto px-2 py-2 rounded-btn text-body font-semibold transition-colors whitespace-nowrap text-center ${location.pathname === item.path ? "bg-[color:var(--color-primary-light)] text-[color:var(--color-primary)]" : "text-[color:var(--color-text-secondary)] hover:bg-[color:var(--color-neutral)]"}`}
              >
                {item.label}
              </button>
            </div>
          ))}
        </div>
      </nav>
      <div className="ml-auto shrink-0 flex items-center gap-3 pl-3 md:pl-4 border-l border-[color:var(--color-border)]">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center gap-3 hover:bg-[color:var(--color-neutral)] rounded-btn px-2 py-1.5 transition-colors">
              <div className="text-right hidden md:block">
                <p className="text-body font-semibold text-[color:var(--color-text-primary)]">{displayName}</p>
                <p className="text-helper text-[color:var(--color-text-tertiary)]">{user.department || "—"}</p>
                <p className="text-helper text-[color:var(--color-text-tertiary)]">{roleLabel}</p>
              </div>
              <div className="w-9 h-9 rounded-full bg-[color:var(--color-neutral)] flex items-center justify-center text-[color:var(--color-text-secondary)]">
                <UserRound className="w-5 h-5" />
              </div>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <div className="px-2 py-1.5 border-b border-[color:var(--color-border)] mb-1">
              <p className="text-body font-semibold text-[color:var(--color-text-primary)]">{displayName}</p>
              <p className="text-helper text-[color:var(--color-text-tertiary)]">{user.department || "—"}</p>
              <p className="text-helper text-[color:var(--color-text-tertiary)]">{roleLabel}</p>
            </div>
            <DropdownMenuItem onClick={() => navigate("/settings")}>个人设置</DropdownMenuItem>
            {canManageUsers && (
              <>
                {isSystemAdmin && <DropdownMenuItem onClick={() => navigate("/admin/settings")}>系统设置</DropdownMenuItem>}
                <DropdownMenuItem onClick={() => navigate("/admin/users")}>用户管理</DropdownMenuItem>
              </>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={logout} className="text-[color:var(--color-error)] focus:text-[color:var(--color-error)]">退出登录</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}
