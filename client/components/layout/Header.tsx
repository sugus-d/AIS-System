import { useEffect, useState } from "react";
import { UserRound } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuList,
} from "@/components/ui/navigation-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import api from "@/lib/api";
import LocaleSwitcher from "@/components/LocaleSwitcher";
import logoUrl from "@/assets/logo.png";

const items = [{ path: "/dashboard", label: "header.navDashboard" }, { path: "/cases", label: "header.navCases" }, { path: "/statistics", label: "header.navStats" }];

const roleLabels: Record<string, string> = {
  system_admin: "enums.roleSystemAdmin",
  institution_admin: "enums.roleInstitutionAdmin",
  operator: "enums.roleOperator",
  admin: "enums.roleAdmin",
};

type CurrentUser = { id?: string; username?: string; name?: string; role?: string; department?: string | null; institution?: string | null };

export default function Header({ isAdmin }: { isAdmin: boolean }) {
  void isAdmin; // 保留向后兼容；真实身份以 /auth/me 从数据库读取为准
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation();

  const [user, setUser] = useState<CurrentUser>(() => ({
    id: sessionStorage.getItem("user_id") || undefined,
    name: sessionStorage.getItem("user_name") || undefined,
    role: sessionStorage.getItem("user_role") || undefined,
    department: sessionStorage.getItem("user_department") || undefined,
    institution: sessionStorage.getItem("user_institution") || undefined,
  }));

  useEffect(() => {
    let cancelled = false;
    api.getCurrentUser()
      .then((fresh) => {
        if (cancelled || !fresh) return;
        setUser(fresh);
        if (fresh.role) sessionStorage.setItem("user_role", fresh.role);
        if (fresh.name) sessionStorage.setItem("user_name", fresh.name);
        if (fresh.department) sessionStorage.setItem("user_department", fresh.department);
        if (fresh.institution) sessionStorage.setItem("user_institution", fresh.institution);
      })
      .catch(() => { /* 保留 localStorage 快照 */ });
    return () => { cancelled = true; };
  }, []);

  const role = user.role || "operator";
  const roleLabel = roleLabels[role] ? t(roleLabels[role]) : role;
  const isSystemAdmin = role === "system_admin";
  const canManageUsers = role === "system_admin" || role === "institution_admin";
  const displayName = user.name || user.username || t("header.notLoggedIn");

  const logout = () => {
    api.clearToken();
    ["user_role", "user_name", "user_department", "user_institution", "user_id", "user_token", "auth_token"].forEach((k) => sessionStorage.removeItem(k));
    navigate("/login");
  };

  return (
    <div className="flex h-full w-full items-center gap-3 px-4 md:px-6">
      {/* 品牌区：蓝色品牌条 + logo + 应用名（无底框） */}
      <div className="flex shrink-0 items-center gap-2.5">
        <span aria-hidden className="h-8 w-1 rounded-full bg-primary" />
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden">
            <img src={logoUrl} alt="AIS" className="h-full w-full object-contain" />
          </div>
          <span className="whitespace-nowrap text-[15px] font-semibold tracking-tight text-card-foreground">{t("header.appName")}</span>
        </div>
      </div>

      {/* 主导航：shadcn NavigationMenu */}
      <nav aria-label="Primary navigation" className="min-w-0 flex-1">
        <div className="overflow-x-auto">
          <NavigationMenu>
            <NavigationMenuList className="w-full min-w-[520px] items-center justify-center gap-1 px-2 py-1">
              {items.map((item) => {
                const active = location.pathname === item.path;
                return (
                  <NavigationMenuItem key={item.path} className="flex-1 sm:flex-none">
                    <button
                      type="button"
                      aria-current={active ? "page" : undefined}
                      onClick={() => navigate(item.path)}
                      className={cn(
                        "mx-auto block h-9 min-w-[92px] rounded-lg px-3 text-body font-semibold whitespace-nowrap transition-colors",
                        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                        active
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                      )}
                    >
                      {t(item.label)}
                    </button>
                  </NavigationMenuItem>
                );
              })}
            </NavigationMenuList>
          </NavigationMenu>
        </div>
      </nav>

      {/* 工具区：语言切换 | 用户 */}
      <div className="ml-auto flex shrink-0 items-center gap-2.5 border-l border-border pl-3 md:gap-3 md:pl-4">
        <LocaleSwitcher />
        <Separator orientation="vertical" className="h-6" />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className={cn(
                "flex items-center gap-2.5 rounded-full p-0.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background hover:bg-accent",
              )}
            >
              <Avatar className="h-9 w-9 border border-border bg-white shadow-sm">
                <AvatarFallback className="bg-transparent text-primary"><UserRound size={18} /></AvatarFallback>
              </Avatar>
              <span className="hidden pr-1 text-right lg:block">
                <span className="block text-body font-semibold text-card-foreground">{displayName}</span>
                <span className="block text-xs leading-4 text-muted-foreground">
                  {[user.department, roleLabel].filter(Boolean).join(" · ") || "—"}
                </span>
              </span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-60">
            <DropdownMenuLabel className="flex flex-col gap-1">
              <span className="text-body font-semibold text-card-foreground">{displayName}</span>
              <span className="text-xs font-normal text-muted-foreground">
                {[user.department, roleLabel].filter(Boolean).join(" · ") || "—"}
              </span>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => navigate("/settings")}>{t("header.profile")}</DropdownMenuItem>
            {canManageUsers && (
              <>
                {isSystemAdmin && <DropdownMenuItem onClick={() => navigate("/admin/settings")}>{t("header.systemSettings")}</DropdownMenuItem>}
                <DropdownMenuItem onClick={() => navigate("/admin/users")}>{t("header.userManagement")}</DropdownMenuItem>
              </>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={logout} className="text-destructive focus:text-destructive">{t("header.logout")}</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}
