import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { BarChart3, ClipboardPlus, FileUp, Search, Users, ShieldCheck } from "lucide-react";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import api from "@/lib/api";

type Stats = { total: number; pendingUpload: number; pendingAnalysis: number; completed: number; withReports: number };
const adminRoles = ["admin", "system_admin", "institution_admin"];

const iconTones: Record<string, string> = {
  blue: "bg-blue-50 text-blue-700",
  amber: "bg-amber-50 text-amber-700",
  green: "bg-emerald-50 text-emerald-700",
  violet: "bg-violet-50 text-violet-700",
};

export default function Dashboard() {
  const navigate = useNavigate(); const { t } = useTranslation(); const [stats, setStats] = useState<Stats | null>(null); const [loading, setLoading] = useState(true);
  const role = sessionStorage.getItem("user_role") || "operator"; const isAdmin = adminRoles.includes(role);

  useEffect(() => { api.getCaseStats().then(setStats).catch(() => setStats(null)).finally(() => setLoading(false)); }, []);

  const cards = useMemo(() => [
    { label: t("dashboard.totalCases"), value: stats?.total, icon: Users, tone: "blue" },
    { label: t("dashboard.pendingAnalysis"), value: stats?.pendingAnalysis, icon: Search, tone: "amber" },
    { label: t("dashboard.completed"), value: stats?.completed, icon: BarChart3, tone: "green" },
    { label: t("dashboard.pendingUpload"), value: stats?.pendingUpload, icon: FileUp, tone: "violet" },
  ], [stats, t]);

  const actions = useMemo(() => {
    const list = [
      { title: t("dashboard.newCase"), description: t("dashboard.newCaseDesc"), icon: ClipboardPlus, onClick: () => navigate("/case-record") },
      { title: t("dashboard.caseManage"), description: t("dashboard.caseManageDesc"), icon: Users, onClick: () => navigate("/cases") },
      { title: t("dashboard.stats"), description: t("dashboard.statsDesc"), icon: BarChart3, onClick: () => navigate("/statistics") },
    ];
    if (isAdmin) list.push({ title: t("dashboard.userManage"), description: t("dashboard.userManageDesc"), icon: ShieldCheck, onClick: () => navigate("/admin/users") });
    return list;
  }, [isAdmin, navigate, t]);

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header"><Header isAdmin={isAdmin} /></div>
      <div className="layout-content">
        <div className="content-wrapper space-y-8">
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-primary">{t("dashboard.eyebrow")}</p>
            <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("dashboard.title")}</h1>
          </div>

          <section>
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-foreground">{t("dashboard.overview")}</h2>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {cards.map(({ label, value, icon: Icon, tone }) => (
                <Metric key={label} label={label} value={loading ? "—" : String(value ?? 0)} icon={<Icon size={20} />} tone={tone} />
              ))}
            </div>
          </section>

          <section>
            <h2 className="mb-4 text-lg font-semibold text-foreground">{t("dashboard.quickActions")}</h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {actions.map(({ title, description, icon: Icon, onClick }) => (
                <Card
                  key={title}
                  onClick={onClick}
                  className="cursor-pointer border-border/80 p-5 transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  <span className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <Icon size={20} />
                  </span>
                  <h3 className="font-semibold text-card-foreground">{title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{description}</p>
                </Card>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, icon, tone }: { label: string; value: string; icon: React.ReactNode; tone: string }) {
  return (
    <Card className="border-border/80 p-5">
      <CardHeader className="flex-row items-start justify-between space-y-0 p-0">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
        <span className={cn("inline-flex h-9 w-9 items-center justify-center rounded-lg", iconTones[tone])}>{icon}</span>
      </CardHeader>
      <CardContent className="mt-5 p-0">
        <p className="text-3xl font-semibold tabular-nums text-foreground">{value}</p>
      </CardContent>
    </Card>
  );
}
