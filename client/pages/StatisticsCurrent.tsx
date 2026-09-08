import { useEffect, useState } from "react";
import { BarChart3, FileStack, RefreshCw, ScanLine, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import api from "@/lib/api";

type Overview = {
  cases: { total: number };
  files: { total: number };
  reports: { total: number; completed: number };
  tasks: { total: number; successRate: string };
  metrics: { avgCobbAngle: string; positiveRate: string };
};

type Distribution = { name: string; value: number; color?: string };
type TrendPoint = { date: string; value: number };

const adminRoles = ["admin", "system_admin", "institution_admin"];
const pieColors = ["#16A34A", "#D97706", "#EA580C", "#DC2626"];

// 后端返回的英文等级名 → 翻译 key
const aisNameKey = (name: string): string => {
  const map: Record<string, string> = { Normal: "enums.severityNegative", Mild: "enums.severityMild", Moderate: "enums.severityModerate", Severe: "enums.severitySevere" };
  return map[name] || "";
};

export default function StatisticsPage() {
  const { t } = useTranslation();
  const [overview, setOverview] = useState<Overview | null>(null);
  const [departmentData, setDepartmentData] = useState<Distribution[]>([]);
  const [aisData, setAisData] = useState<Distribution[]>([]);
  const [trendData, setTrendData] = useState<TrendPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const isAdmin = adminRoles.includes(sessionStorage.getItem("user_role") || "");

  const load = async () => {
    try {
      setLoading(true); setError("");
      const [nextOverview, departments, ais, trend] = await Promise.all([api.getStatistics(), api.getCasesDistribution("department"), api.getAISDistribution(), api.getTimeSeries("cases", "week")]);
      setOverview(nextOverview);
      setDepartmentData(departments);
      setAisData(ais);
      setTrendData(trend.map((item: TrendPoint) => ({ ...item, date: item.date.slice(5) })));
    } catch (caught) { setError(caught instanceof Error ? caught.message : t("stats.loadFailed")); } finally { setLoading(false); }
  };

  useEffect(() => { void load(); }, []);

  const metrics = [
    { label: t("stats.totalCases"), value: overview?.cases.total, icon: Users, tone: "bg-blue-50 text-blue-700" },
    { label: t("stats.scanFiles"), value: overview?.files.total, icon: FileStack, tone: "bg-emerald-50 text-emerald-700" },
    { label: t("stats.generatedReports"), value: overview?.reports.total, icon: ScanLine, tone: "bg-amber-50 text-amber-700" },
    { label: t("stats.taskSuccessRate"), value: overview ? `${overview.tasks.successRate}%` : undefined, icon: BarChart3, tone: "bg-rose-50 text-rose-700" },
  ];

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header"><Header isAdmin={isAdmin} /></div>
      <main className="layout-content">
        <div className="content-wrapper space-y-8">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-primary">{t("stats.eyebrow")}</p>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("stats.title")}</h1>
            </div>
            <Button variant="outline" onClick={() => void load()} disabled={loading}>
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />{t("stats.refresh")}
            </Button>
          </div>

          {error ? (
            <section className="flex items-center justify-between gap-4 rounded-lg border border-red-200 bg-red-50 px-5 py-4 text-sm text-destructive">
              <span>{error}</span>
              <Button variant="ghost" onClick={() => void load()}>{t("stats.retry")}</Button>
            </section>
          ) : (
            <>
              <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label={t("stats.chartCoreMetrics")}>
                {metrics.map(({ label, value, icon: Icon, tone }) => (
                  <Card key={label} className="border-border/80 p-5">
                    <div className="flex items-start justify-between">
                      <p className="text-sm text-muted-foreground">{label}</p>
                      <span className={`inline-flex h-9 w-9 items-center justify-center rounded-lg ${tone}`}><Icon size={20} /></span>
                    </div>
                    <p className="mt-5 text-3xl font-semibold tabular-nums text-foreground">{loading ? "--" : value ?? 0}</p>
                  </Card>
                ))}
              </section>

              <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <Card className="border-border/80 p-5 md:p-6">
                  <div className="mb-5">
                    <h2 className="text-lg font-semibold text-foreground">{t("stats.trendTitle")}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("stats.trendHint")}</p>
                  </div>
                  <div className="h-[280px]" aria-label={t("stats.chartTrendLabel")}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trendData} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
                        <CartesianGrid vertical={false} stroke="#DBEAFE" />
                        <XAxis dataKey="date" tickLine={false} axisLine={false} />
                        <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                        <Tooltip formatter={(value: number) => [value, t("stats.trendTooltip")]} />
                        <Line type="monotone" dataKey="value" stroke="#1E40AF" strokeWidth={3} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                <Card className="border-border/80 p-5 md:p-6">
                  <div className="mb-5">
                    <h2 className="text-lg font-semibold text-foreground">{t("stats.deptTitle")}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("stats.deptHint")}</p>
                  </div>
                  <div className="h-[280px]" aria-label={t("stats.chartDeptLabel")}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={departmentData} layout="vertical" margin={{ top: 4, right: 18, left: 18, bottom: 0 }}>
                        <CartesianGrid horizontal={false} stroke="#DBEAFE" />
                        <XAxis type="number" allowDecimals={false} tickLine={false} axisLine={false} />
                        <YAxis dataKey="name" type="category" width={86} tickLine={false} axisLine={false} />
                        <Tooltip formatter={(value: number) => [value, t("stats.tooltipCases")]} />
                        <Bar dataKey="value" fill="#3B82F6" radius={[4, 4, 4, 4]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </section>

              <section className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
                <Card className="border-border/80 p-5 md:p-6">
                  <div className="mb-5">
                    <h2 className="text-lg font-semibold text-foreground">{t("stats.riskTitle")}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("stats.riskHint")}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-5 md:grid-cols-4">
                    <Stat label={t("stats.completedReports")} value={overview?.reports.completed} loading={loading} />
                    <Stat label={t("stats.analysisTasks")} value={overview?.tasks.total} loading={loading} />
                    <Stat label={t("stats.avgCobb")} value={overview ? `${overview.metrics.avgCobbAngle}°` : undefined} loading={loading} />
                    <Stat label={t("stats.positiveRate")} value={overview ? `${overview.metrics.positiveRate}%` : undefined} loading={loading} />
                  </div>
                </Card>

                <Card className="border-border/80 p-5 md:p-6">
                  <div className="mb-3">
                    <h2 className="text-lg font-semibold text-foreground">{t("stats.aisTitle")}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("stats.aisHint")}</p>
                  </div>
                  <div className="h-[210px]" aria-label={t("stats.chartAisLabel")}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={aisData} dataKey="value" nameKey="name" innerRadius={54} outerRadius={80} paddingAngle={3}>
                          {aisData.map((item, index) => <Cell key={item.name} fill={item.color || pieColors[index]} />)}
                        </Pie>
                        <Tooltip formatter={(value: number) => [value, t("stats.tooltipCases")]} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2">
                    {aisData.map((item, index) => (
                      <div className="flex items-center justify-between gap-2 text-sm" key={item.name}>
                        <span className="flex items-center gap-2 text-muted-foreground">
                          <i className="h-2 w-2 rounded-full" style={{ backgroundColor: item.color || pieColors[index] }} />
                          {aisNameKey(item.name) ? t(aisNameKey(item.name)) : item.name}
                        </span>
                        <strong className="text-foreground">{item.value}</strong>
                      </div>
                    ))}
                  </div>
                </Card>
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function Stat({ label, value, loading }: { label: string; value?: string | number; loading: boolean }) {
  return (
    <div>
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-2 text-xl font-semibold tabular-nums text-foreground">{loading ? "--" : value ?? 0}</p>
    </div>
  );
}
