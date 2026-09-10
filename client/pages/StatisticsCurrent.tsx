import { useEffect, useState } from "react";
import { BarChart3, FileStack, RefreshCw, RotateCcw, ScanLine, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
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
type DoctorStat = { name: string; department: string; patientCount: number };
type InstitutionOption = { id: string; name: string; admin: string | null };
type FilterState = { dateFrom?: string; dateTo?: string; institutionId?: string; department?: string; doctor?: string };

const adminRoles = ["admin", "system_admin", "institution_admin"];
const pieColors = ["#16A34A", "#D97706", "#EA580C", "#DC2626"];
const selectCls = "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2";

// 后端返回的英文等级名 → 翻译 key
const aisNameKey = (name: string): string => {
  const map: Record<string, string> = { Normal: "enums.severityNegative", Mild: "enums.severityMild", Moderate: "enums.severityModerate", Severe: "enums.severitySevere" };
  return map[name] || "";
};

function DoctorBarTooltip({ active, payload }: { active?: boolean; payload?: ReadonlyArray<{ payload: DoctorStat }> }) {
  const { t } = useTranslation();
  if (!active || !payload || payload.length === 0) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="font-semibold text-foreground">{d.name}</p>
      {d.department && d.department !== "未分配" && <p className="mt-0.5 text-xs text-muted-foreground">{d.department}</p>}
      <p className="mt-1 font-semibold tabular-nums text-primary">{d.patientCount}{t("stats.patientUnit")}</p>
    </div>
  );
}

export default function StatisticsPage() {
  const { t } = useTranslation();
  const [overview, setOverview] = useState<Overview | null>(null);
  const [aisData, setAisData] = useState<Distribution[]>([]);
  const [trendData, setTrendData] = useState<TrendPoint[]>([]);
  const [doctorData, setDoctorData] = useState<DoctorStat[]>([]);
  const [institutions, setInstitutions] = useState<InstitutionOption[]>([]);
  const [doctorOptions, setDoctorOptions] = useState<DoctorStat[]>([]);
  const [departmentOptions, setDepartmentOptions] = useState<Distribution[]>([]);
  const [filters, setFilters] = useState<FilterState>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const isAdmin = adminRoles.includes(sessionStorage.getItem("user_role") || "");

  // 加载筛选所需的选项（机构 / 医生 / 科室）
  useEffect(() => {
    api.getInstitutions().then(setInstitutions).catch(() => undefined);
    api.getDoctorDistribution().then(setDoctorOptions).catch(() => undefined);
    api.getCasesDistribution("department").then(setDepartmentOptions).catch(() => undefined);
  }, []);

  const activeFilters = (): Record<string, string> => {
    const q: Record<string, string> = {};
    (Object.keys(filters) as (keyof FilterState)[]).forEach((key) => {
      const value = filters[key];
      if (value) q[key] = value;
    });
    return q;
  };

  const load = async () => {
    try {
      setLoading(true); setError("");
      const q = activeFilters();
      const [nextOverview, ais, doctors, trend] = await Promise.all([api.getStatistics(q), api.getAISDistribution(q), api.getDoctorDistribution(q), api.getTimeSeries("cases", "week")]);
      setOverview(nextOverview);
      setAisData(ais);
      setDoctorData(doctors);
      setTrendData(trend.map((item: TrendPoint) => ({ ...item, date: item.date.slice(5) })));
    } catch (caught) { setError(caught instanceof Error ? caught.message : t("stats.loadFailed")); } finally { setLoading(false); }
  };

  useEffect(() => { void load(); }, []);

  const setFilter = (key: keyof FilterState, value: string) => setFilters((prev) => ({ ...prev, [key]: value }));
  const resetFilters = () => setFilters({});

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
        <div className="content-wrapper space-y-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-primary">{t("stats.eyebrow")}</p>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("stats.title")}</h1>
            </div>
            <Button variant="outline" onClick={() => void load()} disabled={loading}>
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />{t("stats.refresh")}
            </Button>
          </div>

          {/* 筛选栏：时间 / 机构 / 科室 / 人员 */}
          <Card className="border-border/80 p-4">
            <div className="flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">{t("stats.filterTimeFrom")}
                <input type="date" className={selectCls} value={filters.dateFrom || ""} onChange={(e) => setFilter("dateFrom", e.target.value)} />
              </label>
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">{t("stats.filterTo")}
                <input type="date" className={selectCls} value={filters.dateTo || ""} onChange={(e) => setFilter("dateTo", e.target.value)} />
              </label>
              {institutions.length > 1 && (
                <label className="flex flex-col gap-1 text-xs text-muted-foreground">{t("stats.filterInstitution")}
                  <select className={selectCls} value={filters.institutionId || ""} onChange={(e) => setFilter("institutionId", e.target.value)}>
                    <option value="">{t("stats.filterAll")}</option>
                    {institutions.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}
                  </select>
                </label>
              )}
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">{t("stats.filterDepartment")}
                <select className={selectCls} value={filters.department || ""} onChange={(e) => setFilter("department", e.target.value)}>
                  <option value="">{t("stats.filterAll")}</option>
                  {departmentOptions.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">{t("stats.filterDoctor")}
                <select className={selectCls} value={filters.doctor || ""} onChange={(e) => setFilter("doctor", e.target.value)}>
                  <option value="">{t("stats.filterAll")}</option>
                  {doctorOptions.map((d) => <option key={d.name} value={d.name}>{d.name}{d.department ? `（${d.department}）` : ""}</option>)}
                </select>
              </label>
              <Button variant="outline" onClick={() => { resetFilters(); }} className="h-10">
                <RotateCcw size={15} />{t("stats.filterReset")}
              </Button>
              <Button variant="outline" onClick={() => void load()} className="h-10">
                <RefreshCw size={15} />{t("stats.filterApply")}
              </Button>
            </div>
          </Card>

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

              <section>
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
              </section>

              {/* 医生分析统计：每个医生做了多少患者的报告（患者有该医生的报告即记 1） */}
              <section>
                <Card className="border-border/80 p-5 md:p-6">
                  <div className="mb-5">
                    <h2 className="text-lg font-semibold text-foreground">{t("stats.doctorTitle")}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("stats.doctorHint")}</p>
                  </div>
                  {doctorData.length === 0 ? (
                    <p className="py-8 text-center text-sm text-muted-foreground">{t("stats.noData")}</p>
                  ) : (
                    <div
                      className="w-full"
                      style={{ height: Math.min(620, Math.max(300, doctorData.length * 52)) }}
                      aria-label={t("stats.chartDoctorLabel")}
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={[...doctorData].sort((a, b) => b.patientCount - a.patientCount)} margin={{ top: 28, right: 12, left: -14, bottom: 0 }}>
                          <CartesianGrid vertical={false} stroke="#DBEAFE" />
                          <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 12 }} />
                          <YAxis allowDecimals={false} tickLine={false} axisLine={false} width={44} />
                          <Tooltip cursor={{ fill: "rgba(148, 163, 184, 0.14)" }} content={<DoctorBarTooltip />} />
                          <Bar dataKey="patientCount" fill="#1E40AF" radius={[6, 6, 0, 0]} maxBarSize={56}>
                            <LabelList dataKey="patientCount" position="top" style={{ fill: "#1E40AF", fontSize: 13, fontWeight: 600 }} />
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
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
