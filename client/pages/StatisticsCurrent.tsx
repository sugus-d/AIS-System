import { useEffect, useState } from "react";
import { RefreshCw, RotateCcw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import api from "@/lib/api";

type Overview = {
  cases: { total: number };
  caseStatus: { total: number; analyzed: number; analyzing: number; pending: number; insufficient: number };
  files: { total: number };
  reports: { total: number; completed: number; success: number; pendingReview: number; successRate: string };
  tasks: { total: number; successRate: string };
  metrics: { avgCobbAngle: string; positiveRate: string; moderateOrAboveRate: string };
};

type Distribution = { name: string; value: number; color?: string };
type TrendPoint = { date: string; value: number };
type DoctorStat = { name: string; department: string; patientCount: number };
type InstitutionOption = { id: string; name: string; admin: string | null };
type FilterState = { dateFrom?: string; dateTo?: string; institutionId?: string; department?: string; doctor?: string };

const adminRoles = ["admin", "system_admin", "institution_admin"];
const pieColors = ["#16A34A", "#D97706", "#EA580C", "#DC2626"];
const selectCls = "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2";

// 快捷时间范围：一天 / 一周 / 一月（默认一周，含今天）
type RangePreset = "day" | "week" | "month" | "custom";
const isoDate = (date: Date) => `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
const presetRange = (kind: "day" | "week" | "month"): FilterState => {
  const today = new Date();
  const days = kind === "day" ? 0 : kind === "week" ? 6 : 29;
  return {
    dateFrom: isoDate(new Date(today.getFullYear(), today.getMonth(), today.getDate() - days)),
    dateTo: isoDate(today),
  };
};

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
  const [filters, setFilters] = useState<FilterState>(() => presetRange("week"));
  const [preset, setPreset] = useState<RangePreset>("week");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const isAdmin = adminRoles.includes(sessionStorage.getItem("user_role") || "");

  // 加载筛选所需的选项（机构 / 医生 / 科室）
  useEffect(() => {
    api.getInstitutions().then(setInstitutions).catch(() => undefined);
    api.getDoctorDistribution().then(setDoctorOptions).catch(() => undefined);
    api.getCasesDistribution("department").then(setDepartmentOptions).catch(() => undefined);
  }, []);

  const queryOf = (source: FilterState): Record<string, string> => {
    const q: Record<string, string> = {};
    (Object.keys(source) as (keyof FilterState)[]).forEach((key) => {
      const value = source[key];
      if (value) q[key] = value;
    });
    return q;
  };

  const load = async (override?: FilterState) => {
    try {
      setLoading(true); setError("");
      const q = queryOf(override ?? filters);
      const [nextOverview, ais, doctors, trend] = await Promise.all([api.getStatistics(q), api.getAISDistribution(q), api.getDoctorDistribution(q), api.getTimeSeries("cases", q)]);
      setOverview(nextOverview);
      setAisData(ais);
      setDoctorData(doctors);
      // 趋势随所选时间范围联动（后端按范围聚合：≤14天按天、≤120天按周、更长按月）
      setTrendData(Array.isArray(trend) ? trend : []);
    } catch (caught) { setError(caught instanceof Error ? caught.message : t("stats.loadFailed")); } finally { setLoading(false); }
  };

  useEffect(() => { void load(); }, []);

  const setFilter = (key: keyof FilterState, value: string) => {
    setPreset("custom");
    setFilters((prev) => ({ ...prev, [key]: value }));
  };
  const applyPreset = (kind: "day" | "week" | "month") => {
    const next = { ...filters, ...presetRange(kind) };
    setPreset(kind);
    setFilters(next);
    void load(next);
  };
  const resetFilters = () => {
    const next = presetRange("week");
    setPreset("week");
    setFilters(next);
    void load(next);
  };

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
              <div className="flex flex-col gap-1 text-xs text-muted-foreground">
                <span>{t("stats.rangeQuick")}</span>
                <div className="flex items-center gap-1.5">
                  {(["day", "week", "month"] as const).map((kind) => (
                    <button
                      key={kind}
                      type="button"
                      onClick={() => applyPreset(kind)}
                      className={`h-10 rounded-md border px-3 text-sm transition-colors ${preset === kind ? "border-primary bg-primary/10 font-semibold text-primary" : "border-input text-muted-foreground hover:bg-muted"}`}
                    >
                      {kind === "day" ? t("stats.rangeDay") : kind === "week" ? t("stats.rangeWeek") : t("stats.rangeMonth")}
                    </button>
                  ))}
                </div>
              </div>
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">{t("stats.filterTimeFrom")}
                <input type="date" className={selectCls} value={filters.dateFrom || ""} onChange={(e) => setFilter("dateFrom", e.target.value)} />
              </label>
              <label className="flex flex-col gap-1 text-xs text-muted-foreground">{t("stats.filterTo")}
                <input type="date" className={selectCls} value={filters.dateTo || ""} onChange={(e) => setFilter("dateTo", e.target.value)} />
              </label>
              {institutions.length > 0 && (
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
              {/* 首行三卡：受检者状态 / 分析结果 / 分析运行 */}
              <section className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
                <Card className="border-border/80 p-4 md:p-5">
                  <div className="mb-4">
                    <h2 className="text-base font-semibold text-foreground">{t("stats.caseStatusTitle")}</h2>
                    <p className="mt-1 text-xs text-muted-foreground">{t("stats.caseStatusHint")}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-5">
                    <Stat label={t("stats.casesCreated")} value={overview?.caseStatus.total} loading={loading} />
                    <Stat label={t("stats.casesAnalyzed")} value={overview?.caseStatus.analyzed} loading={loading} />
                    <Stat label={t("stats.casesAnalyzing")} value={overview?.caseStatus.analyzing} loading={loading} />
                    <Stat label={t("stats.casesPendingAnalysis")} value={overview?.caseStatus.pending} loading={loading} />
                    <Stat label={t("stats.casesInsufficient")} value={overview?.caseStatus.insufficient} loading={loading} />
                  </div>
                </Card>

                <Card className="border-border/80 p-4 md:p-5">
                  <div className="mb-4">
                    <h2 className="text-base font-semibold text-foreground">{t("stats.resultTitle")}</h2>
                    <p className="mt-1 text-xs text-muted-foreground">{t("stats.resultHint")}</p>
                  </div>
                  <div className="space-y-4">
                    <Stat label={t("stats.avgCobb")} value={overview ? `${overview.metrics.avgCobbAngle}°` : undefined} loading={loading} />
                    <Stat label={t("stats.positiveRate")} value={overview ? `${overview.metrics.positiveRate}%` : undefined} loading={loading} />
                    <Stat label={t("stats.moderateOrAboveRate")} value={overview ? `${overview.metrics.moderateOrAboveRate}%` : undefined} loading={loading} />
                  </div>
                </Card>

                <Card className="border-border/80 p-4 md:p-5">
                  <div className="mb-4">
                    <h2 className="text-base font-semibold text-foreground">{t("stats.runTitle")}</h2>
                    <p className="mt-1 text-xs text-muted-foreground">{t("stats.runHint")}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-5">
                    <Stat label={t("stats.totalReports")} value={overview?.reports.total} loading={loading} />
                    <Stat label={t("stats.analysisSuccess")} value={overview?.reports.success} loading={loading} />
                    <Stat label={t("stats.successRate")} value={overview ? `${overview.reports.successRate}%` : undefined} loading={loading} />
                    <Stat label={t("stats.pendingReview")} value={overview?.reports.pendingReview} loading={loading} />
                  </div>
                </Card>
              </section>

              {/* 第二行：新增受检者数量趋势 + AIS 分级 */}
              <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <Card className="border-border/80 p-5 md:p-6">
                  <div className="mb-5">
                    <h2 className="text-lg font-semibold text-foreground">{t("stats.trendTitle")}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("stats.trendHint", { from: filters.dateFrom || "--", to: filters.dateTo || "--" })}</p>
                  </div>
                  <div className="h-[300px]" aria-label={t("stats.chartTrendLabel")}>
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
                  <div className="mb-3">
                    <h2 className="text-lg font-semibold text-foreground">{t("stats.aisTitle")}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("stats.aisHint")}</p>
                  </div>
                  <div className="h-[300px]" aria-label={t("stats.chartAisLabel")}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={aisData} dataKey="value" nameKey="name" innerRadius={70} outerRadius={110} paddingAngle={3}>
                          {aisData.map((item, index) => <Cell key={item.name} fill={item.color || pieColors[index]} />)}
                        </Pie>
                        <Tooltip formatter={(value: number) => [value, t("stats.tooltipCases")]} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    {aisData.map((item, index) => (
                      <div className="flex items-center justify-between gap-2" key={item.name}>
                        <span className="flex items-center gap-2 text-muted-foreground">
                          <i className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: item.color || pieColors[index] }} />
                          {aisNameKey(item.name) ? t(aisNameKey(item.name)) : item.name}
                        </span>
                        <strong className="tabular-nums text-foreground">{item.value}</strong>
                      </div>
                    ))}
                  </div>
                </Card>
              </section>

              {/* 第三行：医生分析统计（整行，柱状图更舒展） */}
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
