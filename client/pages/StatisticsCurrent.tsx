import { useEffect, useState } from "react";
import { BarChart3, FileStack, RefreshCw, ScanLine, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import api from "@/lib/api";

type Overview = { cases: { total: number }; files: { total: number }; reports: { total: number; completed: number }; tasks: { total: number; successRate: string }; metrics: { avgCobbAngle: string; positiveRate: string } };
type Distribution = { name: string; value: number; color?: string };
type TrendPoint = { date: string; value: number };

const adminRoles = ["admin", "system_admin", "institution_admin"];
const pieColors = ["#16A34A", "#D97706", "#EA580C", "#DC2626"];

export default function StatisticsPage() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [departmentData, setDepartmentData] = useState<Distribution[]>([]);
  const [aisData, setAisData] = useState<Distribution[]>([]);
  const [trendData, setTrendData] = useState<TrendPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const isAdmin = adminRoles.includes(localStorage.getItem("user_role") || "");

  const load = async () => {
    try {
      setLoading(true);
      setError("");
      const [nextOverview, departments, ais, trend] = await Promise.all([api.getStatistics(), api.getCasesDistribution("department"), api.getAISDistribution(), api.getTimeSeries("cases", "week")]);
      setOverview(nextOverview);
      setDepartmentData(departments);
      setAisData(ais);
      setTrendData(trend.map((item: TrendPoint) => ({ ...item, date: item.date.slice(5) })));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "统计数据加载失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);
  const metrics = [
    { label: "受检者总量", value: overview?.cases.total, icon: Users, tone: "bg-blue-50 text-blue-700" },
    { label: "筛查文件", value: overview?.files.total, icon: FileStack, tone: "bg-emerald-50 text-emerald-700" },
    { label: "已生成报告", value: overview?.reports.total, icon: ScanLine, tone: "bg-amber-50 text-amber-700" },
    { label: "任务成功率", value: overview ? `${overview.tasks.successRate}%` : undefined, icon: BarChart3, tone: "bg-rose-50 text-rose-700" },
  ];

  return <div className="layout-main">
    <Sidebar isAdmin={isAdmin} />
    <div className="layout-header"><Header isAdmin={isAdmin} /></div>
    <main className="layout-content"><div className="content-wrapper space-y-8">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between"><div><p className="text-helper uppercase tracking-wider text-[color:var(--color-primary)] mb-2">Analytics</p><h1 className="text-page-title">数据统计</h1><p className="text-body mt-2">基于本地 SQLite 数据库的实时统计概览。</p></div><button className="btn-secondary inline-flex items-center justify-center gap-2" onClick={() => void load()} disabled={loading}><RefreshCw size={16} className={loading ? "animate-spin" : ""} />刷新数据</button></div>
      {error ? <section className="border border-red-200 bg-red-50 px-5 py-4 text-[color:var(--color-error)] rounded-card flex items-center justify-between gap-4"><span>{error}</span><button className="btn-text" onClick={() => void load()}>重试</button></section> : <>
        <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4" aria-label="核心指标">{metrics.map(({ label, value, icon: Icon, tone }) => <div key={label} className="card-base p-5"><div className="flex items-start justify-between"><p className="text-helper">{label}</p><span className={`inline-flex w-9 h-9 items-center justify-center rounded-lg ${tone}`}><Icon size={20} /></span></div><p className="text-data-lg mt-5 text-[color:var(--color-text-primary)]">{loading ? "--" : value ?? 0}</p></div>)}</section>
        <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="card-base p-5 md:p-6"><div className="mb-5"><h2 className="text-card-title">近七日建档趋势</h2><p className="text-helper mt-1">每日新增受检者数量</p></div><div className="h-[280px]" aria-label="近七日建档趋势图"><ResponsiveContainer width="100%" height="100%"><LineChart data={trendData} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}><CartesianGrid vertical={false} stroke="#DBEAFE" /><XAxis dataKey="date" tickLine={false} axisLine={false} /><YAxis allowDecimals={false} tickLine={false} axisLine={false} /><Tooltip formatter={(value: number) => [value, "新增受检者"]} /><Line type="monotone" dataKey="value" stroke="#1E40AF" strokeWidth={3} dot={{ r: 3 }} activeDot={{ r: 5 }} /></LineChart></ResponsiveContainer></div></div>
          <div className="card-base p-5 md:p-6"><div className="mb-5"><h2 className="text-card-title">科室分布</h2><p className="text-helper mt-1">按受检者所属科室统计</p></div><div className="h-[280px]" aria-label="科室分布图"><ResponsiveContainer width="100%" height="100%"><BarChart data={departmentData} layout="vertical" margin={{ top: 4, right: 18, left: 18, bottom: 0 }}><CartesianGrid horizontal={false} stroke="#DBEAFE" /><XAxis type="number" allowDecimals={false} tickLine={false} axisLine={false} /><YAxis dataKey="name" type="category" width={86} tickLine={false} axisLine={false} /><Tooltip formatter={(value: number) => [value, "受检者"]} /><Bar dataKey="value" fill="#3B82F6" radius={[4, 4, 4, 4]} /></BarChart></ResponsiveContainer></div></div>
        </section>
        <section className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px] gap-6">
          <div className="card-base p-5 md:p-6"><div className="mb-5"><h2 className="text-card-title">报告与风险指标</h2><p className="text-helper mt-1">当前数据集的分析结果汇总</p></div><div className="grid grid-cols-2 md:grid-cols-4 gap-5"><Stat label="已完成报告" value={overview?.reports.completed} loading={loading} /><Stat label="分析任务" value={overview?.tasks.total} loading={loading} /><Stat label="平均 Cobb 角" value={overview ? `${overview.metrics.avgCobbAngle}°` : undefined} loading={loading} /><Stat label="阳性比例" value={overview ? `${overview.metrics.positiveRate}%` : undefined} loading={loading} /></div></div>
          <div className="card-base p-5 md:p-6"><div className="mb-3"><h2 className="text-card-title">AIS 分级</h2><p className="text-helper mt-1">按最新分析结果统计</p></div><div className="h-[210px]" aria-label="AIS 分级分布图"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={aisData} dataKey="value" nameKey="name" innerRadius={54} outerRadius={80} paddingAngle={3}>{aisData.map((item, index) => <Cell key={item.name} fill={item.color || pieColors[index]} />)}</Pie><Tooltip formatter={(value: number) => [value, "受检者"]} /></PieChart></ResponsiveContainer></div><div className="grid grid-cols-2 gap-x-3 gap-y-2">{aisData.map((item, index) => <div className="flex items-center justify-between gap-2 text-helper" key={item.name}><span className="flex items-center gap-2"><i className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color || pieColors[index] }} />{item.name}</span><strong>{item.value}</strong></div>)}</div></div>
        </section>
      </>}
    </div></main>
  </div>;
}

function Stat({ label, value, loading }: { label: string; value?: string | number; loading: boolean }) { return <div><p className="text-helper">{label}</p><p className="text-card-title mt-2">{loading ? "--" : value ?? 0}</p></div>; }
