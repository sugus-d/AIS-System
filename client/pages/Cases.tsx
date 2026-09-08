import { useEffect, useMemo, useState } from "react";
import { Search, SlidersHorizontal, ArrowUpDown, Eye, Loader2, Plus, Trash2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { confirmDialog } from "@/components/ConfirmDialog";
import { toast } from "sonner";
import api from "@/lib/api";

const adminRoles = ["admin", "system_admin", "institution_admin"];
const statusLabels: Record<string, string> = { pending_upload: "enums.statusPendingUpload", pending_analysis: "enums.statusPendingAnalysis", analyzing: "enums.statusAnalyzing", under_review: "enums.statusUnderReview", approved: "enums.statusApproved", review_returned: "enums.statusReviewReturned" };

const statusBadge: Record<string, string> = {
  approved: "border-emerald-200 bg-emerald-50 text-emerald-700",
  analyzing: "border-amber-200 bg-amber-50 text-amber-700",
  under_review: "border-amber-200 bg-amber-50 text-amber-700",
  review_returned: "border-red-200 bg-red-50 text-red-700",
  pending_upload: "border-slate-200 bg-slate-100 text-slate-600",
  pending_analysis: "border-red-200 bg-red-50 text-red-700",
};

const severityMeta: Record<string, { color: string; labelKey: string }> = {
  Normal: { color: "text-emerald-600", labelKey: "enums.severityNegative" },
  Mild: { color: "text-yellow-600", labelKey: "enums.severityMild" },
  Moderate: { color: "text-orange-600", labelKey: "enums.severityModerate" },
  Severe: { color: "text-red-600", labelKey: "enums.severitySevere" },
};

const selectCls = "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2";

export default function Cases() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const isAdmin = adminRoles.includes(sessionStorage.getItem("user_role") || "");
  const [items, setItems] = useState<any[]>([]);
  const [keyword, setKeyword] = useState("");
  const [status, setStatus] = useState("all");
  const [sort, setSort] = useState("createdAt");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const load = async () => {
    try {
      setLoading(true); setError("");
      const r = await api.getCases({ pageSize: 100, keyword: keyword || undefined, status: status === "all" ? undefined : status, sortBy: sort, sortOrder: order } as any);
      setItems(r.list || r.data?.list || []);
    } catch (e) { setError(e instanceof Error ? e.message : t("cases.loadFailed")); } finally { setLoading(false); }
  };

  const del = async (c: any) => {
    if (deletingId) return;
    if (!(await confirmDialog(t("cases.deleteConfirm", { name: c.name || c.caseNumber || c.id }), { destructive: true }))) return;
    try { setDeletingId(c.id); await api.deleteCase(c.id); await load(); } catch (e) { toast.error(e instanceof Error ? e.message : t("cases.deleteFailed")); } finally { setDeletingId(null); }
  };

  useEffect(() => { const timer = setTimeout(load, 250); return () => clearTimeout(timer); }, [keyword, status, sort, order]);

  const display = useMemo(() => {
    const arr = [...items];
    const key = sort === "name" ? "name" : sort === "latestReportTime" ? "latestReportTime" : "createdAt";
    arr.sort((a, b) => { const av = String(a[key] ?? ""); const bv = String(b[key] ?? ""); const cmp = av.localeCompare(bv); return order === "desc" ? -cmp : cmp; });
    return arr;
  }, [items, sort, order]);

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header"><Header isAdmin={isAdmin} /></div>
      <div className="layout-content">
        <div className="content-wrapper space-y-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-primary">{t("cases.eyebrow")}</p>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("cases.title")}</h1>
            </div>
            <Button onClick={() => navigate("/case-record")}><Plus size={17} />{t("cases.newCase")}</Button>
          </div>

          <Card className="border-border/80 p-4">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_180px_180px_auto]">
              <div className="relative">
                <Search size={17} className="absolute top-1/2 left-3 -translate-y-1/2 text-muted-foreground" />
                <Input className="pl-9" value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder={t("cases.searchPlaceholder")} />
              </div>
              <select className={selectCls} value={status} onChange={(e) => setStatus(e.target.value)}>
                <option value="all">{t("cases.allStatus")}</option>
                {Object.entries(statusLabels).map(([k, v]) => <option key={k} value={k}>{t(v)}</option>)}
              </select>
              <select className={selectCls} value={sort} onChange={(e) => setSort(e.target.value)}>
                <option value="createdAt">{t("cases.sortCreated")}</option>
                <option value="latestReportTime">{t("cases.sortLatest")}</option>
                <option value="name">{t("cases.sortName")}</option>
              </select>
              <Button type="button" variant="outline" onClick={() => setOrder((o) => (o === "desc" ? "asc" : "desc"))}>
                <ArrowUpDown size={16} />{order === "desc" ? t("cases.desc") : t("cases.asc")}
              </Button>
            </div>
          </Card>

          <Card className="border-border/80 overflow-hidden">
            <div className="flex items-center justify-between border-b border-border/60 px-5 py-4">
              <div className="flex items-center gap-2">
                <SlidersHorizontal size={17} className="text-primary" />
                <span className="font-semibold text-foreground">{t("cases.listTitle")}</span>
              </div>
              <span className="text-sm text-muted-foreground">{t("cases.total", { count: display.length })}</span>
            </div>

            {loading ? (
              <div className="py-16 text-center text-muted-foreground">{t("cases.loading")}</div>
            ) : error ? (
              <div className="py-16 text-center">
                <p className="mb-3 text-destructive">{error}</p>
                <Button variant="outline" onClick={load}>{t("cases.reload")}</Button>
              </div>
            ) : display.length === 0 ? (
              <div className="py-16 text-center text-muted-foreground">{t("cases.empty")}</div>
            ) : (
              <div className="overflow-x-auto">
                <Table className="min-w-[760px]">
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead>{t("cases.colCaseNumber")}</TableHead>
                      <TableHead>{t("cases.colName")}</TableHead>
                      <TableHead>{t("cases.colGender")}</TableHead>
                      <TableHead>{t("cases.colStatus")}</TableHead>
                      <TableHead>{t("cases.colResult")}</TableHead>
                      <TableHead>{t("cases.colLatest")}</TableHead>
                      <TableHead>{t("cases.colCreated")}</TableHead>
                      <TableHead className="text-right">{t("cases.colOps")}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {display.map((c) => (
                      <TableRow key={c.id} className="hover:bg-primary/5">
                        <TableCell className="font-medium text-primary">{c.caseNumber || c.id}</TableCell>
                        <TableCell className="font-semibold text-foreground">{c.name || "--"}</TableCell>
                        <TableCell>{c.gender === "female" || c.gender === "女" ? t("enums.genderFemale") : t("enums.genderMale")}</TableCell>
                        <TableCell><Status value={c.status} /></TableCell>
                        <TableCell className="whitespace-nowrap">
                          {c.latestReport ? (
                            <span className={`font-semibold ${severityMeta[c.latestReport.severity]?.color || "text-foreground"}`}>
                              {t("cases.cobbResult", {
                                angle: Number.isFinite(Number(c.latestReport.cobbAngle)) ? Math.round(Number(c.latestReport.cobbAngle)) : "--",
                                level: severityMeta[c.latestReport.severity] ? t(severityMeta[c.latestReport.severity].labelKey) : c.latestReport.severity,
                              })}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">--</span>
                          )}
                        </TableCell>
                        <TableCell className="whitespace-nowrap">{formatDate(c.latestReportTime)}</TableCell>
                        <TableCell className="whitespace-nowrap">{formatDate(c.createdAt)}</TableCell>
                        <TableCell className="whitespace-nowrap text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button variant="ghost" size="sm" className="h-8 gap-1 px-2" onClick={() => navigate(`/case-detail/${c.id}`)}>
                              <Eye size={16} />{t("cases.viewDetail")}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 gap-1 px-2 text-destructive hover:bg-destructive/10 hover:text-destructive"
                              disabled={deletingId === c.id}
                              onClick={() => void del(c)}
                              title={t("cases.delete")}
                            >
                              {deletingId === c.id ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
                              {t("cases.delete")}
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

function formatDate(value?: string) {
  if (!value) return "--";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "--";
  const parts = new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).formatToParts(d);
  const get = (type: string) => parts.find((part) => part.type === type)?.value || "";
  return `${get("year")}-${get("month")}-${get("day")} ${get("hour")}:${get("minute")}`;
}

function Status({ value }: { value?: string }) {
  const { t } = useTranslation();
  const label = statusLabels[value || ""] ? t(statusLabels[value || ""]) : value || t("enums.statusPendingUpload");
  const cls = statusBadge[value || ""] || statusBadge.pending_analysis;
  return <Badge variant="outline" className={cn("font-medium", cls)}>{label}</Badge>;
}
