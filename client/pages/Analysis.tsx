import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { LoaderCircle, Play, RefreshCw } from "lucide-react";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import api from "@/lib/api";

type CaseItem = { id: string; caseNumber: string; name: string; gender: string; height?: number; weight?: number; fileCount?: number };
type Task = { id: string; status: string; progress: number; failureReason?: string; resultJson?: string };
const labels: Record<string, string> = { pending: "Waiting", running: "Analyzing", success: "Complete", failed: "Failed", cancelled: "Cancelled" };

export default function Analysis() {
  const navigate = useNavigate(); const [params] = useSearchParams(); const [cases, setCases] = useState<CaseItem[]>([]); const [selected, setSelected] = useState<string[]>([]); const [tasks, setTasks] = useState<Task[]>([]); const [loading, setLoading] = useState(false); const [message, setMessage] = useState("");
  const isAdmin = sessionStorage.getItem("user_role") === "system_admin" || sessionStorage.getItem("user_role") === "admin";
  const load = async () => { const result = await api.getCases({ pageSize: 100 }); const list = result.list || result.data?.list || []; setCases(list); const target = params.get("id"); if (target && list.some((item: CaseItem) => item.id === target)) setSelected([target]); };
  useEffect(() => { void load().catch((error) => setMessage(error.message)); }, []);
  useEffect(() => { if (!tasks.some((task) => task.status === "pending" || task.status === "running")) return; const timer = window.setInterval(() => { void Promise.all(tasks.map((task) => api.getTask(task.id).catch(() => task))).then((next) => setTasks(next)); }, 1500); return () => window.clearInterval(timer); }, [tasks]);
  const eligible = useMemo(() => cases.filter((item) => selected.includes(item.id) && item.fileCount && item.height && item.weight && item.gender), [cases, selected]);
  const submit = async (batch: boolean) => { if (!eligible.length) return setMessage("Select a case with clinical data and an uploaded PLY scan."); setLoading(true); setMessage(""); try { if (batch) { const result = await api.analyzeBatch(eligible.map((item) => item.id)); const queued = result.tasks || result.data?.tasks || []; setTasks((old) => [...queued, ...old]); const skipped = result.skipped || result.data?.skipped || []; if (skipped.length) setMessage(`${skipped.length} case(s) were skipped.`); } else { const task = await api.analyzeSingle(eligible[0].id); setTasks((old) => [task, ...old]); } } catch (error: any) { setMessage(error.message || "Unable to queue analysis."); } finally { setLoading(false); } };

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header"><Header isAdmin={isAdmin} /></div>
      <main className="layout-content space-y-5">
        <div className="content-wrapper space-y-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground">AIS Analysis</h1>
              <p className="mt-1 text-muted-foreground">Submit uploaded PLY scans to the local AIS algorithm.</p>
            </div>
            <Button variant="outline" size="icon" title="Refresh cases" onClick={() => void load()}><RefreshCw size={18} /></Button>
          </div>

          {message && <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">{message}</div>}

          <section className="space-y-3">
            <div className="flex gap-2">
              <Button disabled={loading || eligible.length !== 1} onClick={() => void submit(false)}>
                {loading ? <LoaderCircle className="animate-spin" size={16} /> : <Play size={16} />}Analyze selected
              </Button>
              <Button variant="secondary" disabled={loading || !eligible.length} onClick={() => void submit(true)}>Analyze selected batch</Button>
            </div>
            <Card className="overflow-hidden border-border/80">
              <div className="overflow-x-auto">
                <Table className="min-w-[640px]">
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="w-12"></TableHead>
                      <TableHead>Case</TableHead>
                      <TableHead>Sex</TableHead>
                      <TableHead>Scan</TableHead>
                      <TableHead>Ready</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {cases.map((item) => {
                      const ready = Boolean(item.fileCount && item.height && item.weight && item.gender);
                      return (
                        <TableRow key={item.id} className="hover:bg-primary/5">
                          <TableCell>
                            <Checkbox checked={selected.includes(item.id)} disabled={!ready} onCheckedChange={() => setSelected((old) => old.includes(item.id) ? old.filter((id) => id !== item.id) : [...old, item.id])} />
                          </TableCell>
                          <TableCell className="font-medium text-foreground">{item.caseNumber} {item.name}</TableCell>
                          <TableCell>{item.gender}</TableCell>
                          <TableCell>{item.fileCount || 0}</TableCell>
                          <TableCell className={ready ? "text-emerald-600" : "text-muted-foreground"}>{ready ? "Ready" : "Clinical data or PLY scan required"}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </Card>
          </section>

          {tasks.length > 0 && (
            <section>
              <h2 className="mb-3 text-lg font-semibold text-foreground">Local tasks</h2>
              <div className="space-y-2">
                {tasks.map((task) => (
                  <div key={task.id} className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-4 py-2.5 shadow-sm">
                    <span className="truncate font-mono text-xs text-muted-foreground">{task.id}</span>
                    <span className="whitespace-nowrap text-sm font-medium text-foreground">{labels[task.status] || task.status} {task.progress}%</span>
                    {task.status === "success" && <Button variant="link" size="sm" onClick={() => navigate("/reports")}>View reports</Button>}
                    {task.failureReason && <span className="text-sm text-destructive">{task.failureReason}</span>}
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
