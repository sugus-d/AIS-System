import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import api from "@/lib/api";

const roles: Record<string, string> = { system_admin: "系统管理员", institution_admin: "机构管理员", operator: "操作员" };

export default function PersonalSettings() {
  const navigate = useNavigate();
  const role = localStorage.getItem("user_role") || "operator";
  const [profile, setProfile] = useState({ username: "", name: "", department: "" });
  const [pwd, setPwd] = useState({ old: "", next: "", confirm: "" });
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const isAdmin = role === "system_admin" || role === "institution_admin";

  useEffect(() => { void api.getProfile().then((data) => setProfile({ username: data.username || "", name: data.name || "", department: data.department || "" })).catch((caught) => setError(caught instanceof Error ? caught.message : "个人资料加载失败。")); }, []);
  const save = async () => {
    if (!profile.name.trim() || !profile.department.trim()) return setError("姓名和科室为必填项。");
    setBusy(true); setError("");
    try { const data = await api.updateProfile({ name: profile.name.trim(), department: profile.department.trim() }); localStorage.setItem("user_name", data.name); localStorage.setItem("user_department", data.department || ""); setMessage("个人资料已保存到本地数据库。"); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "个人资料保存失败。"); }
    finally { setBusy(false); }
  };
  const changePassword = async () => {
    if (!pwd.old || !pwd.next || !pwd.confirm) return setError("请完整填写密码信息。");
    if (pwd.next.length < 12) return setError("新密码至少需要 12 位。");
    if (pwd.next !== pwd.confirm) return setError("两次输入的新密码不一致。");
    setBusy(true); setError("");
    try { await api.changePassword(pwd.old, pwd.next); setPwd({ old: "", next: "", confirm: "" }); setMessage("密码已更新。"); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "密码修改失败。"); }
    finally { setBusy(false); }
  };
  return <div className="layout-main"><Sidebar isAdmin={isAdmin} /><div className="layout-header"><Header isAdmin={isAdmin} /></div><main className="layout-content"><div className="content-wrapper space-y-6"><h1 className="text-page-title">个人设置</h1>{message && <div className="card-base p-4 text-[color:var(--color-primary)]">{message}</div>}{error && <div className="card-base p-4 text-[color:var(--color-error)]">{error}</div>}<div className="grid grid-cols-1 lg:grid-cols-2 gap-6"><section className="card-base p-8 space-y-4"><h2 className="text-card-title">个人资料</h2><ReadOnly label="账号" value={profile.username || "--"} /><ReadOnly label="角色" value={roles[role] || "操作员"} /><Field label="姓名 *" value={profile.name} onChange={(value) => setProfile((current) => ({ ...current, name: value }))} /><Field label="科室 *" value={profile.department} onChange={(value) => setProfile((current) => ({ ...current, department: value }))} /><button className="btn-primary w-full" disabled={busy} onClick={() => void save()}>保存个人资料</button></section><section className="card-base p-8 space-y-4"><h2 className="text-card-title">密码管理</h2><Field label="当前密码" type="password" value={pwd.old} onChange={(value) => setPwd((current) => ({ ...current, old: value }))} /><Field label="新密码" type="password" value={pwd.next} onChange={(value) => setPwd((current) => ({ ...current, next: value }))} /><Field label="确认新密码" type="password" value={pwd.confirm} onChange={(value) => setPwd((current) => ({ ...current, confirm: value }))} /><button className="btn-primary w-full" disabled={busy} onClick={() => void changePassword()}>修改密码</button></section></div><button className="btn-secondary" onClick={() => navigate("/dashboard")}>返回工作台</button></div></main></div>;
}

function ReadOnly({ label, value }: { label: string; value: string }) { return <div><label className="block text-body font-semibold mb-2">{label}</label><div className="input-base bg-[color:var(--color-neutral)]">{value}</div></div>; }
function Field({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (value: string) => void; type?: string }) { return <label className="block text-body font-semibold">{label}<input className="input-base mt-2 font-normal" type={type} value={value} onChange={(event) => onChange(event.target.value)} /></label>; }
