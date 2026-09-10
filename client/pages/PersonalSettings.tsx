import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import LocaleSwitcher from "@/components/LocaleSwitcher";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import PasswordInput from "@/components/PasswordInput";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import api from "@/lib/api";

const roles: Record<string, string> = { system_admin: "enums.roleSystemAdmin", institution_admin: "enums.roleInstitutionAdmin", operator: "enums.roleOperator" };

export default function PersonalSettings() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const role = sessionStorage.getItem("user_role") || "operator";
  const [profile, setProfile] = useState({ username: "", name: "", department: "" });
  const [pwd, setPwd] = useState({ old: "", next: "", confirm: "" });
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const isAdmin = role === "system_admin" || role === "institution_admin";

  useEffect(() => { void api.getProfile().then((data) => setProfile({ username: data.username || "", name: data.name || "", department: data.department || "" })).catch((caught) => setError(caught instanceof Error ? caught.message : t("settings.loadFailed"))); }, []);
  const save = async () => {
    if (!profile.name.trim() || !profile.department.trim()) return setError(t("settings.required"));
    setBusy(true); setError("");
    try { const data = await api.updateProfile({ name: profile.name.trim(), department: profile.department.trim() }); sessionStorage.setItem("user_name", data.name); sessionStorage.setItem("user_department", data.department || ""); setMessage(t("settings.saved")); }
    catch (caught) { setError(caught instanceof Error ? caught.message : t("settings.saveFailed")); }
    finally { setBusy(false); }
  };
  const changePassword = async () => {
    if (!pwd.old || !pwd.next || !pwd.confirm) return setError(t("settings.pwdRequired"));
    if (pwd.next.length < 12) return setError(t("settings.pwdMinLength"));
    if (pwd.next !== pwd.confirm) return setError(t("settings.pwdMismatch"));
    setBusy(true); setError("");
    try { await api.changePassword(pwd.old, pwd.next); setPwd({ old: "", next: "", confirm: "" }); setMessage(t("settings.pwdUpdated")); }
    catch (caught) { setError(caught instanceof Error ? caught.message : t("settings.pwdChangeFailed")); }
    finally { setBusy(false); }
  };

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header"><Header isAdmin={isAdmin} /></div>
      <main className="layout-content">
        <div className="content-wrapper space-y-6">
          <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("settings.title")}</h1>
          {message && <div className="rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-sm text-primary">{message}</div>}
          {error && <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">{error}</div>}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card className="border-border/80">
              <CardHeader className="border-b border-border/60 px-8 py-6">
                <CardTitle className="text-lg text-foreground">{t("settings.profile")}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5 px-8 py-6">
                <ReadOnly label={t("settings.account")} value={profile.username || "--"} />
                <ReadOnly label={t("settings.role")} value={roles[role] ? t(roles[role]) : t("enums.roleOperator")} />
                <Field label={`${t("settings.name")} *`} value={profile.name} onChange={(value) => setProfile((current) => ({ ...current, name: value }))} />
                <Field label={`${t("settings.department")} *`} value={profile.department} onChange={(value) => setProfile((current) => ({ ...current, department: value }))} />
                <Button className="w-full" disabled={busy} onClick={() => void save()}>{t("settings.saveProfile")}</Button>
              </CardContent>
            </Card>

            <Card className="border-border/80">
              <CardHeader className="border-b border-border/60 px-8 py-6">
                <CardTitle className="text-lg text-foreground">{t("settings.password")}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5 px-8 py-6">
                <Field label={t("settings.currentPwd")} type="password" value={pwd.old} onChange={(value) => setPwd((current) => ({ ...current, old: value }))} />
                <Field label={t("settings.newPwd")} type="password" value={pwd.next} onChange={(value) => setPwd((current) => ({ ...current, next: value }))} />
                <Field label={t("settings.confirmPwd")} type="password" value={pwd.confirm} onChange={(value) => setPwd((current) => ({ ...current, confirm: value }))} />
                <Button className="w-full" disabled={busy} onClick={() => void changePassword()}>{t("settings.changePwd")}</Button>
              </CardContent>
            </Card>
          </div>

          <Card className="border-border/80">
            <CardContent className="flex flex-col gap-4 px-8 py-6 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">{t("settings.language")}</h2>
                <p className="mt-1 text-sm text-muted-foreground">{t("settings.languageHint")}</p>
              </div>
              <LocaleSwitcher />
            </CardContent>
          </Card>

          <Button variant="outline" onClick={() => navigate("/dashboard")}>{t("settings.backToDashboard")}</Button>
        </div>
      </main>
    </div>
  );
}

function ReadOnly({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-2">
      <Label className="font-medium">{label}</Label>
      <div className={cn("flex h-10 w-full items-center rounded-md border border-input bg-muted/50 px-3 text-sm text-muted-foreground")}>{value}</div>
    </div>
  );
}

function Field({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (value: string) => void; type?: string }) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      {type === "password" ? (
        <PasswordInput variant="ui" value={value} onChange={onChange} />
      ) : (
        <Input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
      )}
    </div>
  );
}
