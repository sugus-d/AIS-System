import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import api from "@/lib/api";

type FormData = { name: string; gender: "" | "male" | "female"; birthday: string; height: string; weight: string; idNumber: string; phone: string; medicalHistory: string };
const empty = (): FormData => ({ name: "", gender: "", birthday: "", height: "", weight: "", idNumber: "", phone: "", medicalHistory: "" });
const inputCls = "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50";

export default function CaseRecord() {
  const navigate = useNavigate(); const { t } = useTranslation(); const [params] = useSearchParams(); const caseId = params.get("caseId"); const isEdit = Boolean(caseId);
  const isAdmin = sessionStorage.getItem("user_role") === "admin";
  const isSystemAdmin = ["system_admin", "admin"].includes(sessionStorage.getItem("user_role") || "");
  const [form, setForm] = useState<FormData>(empty()); const [loading, setLoading] = useState(false);
  const [institutions, setInstitutions] = useState<{ id: string; name: string }[]>([]);
  const [institutionId, setInstitutionId] = useState("");

  useEffect(() => { if (isSystemAdmin) api.getInstitutions().then(setInstitutions).catch(() => undefined); }, [isSystemAdmin]);

  useEffect(() => { if (!caseId) return; api.getCase(caseId).then((r: any) => { const c = r.case || r; setForm({ name: c.name || "", gender: c.gender === "female" || c.gender === "女" ? "female" : c.gender ? "male" : "", birthday: c.birthDate ? String(c.birthDate).slice(0, 10) : "", height: String(c.height || ""), weight: String(c.weight || ""), idNumber: c.idNumber || "", phone: c.phone || "", medicalHistory: c.medicalHistory || "" }); }); }, [caseId]);

  const change = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => setForm({ ...form, [e.target.name]: e.target.value });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name || !form.gender || !form.birthday || !form.height || !form.weight) { toast.error(t("caseRecord.validation")); return; }
    if (isSystemAdmin && !isEdit) {
      if (institutions.length === 0) { toast.error(t("caseRecord.noInstitution")); return; }
      if (institutions.length > 1 && !institutionId) { toast.error(t("caseRecord.institutionRequired")); return; }
    }
    try {
      setLoading(true);
      const data: any = { name: form.name, gender: form.gender, birthDate: form.birthday, height: Number(form.height), weight: Number(form.weight), idNumber: form.idNumber, phone: form.phone, medicalHistory: form.medicalHistory };
      if (isSystemAdmin && !isEdit && institutionId) data.institutionId = institutionId;
      if (isEdit && caseId) await api.updateCase(caseId, data); else await api.createCase(data);
      navigate(isEdit && caseId ? `/case-detail/${caseId}` : "/cases");
    } catch (err: any) { toast.error(err?.message || t("caseRecord.saveFailed")); } finally { setLoading(false); }
  };

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header"><Header isAdmin={isAdmin} /></div>
      <div className="layout-content">
        <div className="content-wrapper">
          <div className="mb-8 flex items-center justify-between">
            <h1 className="text-2xl font-semibold tracking-tight text-foreground">{isEdit ? t("caseRecord.editTitle") : t("caseRecord.newTitle")}</h1>
            <Button type="button" variant="outline" onClick={() => navigate(isEdit && caseId ? `/case-detail/${caseId}` : "/cases")}>{t("caseRecord.back")}</Button>
          </div>

          <Card className="mx-auto max-w-4xl border-border/80">
            <form onSubmit={submit}>
              <CardHeader className="border-b border-border/60 px-8 py-6">
                <CardTitle className="text-lg text-foreground">
                  {t("caseRecord.basicInfo")}
                  <span className="ml-2 text-sm font-normal text-muted-foreground">{t("caseRecord.requiredHint")}</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-1 gap-x-6 gap-y-5 px-8 py-6 md:grid-cols-2">
                <Field label={t("caseRecord.name")} name="name" value={form.name} onChange={change} required />
                <div className="space-y-2">
                  <Label htmlFor="gender">{t("caseRecord.gender")}<span className="ml-0.5 text-destructive">*</span></Label>
                  <select id="gender" className={inputCls} name="gender" value={form.gender} onChange={change} required>
                    <option value="">{t("caseRecord.selectGender")}</option>
                    <option value="male">{t("caseRecord.genderMale")}</option>
                    <option value="female">{t("caseRecord.genderFemale")}</option>
                  </select>
                </div>
                <Field label={t("caseRecord.birthday")} name="birthday" type="date" value={form.birthday} onChange={change} required />
                <Field label={t("caseRecord.height")} name="height" type="number" value={form.height} onChange={change} required />
                <Field label={t("caseRecord.weight")} name="weight" type="number" value={form.weight} onChange={change} required />
                {isSystemAdmin && !isEdit && institutions.length === 0 && (
                  <p className="text-sm text-destructive md:col-span-2">{t("caseRecord.noInstitution")}</p>
                )}
                {isSystemAdmin && !isEdit && institutions.length > 1 && (
                  <div className="space-y-2">
                    <Label htmlFor="institutionId">{t("caseRecord.institution")}<span className="ml-0.5 text-destructive">*</span></Label>
                    <select id="institutionId" className={inputCls} value={institutionId} onChange={(e) => setInstitutionId(e.target.value)}>
                      <option value="">{t("caseRecord.selectInstitution")}</option>
                      {institutions.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
                    </select>
                  </div>
                )}
                <Field label={t("caseRecord.idNumber")} name="idNumber" value={form.idNumber} onChange={change} />
                <Field label={t("caseRecord.phone")} name="phone" value={form.phone} onChange={change} />
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="medicalHistory">{t("caseRecord.medicalHistory")}</Label>
                  <Textarea id="medicalHistory" name="medicalHistory" rows={3} value={form.medicalHistory} onChange={change} />
                </div>
              </CardContent>
              <div className="flex justify-end gap-3 border-t border-border/60 px-8 py-5">
                <Button type="button" variant="outline" onClick={() => navigate(isEdit && caseId ? `/case-detail/${caseId}` : "/cases")}>{t("caseRecord.cancel")}</Button>
                <Button type="submit" disabled={loading || (isSystemAdmin && !isEdit && institutions.length === 0)}>{loading ? t("caseRecord.saving") : t("caseRecord.save")}</Button>
              </div>
            </form>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Field({ label, name, value, onChange, type = "text", required = false }: { label: string; name: string; value: string; onChange: (e: React.ChangeEvent<HTMLInputElement>) => void; type?: string; required?: boolean }) {
  return (
    <div className="space-y-2">
      <Label htmlFor={name}>{label}{required && <span className="ml-0.5 text-destructive">*</span>}</Label>
      <Input id={name} name={name} type={type} value={value} onChange={onChange} required={required} />
    </div>
  );
}
