import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import { Card, CardContent } from "@/components/ui/card";

export default function Help() {
  const { t } = useTranslation();
  const isAdmin = ["admin", "system_admin", "institution_admin"].includes(sessionStorage.getItem("user_role") || "");
  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header"><Header isAdmin={isAdmin} /></div>
      <div className="layout-content">
        <div className="content-wrapper space-y-6">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("help.title")}</h1>
            <p className="mt-2 text-muted-foreground">{t("help.subtitle")}</p>
          </div>
          <Card className="border-border/80">
            <CardContent className="divide-y divide-border/60 px-0 py-0">
              <HelpItem title={t("help.q1")} text={t("help.a1")} />
              <HelpItem title={t("help.q2")} text={t("help.a2")} />
              <HelpItem title={t("help.q3")} text={t("help.a3")} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function HelpItem({ title, text }: { title: string; text: string }) {
  return (
    <div className="px-6 py-5">
      <h2 className="mb-1.5 font-semibold text-card-foreground">{title}</h2>
      <p className="text-sm leading-6 text-muted-foreground">{text}</p>
    </div>
  );
}
