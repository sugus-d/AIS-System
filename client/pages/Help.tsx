import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";

export default function Help() {
  const isAdmin = ["admin", "system_admin", "institution_admin"].includes(localStorage.getItem("user_role") || "");
  return <div className="layout-main"><Sidebar isAdmin={isAdmin}/><div className="layout-header"><Header isAdmin={isAdmin}/></div><div className="layout-content"><div className="content-wrapper space-y-6"><div><h1 className="text-page-title">帮助中心</h1><p className="text-body text-[color:var(--color-text-secondary)]">查看系统使用说明和常见问题。</p></div><section className="card-base p-8 space-y-5"><HelpItem title="如何新建受检者？" text="进入受检者管理，点击新建受检者，填写基本信息并保存。"/><HelpItem title="如何开始分析？" text="在受检者详情页选择已有筛查文件，然后点击开始分析。"/><HelpItem title="如何审核报告？" text="打开具体报告详情，在报告右上角执行审核通过或审核不通过。"/></section></div></div></div>;
}
function HelpItem({ title, text }: { title: string; text: string }) { return <div className="border-b border-[color:var(--color-border)] pb-5 last:border-0 last:pb-0"><h2 className="text-card-title mb-2">{title}</h2><p className="text-body text-[color:var(--color-text-secondary)]">{text}</p></div>; }
