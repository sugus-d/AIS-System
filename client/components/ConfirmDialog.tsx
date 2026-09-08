import { useSyncExternalStore } from "react";
import { useTranslation } from "react-i18next";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

type Options = { title?: string; confirmText?: string; destructive?: boolean };
type Request = { message: string; options?: Options; resolve: (ok: boolean) => void };

let store: { request: Request | null } = { request: null };
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((listener) => listener());
const subscribe = (listener: () => void) => { listeners.add(listener); return () => listeners.delete(listener); };
const getSnapshot = () => store;

/** 阻塞式确认（语义等价 window.confirm）：返回用户点“取消”或“确定”。 */
export function confirmDialog(message: string, options?: Options): Promise<boolean> {
  return new Promise<boolean>((resolve) => { store = { request: { message, options, resolve } }; emit(); });
}

/** 在应用根渲染一次；提供 AlertDialog 界面。 */
export function ConfirmHost() {
  const { t } = useTranslation();
  const { request } = useSyncExternalStore(subscribe, getSnapshot);
  const settle = (ok: boolean) => { const active = store.request; if (!active) return; active.resolve(ok); store = { request: null }; emit(); };
  return (
    <AlertDialog open={!!request} onOpenChange={(open) => { if (!open) settle(false); }}>
      <AlertDialogContent>
        <AlertDialogHeader>
          {request?.options?.title ? (
            <AlertDialogTitle>{request.options.title}</AlertDialogTitle>
          ) : (
            <AlertDialogTitle className="sr-only">{request?.options?.confirmText ?? t("common.confirm")}</AlertDialogTitle>
          )}
          <AlertDialogDescription className="whitespace-pre-wrap">{request?.message ?? ""}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => settle(false)}>{t("common.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => settle(true)}
            className={request?.options?.destructive ? "bg-destructive text-destructive-foreground hover:bg-destructive/90" : undefined}
          >
            {request?.options?.confirmText ?? t("common.confirm")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
