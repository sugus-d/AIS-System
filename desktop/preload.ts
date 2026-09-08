import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("ais", {
  getAppVersion: () => ipcRenderer.invoke("app:version"),
  exportDiagnostics: () => ipcRenderer.invoke("diagnostics:export"),
  saveReportPdf: (html: string, suggestedName: string, locale?: string) => ipcRenderer.invoke("report:save-pdf", html, suggestedName, locale),
});
