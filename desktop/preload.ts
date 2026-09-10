import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("ais", {
  getAppVersion: () => ipcRenderer.invoke("app:version"),
  exportDiagnostics: () => ipcRenderer.invoke("diagnostics:export"),
  saveReportPdf: (html: string, suggestedName: string, locale?: string) => ipcRenderer.invoke("report:save-pdf", html, suggestedName, locale),
  rememberLogin: {
    get: () => ipcRenderer.invoke("login:remember:get"),
    set: (value: { username: string; password: string }) => ipcRenderer.invoke("login:remember:set", value),
    clear: () => ipcRenderer.invoke("login:remember:clear"),
  },
});
