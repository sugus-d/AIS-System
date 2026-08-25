import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("ais", {
  getAppVersion: () => ipcRenderer.invoke("app:version"),
  exportDiagnostics: () => ipcRenderer.invoke("diagnostics:export"),
});
