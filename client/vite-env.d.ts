/// <reference types="vite/client" />

interface Window {
  ais?: {
    getAppVersion: () => Promise<string>;
    exportDiagnostics: () => Promise<unknown>;
    saveReportPdf: (
      html: string,
      suggestedName: string,
      locale?: string,
    ) => Promise<{ ok: boolean; canceled?: boolean; filePath?: string } | undefined>;
  };
}
