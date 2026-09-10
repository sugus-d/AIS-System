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
    rememberLogin: {
      get: () => Promise<{ username: string; password: string } | null>;
      set: (value: { username: string; password: string }) => Promise<boolean>;
      clear: () => Promise<boolean>;
    };
  };
}
