-- 报告 ↔ X 光 多对多关联：X 光是受检者的附加数据，可被多份报告引用。
CREATE TABLE "ReportXray" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "reportId" TEXT NOT NULL,
    "xrayId" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ReportXray_reportId_fkey" FOREIGN KEY ("reportId") REFERENCES "Report" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "ReportXray_xrayId_fkey" FOREIGN KEY ("xrayId") REFERENCES "ScanFile" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE UNIQUE INDEX "ReportXray_reportId_xrayId_key" ON "ReportXray"("reportId", "xrayId");
