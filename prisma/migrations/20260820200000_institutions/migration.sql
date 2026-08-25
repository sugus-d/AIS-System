CREATE TABLE "Institution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "active" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);
CREATE UNIQUE INDEX "Institution_name_key" ON "Institution"("name");
CREATE UNIQUE INDEX "Institution_code_key" ON "Institution"("code");

ALTER TABLE "User" ADD COLUMN "institutionId" TEXT;
ALTER TABLE "Case" ADD COLUMN "institutionId" TEXT;

INSERT INTO "Institution" ("id", "name", "code", "active", "updatedAt")
VALUES ('local-default-institution', '本地默认机构', 'LOCAL-DEFAULT', true, CURRENT_TIMESTAMP);
UPDATE "User" SET "institutionId" = 'local-default-institution' WHERE "institutionId" IS NULL;
UPDATE "Case" SET "institutionId" = 'local-default-institution' WHERE "institutionId" IS NULL;
CREATE INDEX "User_institutionId_idx" ON "User"("institutionId");
CREATE INDEX "Case_institutionId_idx" ON "Case"("institutionId");
