-- Add patient identity / contact fields to Case.
ALTER TABLE "Case" ADD COLUMN "idNumber" TEXT;
ALTER TABLE "Case" ADD COLUMN "phone" TEXT;
