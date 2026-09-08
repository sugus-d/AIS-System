-- 文件级身份：上传 PLY 新建报告时记录操作者（科室/医生），供待分析阶段列表展示
ALTER TABLE "ScanFile" ADD COLUMN "department" TEXT;
ALTER TABLE "ScanFile" ADD COLUMN "doctor" TEXT;
