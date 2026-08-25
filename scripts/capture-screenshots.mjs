import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

const baseUrl = "http://192.168.1.6:8080";
const outputDir = path.resolve("docs/screenshots");

const routes = [
    { role: null, route: "/ais/login", name: "page-login.png" },
    { role: "staff", route: "/ais/dashboard", name: "page-dashboard.png" },
    { role: "staff", route: "/ais/cases", name: "page-cases.png" },
    { role: "staff", route: "/ais/case-detail/1", name: "page-case-detail.png" },
    { role: "staff", route: "/ais/case-record", name: "page-case-record.png" },
    { role: "staff", route: "/ais/batch-case-record", name: "page-batch-case-record.png" },
    { role: "staff", route: "/ais/file-upload", name: "page-file-upload.png" },
    { role: "staff", route: "/ais/analysis?id=1", name: "page-analysis.png" },
    { role: "staff", route: "/ais/tasks", name: "page-tasks.png" },
    { role: "staff", route: "/ais/reports", name: "page-reports.png" },
    { role: "staff", route: "/ais/analysis-report?reportId=RPT20260317001", name: "page-analysis-report.png" },
    { role: "staff", route: "/ais/statistics", name: "page-statistics.png" },
    { role: "staff", route: "/ais/help", name: "page-help.png" },
    { role: "staff", route: "/ais/settings", name: "page-personal-settings.png" },
    { role: "admin", route: "/ais/admin/settings", name: "page-admin-settings.png" },
    { role: "admin", route: "/ais/admin/users", name: "page-admin-users.png" },
    { role: "admin", route: "/ais/admin/api-config", name: "page-admin-api-config.png" },
];

function userNameByRole(role) {
    if (role === "admin") return "管理员";
    if (role === "staff") return "李医生";
    return "访客";
}

async function captureRoute(browser, item) {
    const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });

    if (item.role) {
        await context.addInitScript(
            ({ role, userName }) => {
                localStorage.setItem("user_role", role);
                localStorage.setItem("user_name", userName);
            },
            { role: item.role === "staff" ? "user" : "admin", userName: userNameByRole(item.role) },
        );
    } else {
        await context.addInitScript(() => {
            localStorage.removeItem("user_role");
            localStorage.removeItem("user_name");
        });
    }

    const page = await context.newPage();
    const url = `${baseUrl}${item.route}`;
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: path.join(outputDir, item.name), fullPage: true });

    await context.close();
}

async function main() {
    await fs.mkdir(outputDir, { recursive: true });
    const browser = await chromium.launch({ headless: true });

    for (const item of routes) {
        try {
            await captureRoute(browser, item);
            console.log(`Captured: ${item.name}`);
        } catch (error) {
            console.error(`Failed: ${item.name}`, error);
        }
    }

    await browser.close();
}

main();
