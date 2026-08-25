import "dotenv/config";
import express from "express";
import cors from "cors";

// Routes
import authRoutes from "./routes/auth";
import usersRoutes from "./routes/users";
import casesRoutes from "./routes/cases";
import filesRoutes from "./routes/files";
import analysisRoutes from "./routes/analysis";
import reportsRoutes from "./routes/reports";
import tasksRoutes from "./routes/tasks";
import statisticsRoutes from "./routes/statistics";
import settingsRoutes from "./routes/settings";
import profileRoutes from "./routes/profile";
import helpRoutes from "./routes/help";
import reviewsRoutes from "./routes/reviews";
import annotationRoutes from "./routes/annotation";
import backupRoutes from "./routes/backups";
import { authenticate } from './middleware/access';
import { ensureInitialAdmin } from "./services/database";

const PORT = Number(process.env.PORT) || 8080;
const HOST = process.env.HOST || "127.0.0.1";

export function createServer() {
  void ensureInitialAdmin().catch((error) => console.error("[AIS] 数据库初始化失败:", error));
  const app = express();
  const BASE_PATH = '/api';

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  app.get(`${BASE_PATH}/ping`, (_req, res) => {
    res.json({ message: "pong", timestamp: new Date().toISOString() });
  });

  // Auth routes
  app.use(`${BASE_PATH}/auth`, authRoutes);
  app.use(`${BASE_PATH}`, authenticate);

  // User management (admin)
  app.use(`${BASE_PATH}/users`, usersRoutes);

  // Cases management
  app.use(`${BASE_PATH}/cases`, casesRoutes);

  // Files management
  app.use(`${BASE_PATH}/files`, filesRoutes);

  // Analysis
  app.use(`${BASE_PATH}/analysis`, analysisRoutes);

  // Reports
  app.use(`${BASE_PATH}/reports`, reportsRoutes);

  // Tasks
  app.use(`${BASE_PATH}/tasks`, tasksRoutes);

  // Statistics
  app.use(`${BASE_PATH}/statistics`, statisticsRoutes);

  // Settings (admin)
  app.use(`${BASE_PATH}/settings`, settingsRoutes);

  // Profile
  app.use(`${BASE_PATH}/profile`, profileRoutes);

  // Help
  app.use(`${BASE_PATH}/help`, helpRoutes);
  app.use(`${BASE_PATH}/reviews`, reviewsRoutes);
  app.use(`${BASE_PATH}/annotation`, annotationRoutes);
  app.use(`${BASE_PATH}/backups`, backupRoutes);

  // 404 handler (API only)
  // In dev, this Express app is mounted as Vite middleware.
  // Do not intercept non-API requests, let Vite/Express handle SPA routes.
  app.use((req, res, next) => {
    if (req.path.startsWith(BASE_PATH)) {
      return res.status(404).json({ success: false, message: "API not found" });
    }
    return next();
  });

  return app;
}

// 启动服务器
if (import.meta.url === `file://${process.argv[1]}`) {
  const app = createServer();
  app.listen(PORT, HOST, () => {
    console.log(`AIS local API server running on http://${HOST}:${PORT}/api`);
  });
}
