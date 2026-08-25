import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

// Pages
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import CaseRecord from "./pages/CaseRecord";
import Cases from "./pages/Cases";
import CaseDetail from "./pages/CaseDetail";
import Analysis from "./pages/Analysis";
import AnalysisReport from "./pages/AnalysisReport";
import Statistics from "./pages/Statistics";
import Help from "./pages/Help";
import PersonalSettings from "./pages/PersonalSettings";
import AdminUsers from "./pages/admin/Users";
import AdminSettings from "./pages/admin/Settings";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Auto-redirect based on auth status
const AuthRedirect = () => {
  const userRole = localStorage.getItem("user_role");
  
  if (!userRole) {
    return <Navigate to="/login" replace />;
  }
  
  return <Navigate to="/dashboard" replace />;
};

// Protected route wrapper
const ProtectedRoute = ({ element }: { element: React.ReactNode }) => {
  const userRole = localStorage.getItem("user_role");
  
  if (!userRole) {
    return <Navigate to="/login" replace />;
  }
  
  return element;
};

const RoleRoute = ({ roles, element }: { roles: string[]; element: React.ReactNode }) => {
  const role = localStorage.getItem("user_role");
  if (!role) return <Navigate to="/login" replace />;
  return roles.includes(role) ? element : <Navigate to="/dashboard" replace />;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Auth Routes */}
          <Route path="/login" element={<Login />} />

          {/* Protected Routes */}
          <Route path="/dashboard" element={<ProtectedRoute element={<Dashboard />} />} />
          <Route path="/case-record" element={<ProtectedRoute element={<CaseRecord />} />} />
          <Route path="/cases" element={<ProtectedRoute element={<Cases />} />} />
          <Route path="/case-detail/:caseId" element={<ProtectedRoute element={<CaseDetail />} />} />
          <Route path="/analysis" element={<ProtectedRoute element={<Analysis />} />} />
          <Route path="/analysis-report" element={<ProtectedRoute element={<AnalysisReport />} />} />
          <Route path="/statistics" element={<ProtectedRoute element={<Statistics />} />} />
          <Route path="/help" element={<ProtectedRoute element={<Help />} />} />
          <Route path="/settings" element={<ProtectedRoute element={<PersonalSettings />} />} />

          {/* Admin Routes */}
          <Route path="/admin/users" element={<RoleRoute roles={["system_admin", "institution_admin"]} element={<AdminUsers />} />} />
          <Route path="/admin/settings" element={<RoleRoute roles={["system_admin"]} element={<AdminSettings />} />} />

          {/* Default Routes - Auto redirect based on auth status */}
          <Route path="/" element={<AuthRedirect />} />

          {/* Catch-all */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
