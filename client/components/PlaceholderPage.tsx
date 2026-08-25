import { useNavigate } from "react-router-dom";
import Header from "./layout/Header";
import Sidebar from "./layout/Sidebar";

export default function PlaceholderPage({
  title,
  description,
  icon,
  isAdmin,
}: {
  title: string;
  description: string;
  icon: string;
  isAdmin: boolean;
}) {
  const navigate = useNavigate();

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header">
        <Header isAdmin={isAdmin} />
      </div>

      <div className="layout-content">
        <div className="content-wrapper">
          <div className="flex flex-col items-center justify-center min-h-[600px]">
            <div className="text-center">
              <div className="text-9xl mb-6">{icon}</div>
              <h1 className="text-page-title text-[color:var(--color-text-primary)] mb-4">
                {title}
              </h1>
              <p className="text-body text-[color:var(--color-text-secondary)] mb-8 max-w-md">
                {description}
              </p>

              <div className="card-base p-8 bg-blue-50 border-blue-200 max-w-md">
                <p className="text-body text-[color:var(--color-text-secondary)] mb-4">
                  此页面的功能模块正在开发中，您可以继续在其他功能区域进行操作。
                </p>
                <button
                  onClick={() => navigate("/dashboard")}
                  className="btn-primary w-full"
                >
                  返回工作台
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
