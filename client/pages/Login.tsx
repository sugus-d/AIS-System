import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "@/lib/api";

const REMEMBER_KEY = "ais_remembered_login";

export default function Login() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(REMEMBER_KEY);
      if (!saved) return;
      const parsed = JSON.parse(saved);
      if (parsed && typeof parsed.username === "string" && parsed.username) {
        setUsername(parsed.username);
        setPassword(typeof parsed.password === "string" ? parsed.password : "");
        setRemember(true);
      }
    } catch {
      localStorage.removeItem(REMEMBER_KEY);
    }
  }, []);

  const login = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!username.trim() || !password) return setError("请输入账号和密码。");
    setLoading(true); setError("");
    try {
      const data = await api.login(username.trim(), password);
      localStorage.setItem("user_token", data.token);
      localStorage.setItem("user_role", data.user.role);
      localStorage.setItem("user_name", data.user.name);
      localStorage.setItem("user_department", data.user.department || "");
      localStorage.setItem("user_id", data.user.id);
      if (remember) {
        localStorage.setItem(REMEMBER_KEY, JSON.stringify({ username: username.trim(), password }));
      } else {
        localStorage.removeItem(REMEMBER_KEY);
      }
      navigate("/dashboard");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "登录失败。"); }
    finally { setLoading(false); }
  };
  return (
    <main className="min-h-screen bg-[#f5f7fa] flex items-center justify-center p-5">
      <section className="w-full max-w-[440px] card-base p-8 sm:p-10">
        <div className="text-center mb-8">
          <h1 className="text-page-title">AIS 筛查系统</h1>
          <p className="text-helper mt-2">本地脊柱侧弯筛查工作站</p>
        </div>
        <form className="space-y-4" onSubmit={login}>
          <label className="block text-body font-semibold">账号
            <input className="input-base mt-2" autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} />
          </label>
          <label className="block text-body font-semibold">密码
            <input className="input-base mt-2" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          <label className="flex items-center gap-2 text-body font-semibold cursor-pointer select-none">
            <input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} className="accent-[color:var(--color-primary)]" />
            记住密码
          </label>
          {error && <p className="text-helper text-[color:var(--color-error)]">{error}</p>}
          <button className="btn-primary w-full mt-3" disabled={loading} type="submit">{loading ? "登录中..." : "登录"}</button>
        </form>
        <p className="text-helper text-center mt-7">忘记密码请联系本机系统管理员重置。</p>
      </section>
    </main>
  );
}
