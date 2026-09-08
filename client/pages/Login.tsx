import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import api from "@/lib/api";
import LocaleSwitcher from "@/components/LocaleSwitcher";
import logoUrl from "@/assets/logo.png";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";

const REMEMBER_KEY = "ais_remembered_login";

export default function Login() {
  const navigate = useNavigate();
  const { t } = useTranslation();
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
    if (!username.trim() || !password) return setError(t("login.requireCredentials"));
    setLoading(true); setError("");
    try {
      const data = await api.login(username.trim(), password);
      sessionStorage.setItem("user_token", data.token);
      sessionStorage.setItem("user_role", data.user.role);
      sessionStorage.setItem("user_name", data.user.name);
      sessionStorage.setItem("user_department", data.user.department || "");
      sessionStorage.setItem("user_id", data.user.id);
      if (remember) {
        localStorage.setItem(REMEMBER_KEY, JSON.stringify({ username: username.trim(), password }));
      } else {
        localStorage.removeItem(REMEMBER_KEY);
      }
      navigate("/dashboard");
    } catch (caught) { setError(caught instanceof Error ? caught.message : t("login.loginFailed")); }
    finally { setLoading(false); }
  };

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background p-5">
      {/* 品牌蓝色装饰光晕（纯装饰） */}
      <div aria-hidden className="pointer-events-none absolute -top-32 -left-32 h-96 w-96 rounded-full bg-primary/10 blur-3xl" />
      <div aria-hidden className="pointer-events-none absolute -right-32 -bottom-32 h-96 w-96 rounded-full bg-blue-400/10 blur-3xl" />

      <div className="absolute top-4 right-4 z-10">
        <LocaleSwitcher />
      </div>

      <Card className="relative z-10 w-full max-w-[420px] border-border/80 bg-card/90 shadow-xl backdrop-blur">
        <CardHeader className="items-center pb-2 pt-8 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center overflow-hidden rounded-2xl border border-border bg-white shadow-sm">
            <img src={logoUrl} alt="AIS" className="h-full w-full object-cover" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-card-foreground">{t("login.title")}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{t("login.subtitle")}</p>
        </CardHeader>

        <form onSubmit={login} className="contents">
          <CardContent className="space-y-4 px-8 pt-4">
            <div className="space-y-2">
              <Label htmlFor="login-username">{t("login.username")}</Label>
              <Input id="login-username" autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} disabled={loading} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="login-password">{t("login.password")}</Label>
              <Input id="login-password" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} disabled={loading} />
            </div>
            <div className="flex items-center gap-2 pt-1">
              <Checkbox id="login-remember" checked={remember} onCheckedChange={(checked) => setRemember(checked === true)} />
              <Label htmlFor="login-remember" className="cursor-pointer select-none font-medium text-foreground">{t("login.remember")}</Label>
            </div>
            {error && <p role="alert" className="text-sm font-medium text-destructive">{error}</p>}
            <Button type="submit" size="lg" className="w-full" disabled={loading}>
              {loading ? t("login.loggingIn") : t("login.login")}
            </Button>
          </CardContent>
        </form>

        <CardFooter className="justify-center px-8 pb-8 pt-4">
          <p className="text-xs text-muted-foreground">{t("login.forgot")}</p>
        </CardFooter>
      </Card>
    </main>
  );
}
