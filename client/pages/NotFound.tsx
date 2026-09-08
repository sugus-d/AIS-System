import { useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname,
    );
  }, [location.pathname]);

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background p-5">
      <div aria-hidden className="pointer-events-none absolute -top-24 -left-24 h-80 w-80 rounded-full bg-primary/10 blur-3xl" />
      <div className="text-center">
        <p className="text-[88px] font-bold leading-none tracking-tight text-primary/90">404</p>
        <h1 className="mt-4 text-xl font-semibold text-foreground">Oops! Page not found</h1>
        <p className="mt-1 text-sm text-muted-foreground">The page you are looking for does not exist.</p>
        <Button className="mt-6" onClick={() => navigate("/")}>Return to Home</Button>
      </div>
    </div>
  );
};

export default NotFound;
