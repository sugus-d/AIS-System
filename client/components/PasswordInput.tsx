import { useState, type ChangeEvent } from "react";
import { Eye, EyeOff } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type Props = {
  value: string;
  onChange: (value: string) => void;
  /** 使用 shadcn Input 样式（登录页 / 个人设置）还是页面自定义类（用户管理） */
  variant?: "ui" | "plain";
  className?: string;
  wrapperClassName?: string;
  id?: string;
  autoComplete?: string;
  disabled?: boolean;
  placeholder?: string;
};

/** 带“小眼睛”显示/隐藏切换的密码输入框 */
export default function PasswordInput({
  value,
  onChange,
  variant = "plain",
  className,
  wrapperClassName,
  id,
  autoComplete,
  disabled,
  placeholder,
}: Props) {
  const { t } = useTranslation();
  const [visible, setVisible] = useState(false);
  const inputType = visible ? "text" : "password";
  const inputProps = {
    id,
    type: inputType,
    value,
    autoComplete,
    disabled,
    placeholder,
    onChange: (event: ChangeEvent<HTMLInputElement>) => onChange(event.target.value),
  };

  return (
    <span className={cn("relative block", wrapperClassName)}>
      {variant === "ui" ? (
        <Input className={cn("pr-10", className)} {...inputProps} />
      ) : (
        <input className={cn("pr-10", className)} {...inputProps} />
      )}
      <button
        type="button"
        tabIndex={-1}
        className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none"
        aria-label={visible ? t("common.hidePassword") : t("common.showPassword")}
        title={visible ? t("common.hidePassword") : t("common.showPassword")}
        onClick={() => setVisible((prev) => !prev)}
      >
        {visible ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </span>
  );
}
