import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import DataTable, { Column } from "@/components/DataTable";
import { confirmDialog } from "@/components/ConfirmDialog";
import api from "@/lib/api";

type UserRecord = {
  id: string;
  account: string;
  name: string;
  role: "system_admin" | "institution_admin" | "operator";
  department: string;
  institutionId: string | null;
  institutionName?: string;
  institutionAdmin?: string | null;
  createTime: string;
  lastLogin: string;
  status: "active" | "disabled";
};
type NewUser = {
  username: string;
  name: string;
  role: "system_admin" | "institution_admin" | "operator";
  department: string;
  institutionId: string;
  password: string;
};

const roleLabels: Record<UserRecord["role"], string> = {
  system_admin: "enums.roleSystemAdmin",
  institution_admin: "enums.roleInstitutionAdmin",
  operator: "enums.roleOperator",
};
type InstitutionOption = { id: string; name: string; admin: string | null };
const initialNewUser: NewUser = {
  username: "",
  name: "",
  role: "operator",
  department: "",
  institutionId: "",
  password: "",
};

const formatDate = (value?: string | null) => {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
    .format(date)
    .replaceAll("/", "-");
};

const toRecord = (user: any): UserRecord => ({
  id: user.id,
  account: user.username,
  name: user.name || "--",
  role: user.role,
  department: user.department || "--",
  institutionId: user.institutionId ?? null,
  institutionName: user.institutionName || undefined,
  institutionAdmin: user.institutionAdmin || null,
  createTime: formatDate(user.createdAt),
  lastLogin: formatDate(user.lastLogin),
  status: user.status === "disabled" ? "disabled" : "active",
});

export default function AdminUsers() {
  const { t } = useTranslation();
  const role = sessionStorage.getItem("user_role");
  const isAdmin =
    role === "admin" || role === "system_admin" || role === "institution_admin";
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [institutions, setInstitutions] = useState<InstitutionOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState("");
  const [actionId, setActionId] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [newUser, setNewUser] = useState<NewUser>(initialNewUser);
  const [createError, setCreateError] = useState("");
  const [editOpen, setEditOpen] = useState(false);
  const [editing, setEditing] = useState<UserRecord | null>(null);
  const [editForm, setEditForm] = useState<{ username: string; name: string; role: UserRecord["role"]; department: string; institutionId: string; password: string }>({ username: "", name: "", role: "operator", department: "", institutionId: "", password: "" });
  const [editError, setEditError] = useState("");

  const loadUsers = async () => {
    try {
      setLoading(true);
      setError("");
      const result = await api.getUsers({ pageSize: 100 });
      setUsers((result.list || result.data?.list || []).map(toRecord));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("users.loadFailed"));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isAdmin) void loadUsers();
  }, [isAdmin]);

  const runAction = async (
    id: string,
    action: "toggle" | "delete",
    account: string,
  ) => {
    if (action === "delete") {
      const confirmed = await confirmDialog(t("users.deleteConfirm", { name: account }), { destructive: true });
      if (!confirmed) return;
    }
    try {
      setActionId(id);
      setFeedback("");
      if (action === "toggle") {
        await api.toggleUserStatus(id);
        setFeedback(t("users.statusUpdated", { name: account }));
        await loadUsers();
      }
      if (action === "delete") {
        await api.deleteUser(id);
        setFeedback(t("users.deleted", { name: account }));
        await loadUsers();
      }
    } catch (caught) {
      setFeedback(
        caught instanceof Error ? caught.message : t("users.opFailed"),
      );
    } finally {
      setActionId(null);
    }
  };
  const createUser = async () => {
    if (!/^[\p{L}\p{N}._-]{2,32}$/u.test(newUser.username.trim())) {
      setCreateError(t("users.usernameInvalid"));
      return;
    }
    if (!newUser.name.trim()) {
      setCreateError(t("users.nameRequired"));
      return;
    }
    if (isSystemAdmin && newUser.role !== "system_admin" && !newUser.institutionId) {
      setCreateError(t("users.institutionRequired"));
      return;
    }
    try {
      setActionId("create");
      setCreateError("");
      const payload: any = { ...newUser, username: newUser.username.trim() };
      if (isSystemAdmin && newUser.role !== "system_admin") payload.institutionId = newUser.institutionId;
      const createdUser = await api.createUser(payload);
      setCreateOpen(false);
      setNewUser(initialNewUser);
      setFeedback(t("users.created", { name: createdUser.username }));
      await loadUsers();
    } catch (caught) {
      setCreateError(
        caught instanceof Error ? caught.message : t("users.createFailed"),
      );
    } finally {
      setActionId(null);
    }
  };

  const isSystemAdmin = role === "system_admin" || role === "admin";
  useEffect(() => {
    if (isSystemAdmin) api.getInstitutions().then(setInstitutions).catch(() => undefined);
  }, [isSystemAdmin]);
  const openEdit = (row: UserRecord) => {
    setEditError("");
    setEditing(row);
    setEditForm({
      username: row.account,
      name: row.name === "--" ? "" : row.name,
      role: row.role,
      department: row.department === "--" ? "" : row.department,
      institutionId: row.institutionId ?? "",
      password: "",
    });
    setEditOpen(true);
  };
  const saveEdit = async () => {
    if (!editing) return;
    const username = editForm.username.trim();
    if (!/^[\p{L}\p{N}._-]{2,32}$/u.test(username)) {
      setEditError(t("users.usernameInvalid"));
      return;
    }
    if (!editForm.name.trim()) {
      setEditError(t("users.nameRequired"));
      return;
    }
    if (isSystemAdmin && editForm.role !== "system_admin" && !editForm.institutionId) {
      setEditError(t("users.institutionRequired"));
      return;
    }
    if (editForm.password && editForm.password.length < 12) {
      setEditError(t("users.pwdMinLength"));
      return;
    }
    try {
      setActionId(editing.id);
      setEditError("");
      const payload: any = {
        username,
        name: editForm.name.trim(),
        department: editForm.department,
      };
      if (isSystemAdmin) {
        payload.role = editForm.role;
        if (editForm.role !== "system_admin") payload.institutionId = editForm.institutionId;
      }
      if (editForm.password) payload.password = editForm.password;
      await api.updateUser(editing.id, payload);
      setEditOpen(false);
      setEditing(null);
      setFeedback(t("users.updated", { name: username }));
      await loadUsers();
    } catch (caught) {
      setEditError(caught instanceof Error ? caught.message : t("users.saveFailed"));
    } finally {
      setActionId(null);
    }
  };

  if (!isAdmin) return <Navigate to="/dashboard" replace />;

  const columns: Column<UserRecord>[] = [
    {
      key: "account",
      label: t("users.colAccount"),
      width: "170px",
      sortable: true,
      render: (value, row) => (
        <div>
          <p className="font-semibold text-[color:var(--color-primary)]">
            {value}
          </p>
          <p className="text-sm text-muted-foreground mt-0.5">{row.name}</p>
        </div>
      ),
    },
    {
      key: "role",
      label: t("users.colRole"),
      width: "120px",
      render: (value) => roleLabels[value as UserRecord["role"]] ? t(roleLabels[value as UserRecord["role"]]) : value,
    },
    { key: "department", label: t("users.colDepartment"), width: "140px" },
    {
      key: "institutionName",
      label: t("users.colInstitution"),
      width: "200px",
      render: (_value, row) => {
        // 三级模型：系统管理员不归属机构；机构管理员显示所辖机构；临床操作员显示所属机构 + 由机构管理员管辖
        if (row.role === "system_admin") {
          return <p className="text-sm text-muted-foreground">--</p>;
        }
        const inst = row.institutionName || "--";
        return (
          <div>
            <p className="text-sm font-medium text-[color:var(--color-text-primary)]">{inst}</p>
            {row.role === "operator" && row.institutionAdmin ? (
              <p className="mt-0.5 text-xs text-muted-foreground">{t("users.governedBy", { admin: row.institutionAdmin })}</p>
            ) : null}
          </div>
        );
      },
    },
    { key: "createTime", label: t("users.colCreateTime"), width: "165px", sortable: true },
    { key: "lastLogin", label: t("users.colLastLogin"), width: "165px", sortable: true },
    {
      key: "status",
      label: t("users.colStatus"),
      width: "90px",
      align: "center",
      render: (value) => (
        <span className={value === "active" ? "tag-success" : "tag-warning"}>
          {value === "active" ? t("users.statusActive") : t("users.statusDisabled")}
        </span>
      ),
    },
    {
      key: "id",
      label: t("users.colOps"),
      width: "380px",
      align: "right",
      render: (_value, row) => (
        <div
          className="flex items-center justify-end gap-2 whitespace-nowrap"
          onClick={(event) => event.stopPropagation()}
        >
          {(isSystemAdmin || row.role === "operator") && (
            <button
              className="btn-secondary px-3 py-2"
              disabled={actionId === row.id}
              onClick={() => openEdit(row)}
            >
              {t("users.edit")}
            </button>
          )}
          {isSystemAdmin && (
            <button
              className="btn-secondary px-3 py-2"
              disabled={actionId === row.id}
              onClick={() => void runAction(row.id, "toggle", row.account)}
            >
              {row.status === "active" ? t("users.disable") : t("users.enable")}
            </button>
          )}
          {isSystemAdmin && (
            <button
              className="btn-danger px-3 py-2"
              disabled={actionId === row.id}
              onClick={() => void runAction(row.id, "delete", row.account)}
            >
              {t("users.delete")}
            </button>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="layout-main">
      <Sidebar isAdmin={isAdmin} />
      <div className="layout-header">
        <Header isAdmin={isAdmin} />
      </div>
      <main className="layout-content">
        <div className="content-wrapper space-y-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-sm text-muted-foreground uppercase tracking-wider text-[color:var(--color-primary)] mb-2">
                {t("users.eyebrow")}
              </p>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground">{t("users.title")}</h1>
              <p className="text-sm mt-2">
                {t("users.subtitle")}
              </p>
            </div>
            <button
              className="btn-primary"
              onClick={() => {
                setCreateError("");
                setCreateOpen(true);
              }}
            >
              {t("users.addUser")}
            </button>
          </div>
          {feedback && (
            <div className="border border-blue-200 bg-blue-50 px-4 py-3 rounded-card flex items-center justify-between gap-4">
              <p className="text-sm text-[color:var(--color-primary)]">
                {feedback}
              </p>
              <button className="btn-text" onClick={() => setFeedback("")}>
                {t("users.close")}
              </button>
            </div>
          )}
          {error ? (
            <section className="card-base p-8 text-center">
              <p className="text-sm text-[color:var(--color-error)] mb-4">
                {error}
              </p>
              <button
                className="btn-secondary"
                onClick={() => void loadUsers()}
              >
                {t("users.reload")}
              </button>
            </section>
          ) : (
            <>
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  {t("users.totalUsers", { count: users.length })}
                </p>
                <button
                  className="btn-text"
                  onClick={() => void loadUsers()}
                  disabled={loading}
                >
                  {t("users.refresh")}
                </button>
              </div>
              <DataTable<UserRecord>
                columns={columns}
                data={users}
                rowKey="id"
                pageSize={10}
                loading={loading}
                emptyMessage={t("users.empty")}
              />
            </>
          )}
        </div>
      </main>
      {createOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="create-user-title"
        >
          <div className="card-base w-full max-w-xl p-6 shadow-xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id="create-user-title" className="text-lg font-semibold text-foreground">
                  {t("users.createTitle")}
                </h2>
                <p className="text-sm text-muted-foreground mt-1">
                  {t("users.createHint")}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
              <Field
                label={t("users.usernameLabel")}
                value={newUser.username}
                onChange={(value) =>
                  setNewUser((prev) => ({ ...prev, username: value }))
                }
              />
              <Field
                label={t("users.name")}
                value={newUser.name}
                onChange={(value) =>
                  setNewUser((prev) => ({ ...prev, name: value }))
                }
              />
              <label className="block text-sm font-semibold">
                {t("users.role")}
                <select
                  className="input-base mt-2 font-normal"
                  value={newUser.role}
                  onChange={(event) =>
                    setNewUser((prev) => ({
                      ...prev,
                      role: event.target.value as NewUser["role"],
                      institutionId: event.target.value === "system_admin" ? "" : prev.institutionId,
                    }))
                  }
                >
                  {Object.entries(roleLabels)
                    .filter(([value]) => isSystemAdmin || value === "operator")
                    .map(([value, label]) => (
                      <option key={value} value={value}>
                        {t(label)}
                      </option>
                    ))}
                </select>
              </label>
              <Field
                label={t("users.department")}
                value={newUser.department}
                onChange={(value) =>
                  setNewUser((prev) => ({ ...prev, department: value }))
                }
              />
              {isSystemAdmin && newUser.role !== "system_admin" && (
                <label className="block text-sm font-semibold">
                  {t("users.chooseInstitution")}
                  <select
                    className="input-base mt-2 font-normal"
                    value={newUser.institutionId}
                    onChange={(event) =>
                      setNewUser((prev) => ({ ...prev, institutionId: event.target.value }))
                    }
                  >
                    <option value="">--</option>
                    {institutions.map((i) => (
                      <option key={i.id} value={i.id}>
                        {i.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {!isSystemAdmin && (
                <p className="text-sm text-muted-foreground md:col-span-2">
                  {t("users.instBoundHint")}
                </p>
              )}
              <div className="md:col-span-2">
                <Field
                  label={t("users.initialPwd")}
                  value={newUser.password}
                  onChange={(value) =>
                    setNewUser((prev) => ({ ...prev, password: value }))
                  }
                  type="password"
                />
              </div>
            </div>
            {createError && (
              <p className="text-sm text-muted-foreground text-[color:var(--color-error)] mt-4">
                {createError}
              </p>
            )}
            <div className="mt-6 flex justify-end gap-3">
              <button
                className="btn-secondary"
                disabled={actionId === "create"}
                onClick={() => setCreateOpen(false)}
              >
                {t("common.cancel")}
              </button>
              <button
                className="btn-primary"
                disabled={actionId === "create"}
                onClick={() => void createUser()}
              >
                {actionId === "create" ? t("users.creating") : t("users.create")}
              </button>
            </div>
          </div>
        </div>
      )}
      {editOpen && editing && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="edit-user-title"
        >
          <div className="card-base w-full max-w-xl p-6 shadow-xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id="edit-user-title" className="text-lg font-semibold text-foreground">
                  {t("users.editTitle")}
                </h2>
                <p className="text-sm text-muted-foreground mt-1">
                  {t("users.editHint")}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
              <Field
                label={t("users.usernameLabel")}
                value={editForm.username}
                onChange={(value) =>
                  setEditForm((prev) => ({ ...prev, username: value }))
                }
              />
              <Field
                label={t("users.name")}
                value={editForm.name}
                onChange={(value) =>
                  setEditForm((prev) => ({ ...prev, name: value }))
                }
              />
              <label className="block text-sm font-semibold">
                {t("users.role")}
                <select
                  className="input-base mt-2 font-normal"
                  value={editForm.role}
                  disabled={!isSystemAdmin}
                  onChange={(event) =>
                    setEditForm((prev) => ({
                      ...prev,
                      role: event.target.value as UserRecord["role"],
                      institutionId: event.target.value === "system_admin" ? "" : prev.institutionId,
                    }))
                  }
                >
                  {Object.entries(roleLabels)
                    .filter(([value]) => isSystemAdmin || value === "operator")
                    .map(([value, label]) => (
                      <option key={value} value={value}>
                        {t(label)}
                      </option>
                    ))}
                </select>
                {!isSystemAdmin && (
                  <p className="text-sm text-muted-foreground mt-1 font-normal">
                    {t("users.roleLockedHint")}
                  </p>
                )}
              </label>
              <Field
                label={t("users.department")}
                value={editForm.department}
                onChange={(value) =>
                  setEditForm((prev) => ({ ...prev, department: value }))
                }
              />
              {isSystemAdmin && editForm.role !== "system_admin" && (
                <label className="block text-sm font-semibold">
                  {t("users.chooseInstitution")}
                  <select
                    className="input-base mt-2 font-normal"
                    value={editForm.institutionId}
                    onChange={(event) =>
                      setEditForm((prev) => ({ ...prev, institutionId: event.target.value }))
                    }
                  >
                    <option value="">--</option>
                    {institutions.map((i) => (
                      <option key={i.id} value={i.id}>
                        {i.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="md:col-span-2">
                <Field
                  label={t("users.pwdLabel")}
                  value={editForm.password}
                  onChange={(value) =>
                    setEditForm((prev) => ({ ...prev, password: value }))
                  }
                  type="password"
                />
              </div>
            </div>
            {editError && (
              <p className="text-sm text-muted-foreground text-[color:var(--color-error)] mt-4">
                {editError}
              </p>
            )}
            <div className="mt-6 flex justify-end gap-3">
              <button
                className="btn-secondary"
                disabled={actionId === editing.id}
                onClick={() => {
                  setEditOpen(false);
                  setEditing(null);
                }}
              >
                {t("common.cancel")}
              </button>
              <button
                className="btn-primary"
                disabled={actionId === editing.id}
                onClick={() => void saveEdit()}
              >
                {actionId === editing.id ? t("users.saving") : t("users.save")}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
}) {
  return (
    <label className="block text-sm font-semibold">
      {label}
      <input
        className="input-base mt-2 font-normal"
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}
