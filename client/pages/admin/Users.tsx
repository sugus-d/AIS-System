import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import Header from "@/components/layout/Header";
import Sidebar from "@/components/layout/Sidebar";
import DataTable, { Column } from "@/components/DataTable";
import api from "@/lib/api";

type UserRecord = {
  id: string;
  account: string;
  name: string;
  role: "system_admin" | "institution_admin" | "operator";
  department: string;
  createTime: string;
  lastLogin: string;
  status: "active" | "disabled";
};
type NewUser = {
  username: string;
  name: string;
  role: "system_admin" | "institution_admin" | "operator";
  department: string;
  password: string;
};

const roleLabels: Record<UserRecord["role"], string> = {
  system_admin: "系统管理员",
  institution_admin: "机构管理员",
  operator: "临床操作员",
};
const initialNewUser: NewUser = {
  username: "",
  name: "",
  role: "operator",
  department: "",
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
  createTime: formatDate(user.createdAt),
  lastLogin: formatDate(user.lastLogin),
  status: user.status === "disabled" ? "disabled" : "active",
});

export default function AdminUsers() {
  const role = localStorage.getItem("user_role");
  const isAdmin =
    role === "admin" || role === "system_admin" || role === "institution_admin";
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState("");
  const [actionId, setActionId] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [newUser, setNewUser] = useState<NewUser>(initialNewUser);
  const [createError, setCreateError] = useState("");
  const [editOpen, setEditOpen] = useState(false);
  const [editing, setEditing] = useState<UserRecord | null>(null);
  const [editForm, setEditForm] = useState<{ username: string; name: string; role: UserRecord["role"]; department: string; password: string }>({ username: "", name: "", role: "operator", department: "", password: "" });
  const [editError, setEditError] = useState("");

  const loadUsers = async () => {
    try {
      setLoading(true);
      setError("");
      const result = await api.getUsers({ pageSize: 100 });
      setUsers((result.list || result.data?.list || []).map(toRecord));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "用户列表加载失败");
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
    if (action === "delete" && !window.confirm(`确认删除用户“${account}”吗？`))
      return;
    try {
      setActionId(id);
      setFeedback("");
      if (action === "toggle") {
        await api.toggleUserStatus(id);
        setFeedback(`已更新 ${account} 的账户状态。`);
        await loadUsers();
      }
      if (action === "delete") {
        await api.deleteUser(id);
        setFeedback(`已删除用户 ${account}。`);
        await loadUsers();
      }
    } catch (caught) {
      setFeedback(
        caught instanceof Error ? caught.message : "操作失败，请重试。",
      );
    } finally {
      setActionId(null);
    }
  };
  const createUser = async () => {
    if (!/^[\p{L}\p{N}._-]{2,32}$/u.test(newUser.username.trim())) {
      setCreateError("用户名需为 2-32 位字母、数字或 . _ -，且不能包含空格。");
      return;
    }
    if (!newUser.name.trim()) {
      setCreateError("请填写姓名。");
      return;
    }
    try {
      setActionId("create");
      setCreateError("");
      const createdUser = await api.createUser({ ...newUser, username: newUser.username.trim() });
      setCreateOpen(false);
      setNewUser(initialNewUser);
      setFeedback(`已创建用户 ${createdUser.username}。`);
      await loadUsers();
    } catch (caught) {
      setCreateError(
        caught instanceof Error ? caught.message : "创建用户失败。",
      );
    } finally {
      setActionId(null);
    }
  };

  const isSystemAdmin = role === "system_admin" || role === "admin";
  const openEdit = (row: UserRecord) => {
    setEditError("");
    setEditing(row);
    setEditForm({
      username: row.account,
      name: row.name === "--" ? "" : row.name,
      role: row.role,
      department: row.department === "--" ? "" : row.department,
      password: "",
    });
    setEditOpen(true);
  };
  const saveEdit = async () => {
    if (!editing) return;
    const username = editForm.username.trim();
    if (!/^[\p{L}\p{N}._-]{2,32}$/u.test(username)) {
      setEditError("用户名需为 2-32 位字母、数字或 . _ -，且不能包含空格。");
      return;
    }
    if (!editForm.name.trim()) {
      setEditError("请填写姓名。");
      return;
    }
    if (editForm.password && editForm.password.length < 12) {
      setEditError("新密码至少需要 12 位。");
      return;
    }
    try {
      setActionId(editing.id);
      setEditError("");
      const payload: any = {
        username,
        name: editForm.name.trim(),
        role: editForm.role,
        department: editForm.department,
      };
      if (editForm.password) payload.password = editForm.password;
      await api.updateUser(editing.id, payload);
      setEditOpen(false);
      setEditing(null);
      setFeedback(`已更新用户 ${username}。`);
      await loadUsers();
    } catch (caught) {
      setEditError(caught instanceof Error ? caught.message : "保存失败。");
    } finally {
      setActionId(null);
    }
  };

  if (!isAdmin) return <Navigate to="/dashboard" replace />;

  const columns: Column<UserRecord>[] = [
    {
      key: "account",
      label: "账户",
      width: "170px",
      sortable: true,
      render: (value, row) => (
        <div>
          <p className="font-semibold text-[color:var(--color-primary)]">
            {value}
          </p>
          <p className="text-helper mt-0.5">{row.name}</p>
        </div>
      ),
    },
    {
      key: "role",
      label: "角色",
      width: "120px",
      render: (value) => roleLabels[value as UserRecord["role"]] || value,
    },
    { key: "department", label: "科室/岗位", width: "140px" },
    { key: "createTime", label: "创建时间", width: "165px", sortable: true },
    { key: "lastLogin", label: "最后登录", width: "165px", sortable: true },
    {
      key: "status",
      label: "状态",
      width: "90px",
      align: "center",
      render: (value) => (
        <span className={value === "active" ? "tag-success" : "tag-warning"}>
          {value === "active" ? "正常" : "已禁用"}
        </span>
      ),
    },
    {
      key: "id",
      label: "操作",
      width: "380px",
      align: "right",
      render: (_value, row) => (
        <div
          className="flex items-center justify-end gap-2 whitespace-nowrap"
          onClick={(event) => event.stopPropagation()}
        >
          <button
            className="btn-secondary px-3 py-2"
            disabled={actionId === row.id}
            onClick={() => openEdit(row)}
          >
            编辑
          </button>
          {isSystemAdmin && (
            <button
              className="btn-secondary px-3 py-2"
              disabled={actionId === row.id}
              onClick={() => void runAction(row.id, "toggle", row.account)}
            >
              {row.status === "active" ? "禁用" : "启用"}
            </button>
          )}
          {isSystemAdmin && (
            <button
              className="btn-danger px-3 py-2"
              disabled={actionId === row.id}
              onClick={() => void runAction(row.id, "delete", row.account)}
            >
              删除
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
              <p className="text-helper uppercase tracking-wider text-[color:var(--color-primary)] mb-2">
                Accounts
              </p>
              <h1 className="text-page-title">用户管理</h1>
              <p className="text-body mt-2">
                管理用户账户、角色、科室和账户状态。
              </p>
            </div>
            <button
              className="btn-primary"
              onClick={() => {
                setCreateError("");
                setCreateOpen(true);
              }}
            >
              新增用户
            </button>
          </div>
          {feedback && (
            <div className="border border-blue-200 bg-blue-50 px-4 py-3 rounded-card flex items-center justify-between gap-4">
              <p className="text-body text-[color:var(--color-primary)]">
                {feedback}
              </p>
              <button className="btn-text" onClick={() => setFeedback("")}>
                关闭
              </button>
            </div>
          )}
          {error ? (
            <section className="card-base p-8 text-center">
              <p className="text-body text-[color:var(--color-error)] mb-4">
                {error}
              </p>
              <button
                className="btn-secondary"
                onClick={() => void loadUsers()}
              >
                重新加载
              </button>
            </section>
          ) : (
            <>
              <div className="flex items-center justify-between">
                <p className="text-helper">
                  共 {users.length} 位用户
                </p>
                <button
                  className="btn-text"
                  onClick={() => void loadUsers()}
                  disabled={loading}
                >
                  刷新列表
                </button>
              </div>
              <DataTable<UserRecord>
                columns={columns}
                data={users}
                rowKey="id"
                pageSize={10}
                loading={loading}
                emptyMessage="暂无用户数据"
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
                <h2 id="create-user-title" className="text-card-title">
                  新增用户
                </h2>
                <p className="text-helper mt-1">
                  请设置一个易记的用户名（2-32 位字母、数字或 . _ -），创建后使用该用户名和初始密码登录。
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
              <Field
                label="用户名（登录账号）"
                value={newUser.username}
                onChange={(value) =>
                  setNewUser((prev) => ({ ...prev, username: value }))
                }
              />
              <Field
                label="姓名"
                value={newUser.name}
                onChange={(value) =>
                  setNewUser((prev) => ({ ...prev, name: value }))
                }
              />
              <label className="block text-body font-semibold">
                角色
                <select
                  className="input-base mt-2 font-normal"
                  value={newUser.role}
                  onChange={(event) =>
                    setNewUser((prev) => ({
                      ...prev,
                      role: event.target.value as NewUser["role"],
                    }))
                  }
                >
                  {Object.entries(roleLabels).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <Field
                label="科室"
                value={newUser.department}
                onChange={(value) =>
                  setNewUser((prev) => ({ ...prev, department: value }))
                }
              />
              <div className="md:col-span-2">
                <Field
                  label="初始密码"
                  value={newUser.password}
                  onChange={(value) =>
                    setNewUser((prev) => ({ ...prev, password: value }))
                  }
                  type="password"
                />
              </div>
            </div>
            {createError && (
              <p className="text-helper text-[color:var(--color-error)] mt-4">
                {createError}
              </p>
            )}
            <div className="mt-6 flex justify-end gap-3">
              <button
                className="btn-secondary"
                disabled={actionId === "create"}
                onClick={() => setCreateOpen(false)}
              >
                取消
              </button>
              <button
                className="btn-primary"
                disabled={actionId === "create"}
                onClick={() => void createUser()}
              >
                {actionId === "create" ? "创建中..." : "创建用户"}
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
                <h2 id="edit-user-title" className="text-card-title">
                  编辑用户
                </h2>
                <p className="text-helper mt-1">
                  修改用户名、姓名、角色或科室。
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
              <Field
                label="用户名（登录账号）"
                value={editForm.username}
                onChange={(value) =>
                  setEditForm((prev) => ({ ...prev, username: value }))
                }
              />
              <Field
                label="姓名"
                value={editForm.name}
                onChange={(value) =>
                  setEditForm((prev) => ({ ...prev, name: value }))
                }
              />
              <label className="block text-body font-semibold">
                角色
                <select
                  className="input-base mt-2 font-normal"
                  value={editForm.role}
                  disabled={!isSystemAdmin}
                  onChange={(event) =>
                    setEditForm((prev) => ({
                      ...prev,
                      role: event.target.value as UserRecord["role"],
                    }))
                  }
                >
                  {Object.entries(roleLabels).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                {!isSystemAdmin && (
                  <p className="text-helper mt-1 font-normal">
                    仅系统管理员可修改角色。
                  </p>
                )}
              </label>
              <Field
                label="科室"
                value={editForm.department}
                onChange={(value) =>
                  setEditForm((prev) => ({ ...prev, department: value }))
                }
              />
              <div className="md:col-span-2">
                <Field
                  label="修改密码（留空则不修改）"
                  value={editForm.password}
                  onChange={(value) =>
                    setEditForm((prev) => ({ ...prev, password: value }))
                  }
                  type="password"
                />
              </div>
            </div>
            {editError && (
              <p className="text-helper text-[color:var(--color-error)] mt-4">
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
                取消
              </button>
              <button
                className="btn-primary"
                disabled={actionId === editing.id}
                onClick={() => void saveEdit()}
              >
                {actionId === editing.id ? "保存中..." : "保存"}
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
    <label className="block text-body font-semibold">
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
