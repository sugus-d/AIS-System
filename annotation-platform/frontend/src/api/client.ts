// src/api/client.ts
const BASE = '/api';
const DEFAULT_TIMEOUT_MS = 30_000;

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** 带超时的 fetch，与已有 signal 组合不冲突 */
async function fetchWithTimeout(url: string, options?: RequestInit, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // 若传入方已带 signal，父 signal 取消时也中止此 controller
  const existingSignal = options?.signal;
  if (existingSignal && existingSignal.aborted) {
    clearTimeout(timer);
    throw new DOMException('Aborted', 'AbortError');
  }
  const onExistingAbort = existingSignal
    ? () => controller.abort()
    : null;
  if (onExistingAbort) existingSignal!.addEventListener('abort', onExistingAbort, { once: true });

  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timer);
    if (onExistingAbort) existingSignal!.removeEventListener('abort', onExistingAbort);
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetchWithTimeout(
    `${BASE}${path}`,
    {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    },
    DEFAULT_TIMEOUT_MS,
  );
  if (!res.ok) {
    let detail = `${path}: ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail?.issues) {
        detail = JSON.stringify(body.detail.issues);
      } else if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
    } catch { /* ignore parse errors */ }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export { fetchWithTimeout };

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  });
}

export function apiPut<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'PUT',
    body: body ? JSON.stringify(body) : undefined,
  });
}
