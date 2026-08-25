// API client for the local SQLite-backed service.
const API_BASE = '/api';

class ApiClient {
  private token: string | null = null;

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }

  getToken(): string | null {
    if (!this.token) {
      this.token = localStorage.getItem('auth_token');
    }
    return this.token;
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('auth_token');
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers as Record<string, string>,
    };

    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: '请求失败' }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    const result = await response.json();
    if (!result.success) {
      throw new Error(result.message || '请求失败');
    }

    return result.data;
  }

  // ============ 认证 ============
  async login(username: string, password: string) {
    const data = await this.request<{ token: string; user: any }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    this.setToken(data.token);
    return data;
  }

  async logout() {
    await this.request('/auth/logout', { method: 'POST' });
    this.clearToken();
  }

  async getCurrentUser() {
    return this.request<any>('/auth/me');
  }

  // ============ 受检者管理 ============
  async getCases(params: {
    page?: number;
    pageSize?: number;
    keyword?: string;
    gender?: string;
    status?: string;
    department?: string;
    doctor?: string;
  } = {}) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && query.set(k, String(v)));
    return this.request<any>(`/cases?${query}`);
  }

  async getCase(id: string) {
    return this.request<any>(`/cases/${id}`);
  }

  async createCase(data: any) {
    return this.request<any>('/cases', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateCase(id: string, data: any) {
    return this.request<any>(`/cases/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteCase(id: string) {
    return this.request<any>(`/cases/${id}`, { method: 'DELETE' });
  }

  async batchCreateCases(cases: any[]) {
    return this.request<any>('/cases/batch', {
      method: 'POST',
      body: JSON.stringify({ cases }),
    });
  }

  async deleteCases(ids: string[]) {
    return this.request<any>('/cases/batch-delete', {
      method: 'POST',
      body: JSON.stringify({ ids }),
    });
  }

  async getCaseStats() {
    return this.request<any>('/cases/stats/summary');
  }

  // ============ 文件管理 ============
  async getFiles(params: { caseId?: string; page?: number; pageSize?: number } = {}) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && query.set(k, String(v)));
    return this.request<any>(`/files?${query}`);
  }

  async uploadFile(data: { caseId: string; file: File; scanTime?: string }) {
    const form = new FormData();
    form.append('caseId', data.caseId);
    form.append('file', data.file);
    if (data.scanTime) form.append('scanTime', data.scanTime);
    const token = this.getToken();
    const response = await fetch(`${API_BASE}/files`, { method: 'POST', body: form, headers: token ? { Authorization: `Bearer ${token}` } : {} });
    const result = await response.json().catch(() => ({ message: '上传失败' }));
    if (!response.ok || !result.success) throw new Error(result.message || `HTTP ${response.status}`);
    return result.data;
  }

  async deleteFile(id: string) {
    return this.request<any>(`/files/${id}`, { method: 'DELETE' });
  }

  // ============ 算法分析 ============
  async analyzeSingle(caseId: string, fileId?: string) {
    return this.request<any>('/analysis/single', {
      method: 'POST',
      body: JSON.stringify({ caseId, fileId }),
    });
  }

  async analyzeBatch(caseIds: string[]) {
    return this.request<any>('/analysis/batch', {
      method: 'POST',
      body: JSON.stringify({ caseIds }),
    });
  }

  async checkAnalysis(caseId: string) {
    return this.request<any>(`/analysis/check/${caseId}`);
  }

  // ============ 报告管理 ============
  async getReports(params: {
    page?: number;
    pageSize?: number;
    caseId?: string;
    caseName?: string;
    aisLevel?: string;
  } = {}) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && query.set(k, String(v)));
    return this.request<any>(`/reports?${query}`);
  }

  async getReport(id: string) {
    return this.request<any>(`/reports/${id}`);
  }

  async updateReportDiagnosis(id: string, data: any) {
    return this.request<any>(`/reports/${id}/diagnosis`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async getReviewQueue() { return this.request<any>('/reviews'); }
  async approveReview(reportId: string) { return this.request<any>(`/reviews/${reportId}/approve`, { method: 'POST' }); }
  async returnReview(reportId: string) { return this.request<any>(`/reviews/${reportId}/return`, { method: 'POST' }); }
  async getAnnotationSubjects() { return this.request<{ subjects: string[] }>('/annotation/subjects'); }
  async createAnnotationSession(reportId: string) { return this.request<{ annotationUrl: string; expiresAt: number }>('/annotation/sessions', { method: 'POST', body: JSON.stringify({ reportId }) }); }
  async completeAnnotation(reportId: string) { return this.request<any>(`/annotation/reports/${reportId}/completed`, { method: 'POST' }); }
  async resubmitReview(caseId: string) { return this.request<any>(`/reviews/${caseId}/resubmit`, { method: 'POST' }); }

  // ============ 任务管理 ============
  async getTasks(params: {
    page?: number;
    pageSize?: number;
    type?: string;
    status?: string;
    submitter?: string;
  } = {}) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && query.set(k, String(v)));
    return this.request<any>(`/tasks?${query}`);
  }

  async getTask(id: string) {
    return this.request<any>(`/tasks/${id}`);
  }

  async cancelTask(id: string) {
    return this.request<any>(`/tasks/${id}/cancel`, { method: 'POST' });
  }

  async retryTask(id: string) {
    return this.request<any>(`/tasks/${id}/retry`, { method: 'POST' });
  }

  async getTaskStats() {
    return this.request<any>('/tasks/stats');
  }

  async getBackups() { return this.request<any>('/backups'); }
  async createBackup() { return this.request<any>('/backups', { method: 'POST' }); }
  async restoreBackup(name: string) { return this.request<any>(`/backups/${encodeURIComponent(name)}/restore`, { method: 'POST' }); }

  // ============ 统计 ============
  async getStatistics() {
    return this.request<any>('/statistics/overview');
  }

  async getCasesDistribution(type: string = 'department') {
    return this.request<any>(`/statistics/cases-distribution?type=${type}`);
  }

  async getAISDistribution() {
    return this.request<any>('/statistics/ais-distribution');
  }

  async getTimeSeries(metric: string = 'cases', period: string = 'week') {
    return this.request<any>(`/statistics/time-series?metric=${metric}&period=${period}`);
  }

  // ============ 用户管理 ============
  async getUsers(params: { page?: number; pageSize?: number; keyword?: string; role?: string } = {}) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && query.set(k, String(v)));
    return this.request<any>(`/users?${query}`);
  }

  async createUser(data: any) {
    return this.request<any>('/users', { method: 'POST', body: JSON.stringify(data) });
  }

  async updateUser(id: string, data: any) {
    return this.request<any>(`/users/${id}`, { method: 'PUT', body: JSON.stringify(data) });
  }

  async deleteUser(id: string) {
    return this.request<any>(`/users/${id}`, { method: 'DELETE' });
  }

  async resetPassword(id: string, password: string) {
    return this.request<any>(`/users/${id}/reset-password`, { method: 'POST', body: JSON.stringify({ password }) });
  }

  async toggleUserStatus(id: string) {
    return this.request<any>(`/users/${id}/toggle-status`, { method: 'POST' });
  }

  // ============ 设置 ============
  async getSettings() {
    return this.request<any>('/settings');
  }

  async updateSettings(section: string, data: any) {
    return this.request<any>(`/settings/${section}`, { method: 'PUT', body: JSON.stringify(data) });
  }

  async testAlgorithmConnection() {
  }

  // ============ 个人中心 ============
  async getProfile() {
    return this.request<any>('/profile');
  }

  async updateProfile(data: { name?: string; department?: string }) {
    return this.request<any>('/profile', { method: 'PUT', body: JSON.stringify(data) });
  }

  async changePassword(oldPassword: string, newPassword: string) {
    return this.request<any>('/profile/password', {
      method: 'PUT',
      body: JSON.stringify({ oldPassword, newPassword, confirmPassword: newPassword }),
    });
  }

  // ============ 帮助 ============
  async getHelpDocs() {
    return this.request<any>('/help');
  }

  async getFAQ() {
    return this.request<any>('/help/faq');
  }

  async submitFeedback(data: { title: string; content: string; contact?: string }) {
    return this.request<any>('/help/feedback', { method: 'POST', body: JSON.stringify(data) });
  }
}

export const api = new ApiClient();
export default api;
