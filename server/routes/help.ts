import { Router, Request, Response } from 'express';
import { audit, db } from '../services/database';

const router = Router();

// 获取帮助文档列表
router.get('/', (req: Request, res: Response) => {
  const { category } = req.query;
  
  const isAdmin = ['system_admin', 'institution_admin'].includes((req as Request & { user?: { role?: string } }).user?.role || '');
  
  const docs = [
    { id: 1, title: 'PLY scan upload', content: 'Upload an original .ply back scan before running analysis.', category: 'common' },
    { id: 2, title: 'AIS analysis', content: 'Analysis runs locally through the bundled AIS algorithm service.', category: 'common' },
    ...(isAdmin ? [{ id: 3, title: 'Local administration', content: 'Manage local accounts and application settings from the administration area.', category: 'admin' }] : []),
  ];
  
  if (category) {
    // 按分类筛选
  }
  
  res.json({
    success: true,
    data: docs
  });
});

// 获取 FAQ - 必须在 /:id 之前注册
router.get('/faq', (_req: Request, res: Response) => {
  const faqs = [
    {
      id: 1,
      question: '如何创建受检者档案？',
      answer: '进入受检者管理页面，点击"单个建档"按钮，填写完整信息后保存即可。',
    },
    {
      id: 2,
      question: '上传文件支持哪些格式？',
      answer: '目前仅支持 STL 格式的 3D 扫描文件，单个文件大小不超过 100MB。',
    },
    {
      id: 3,
      question: '如何查看历史报告版本？',
      answer: '在报告详情页点击"历史版本"按钮，可以查看和恢复历史版本。',
    },
    {
      id: 4,
      question: '密码忘记了怎么办？',
      answer: '在登录页点击"忘记密码"，通过绑定的邮箱或手机号重置密码。',
    },
    {
      id: 5,
      question: '为什么无法发起分析？',
      answer: '请确保受检者已完善基本信息且已上传 3D 扫描文件。',
    },
  ];
  
  res.json({
    success: true,
    data: faqs
  });
});

// 获取单个帮助文档
router.get('/:id', (req: Request, res: Response) => {
  const id = parseInt(String(req.params.id));
  
  const allDocs = [
    { id: 1, title: 'PLY scan upload', content: 'Upload an original .ply back scan before running analysis.' },
    { id: 2, title: 'AIS analysis', content: 'Analysis runs locally through the bundled AIS algorithm service.' },
    { id: 3, title: 'Local administration', content: 'Manage local accounts and application settings from the administration area.' },
  ];
  const doc = allDocs.find(d => d.id === id);
  
  if (!doc) {
    return res.status(404).json({ success: false, message: '文档不存在' });
  }
  
  res.json({
    success: true,
    data: doc
  });
});

// 提交反馈
router.post('/feedback', async (req: Request & { user?: { id?: string } }, res: Response) => {
  const { title, content, contact } = req.body;
  if (!String(title || '').trim() || !String(content || '').trim()) {
    return res.status(400).json({ success: false, message: '标题和反馈内容不能为空。' });
  }
  const userId = (req as Request & { user?: { id?: string } }).user?.id;
  try {
    const feedback = await db.feedback.create({ data: { userId, title: String(title).trim(), content: String(content).trim(), contact: contact ? String(contact).trim() : null } });
    await audit(userId, 'submit_feedback', 'Feedback', feedback.id);
    return res.status(201).json({ success: true, data: { id: feedback.id, createdAt: feedback.createdAt } });
  } catch {
    return res.status(500).json({ success: false, message: '反馈保存失败。' });
  }
});

export default router;
