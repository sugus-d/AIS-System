// src/components/panels/RightPanel.tsx
import { useEffect, useState, useCallback, useMemo } from 'react';
import { Box, Typography, Chip, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useSubjectStore } from '../../stores/subjectStore';
import { useUIStore } from '../../stores/uiStore';
import { fetchMetrics, fetchClinicalData } from '../../api/metrics';
import SubjectInfo from './SubjectInfo';
import type { ClinicalData } from '../../types';

const ALL_METRICS = [
  // Symmetry
  { id: 'neck_dy', label: '颈根ΔY', group: 'sym', fn: (m: any) => m.neck_root?.dy },
  { id: 'shoulder_dy', label: '肩臂ΔY', group: 'sym', fn: (m: any) => m.shoulder_transition?.dy },
  { id: 'scapular_dy', label: '肩胛ΔY', group: 'sym', fn: (m: any) => m.scapular_peaks?.dy },
  { id: 'axilla_dy', label: '腋窝ΔY', group: 'sym', fn: (m: any) => m.axilla?.dy },
  { id: 'waist_dy', label: '腰部ΔY', group: 'sym', fn: (m: any) => m.waist?.dy },
  // Width
  { id: 'neck_w', label: '颈宽', group: 'width', fn: (m: any) => m.neck_root?.dx },
  { id: 'shoulder_w', label: '肩宽', group: 'width', fn: (m: any) => m.shoulder_transition?.dx },
  { id: 'ax_w', label: '腋宽', group: 'width', fn: (m: any) => m.axilla?.dx },
  { id: 'wa_w', label: '腰宽', group: 'width', fn: (m: any) => m.waist?.dx },
  // Ratios
  { id: 'neck_ax_ratio', label: '颈腋宽比', group: 'ratio', fn: (m: any) => m.neck_axilla_ratio },
  { id: 'waist_ax_ratio', label: '腰腋宽比', group: 'ratio', fn: (m: any) => m.waist_axilla_ratio },
];

function MetricRow({ label, value, threshold }: { label: string; value: number | string; threshold?: number }) {
  const num = typeof value === 'number' ? value : parseFloat(value);
  const isNum = !isNaN(num);
  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.4, px: 0.5, fontSize: 12, '&:hover': { bgcolor: 'rgba(56,189,248,0.04)' } }}>
      <Typography variant="caption" sx={{ fontSize: 11, color: 'text.secondary' }}>{label}</Typography>
      <Chip label={isNum ? (typeof value === 'number' ? value.toFixed(1) : value) : value}
        size="small"
        color={threshold && isNum ? (num > threshold ? 'error' : 'success') : 'default'}
        variant="outlined" sx={{ fontSize: 10, height: 18, '& .MuiChip-label': { px: 0.8 } }} />
    </Box>
  );
}

export default function RightPanel() {
  const { currentId, currentDetail } = useSubjectStore();
  const [metrics, setMetrics] = useState<any>(null);
  const [clinicalMap, setClinicalMap] = useState<ClinicalData>({});
  const validationIssues = useUIStore((s) => s.validationIssues);

  const loadMetrics = useCallback(async () => {
    if (!currentId) return;
    try {
      const r = await fetchMetrics(currentId);
      setMetrics(r.metrics);
    } catch (e) { console.error("加载指标失败", e); }
  }, [currentId]);

  useEffect(() => { fetchClinicalData().then(setClinicalMap).catch((e) => console.error("加载临床数据失败", e)); }, []);
  useEffect(() => { loadMetrics(); }, [loadMetrics]);

  const clin = currentId ? clinicalMap[currentId] : null;

  // 预计算 metric 分组，避免每次 render 反复调用 m.fn(metrics)
  const metricGroups = useMemo(() => {
    if (!metrics) return null;
    const hasValue = (m: typeof ALL_METRICS[number]) => {
      const v = m.fn(metrics);
      return v !== undefined && v !== null;
    };
    return {
      sym: ALL_METRICS.filter((m) => m.group === 'sym' && hasValue(m)).map((m) => ({ ...m, value: m.fn(metrics) })),
      width: ALL_METRICS.filter((m) => m.group === 'width' && hasValue(m)).map((m) => ({ ...m, value: m.fn(metrics) })),
      ratio: ALL_METRICS.filter((m) => m.group === 'ratio' && hasValue(m)).map((m) => ({ ...m, value: m.fn(metrics) })),
    };
  }, [metrics]);

  const curveRows = (c: any) => {
    const rows: { label: string; value: string }[] = [];
    for (let i = 1; i <= 4; i++) {
      const apex = c[`curve${i}_apex`];
      const cobb = c[`curve${i}_cobb`];
      const dir = c[`curve${i}_direction`];
      const lvl = c[`curve${i}_level`];
      if (apex || cobb != null) {
        rows.push({
          label: `曲线${i}`,
          value: `${lvl || ''} ${dir || ''} ${apex || ''} ${cobb != null ? cobb.toFixed(1) + '°' : ''}`.trim(),
        });
      }
    }
    return rows;
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', bgcolor: 'background.paper', borderLeft: 1, borderColor: 'divider' }}>
      <Box sx={{ px: 1.5, py: 1, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', gap: 1 }}>
        <Box sx={{ width: 3, height: 14, borderRadius: 0.5, bgcolor: 'secondary.main' }} />
        <Typography variant="subtitle2" sx={{ color: 'text.secondary', fontWeight: 600, letterSpacing: '0.04em', fontSize: 11 }}>
          关键指标
        </Typography>
      </Box>
      <Box sx={{ flex: 1, overflow: 'auto', px: 1.5, py: 1.5 }}>
        <SubjectInfo />

        {/* Validation issues */}
        {validationIssues.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block', mb: 0.5, fontSize: 11, borderBottom: '1px solid', borderColor: 'divider', pb: 0.4, letterSpacing: '0.03em' }}>
              坐标校验
            </Typography>
            {validationIssues.filter(i => i.severity === 'error').map((issue, idx) => (
              <Box key={`err-${idx}`} sx={{ display: 'flex', gap: 0.5, py: 0.25, px: 0.5, fontSize: 11, color: 'error.main' }}>
                <Typography variant="caption" sx={{ fontSize: 11 }}>● {issue.message}</Typography>
              </Box>
            ))}
            {validationIssues.filter(i => i.severity === 'warning').map((issue, idx) => (
              <Box key={`warn-${idx}`} sx={{ display: 'flex', gap: 0.5, py: 0.25, px: 0.5, fontSize: 11, color: 'warning.dark' }}>
                <Typography variant="caption" sx={{ fontSize: 11 }}>● {issue.message}</Typography>
              </Box>
            ))}
          </Box>
        )}

        {/* Symmetry metrics */}
        {metricGroups?.sym && metricGroups.sym.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block', mb: 0.5, fontSize: 11, borderBottom: '1px solid', borderColor: 'divider', pb: 0.4, letterSpacing: '0.03em' }}>对称性指标</Typography>
            {metricGroups.sym.map((m) => (
              <MetricRow key={m.id} label={m.label} value={m.value} threshold={20} />
            ))}
          </Box>
        )}

        {/* Width metrics */}
        {metricGroups?.width && metricGroups.width.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block', mb: 0.5, fontSize: 11, borderBottom: '1px solid', borderColor: 'divider', pb: 0.4, letterSpacing: '0.03em' }}>宽度指标</Typography>
            {metricGroups.width.map((m) => (
              <MetricRow key={m.id} label={m.label} value={m.value} />
            ))}
          </Box>
        )}

        {/* Ratio metrics */}
        {metricGroups?.ratio && metricGroups.ratio.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block', mb: 0.5, fontSize: 11, borderBottom: '1px solid', borderColor: 'divider', pb: 0.4, letterSpacing: '0.03em' }}>比率指标</Typography>
            {metricGroups.ratio.map((m) => (
              <MetricRow key={m.id} label={m.label} value={m.value} />
            ))}
          </Box>
        )}

        {/* Clinical data */}
        {clin && (
          <Accordion defaultExpanded sx={{ bgcolor: 'transparent', boxShadow: 'none', '&:before': { display: 'none' } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ fontSize: 14 }} />}>
              <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', fontSize: 11, letterSpacing: '0.03em' }}>临床数据</Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0, py: 0.5 }}>
              {clin.max_cobb != null && <MetricRow label="最大 Cobb 角" value={clin.max_cobb + '°'} threshold={25} />}
              {clin.atr != null && <MetricRow label="ATR" value={clin.atr + '°'} />}
              {clin.sex && <MetricRow label="性别" value={clin.sex} />}
              {clin.height_cm != null && <MetricRow label="身高" value={clin.height_cm + ' cm'} />}
              {clin.weight_kg != null && <MetricRow label="体重" value={clin.weight_kg + ' kg'} />}
              {clin.height_cm != null && clin.weight_kg != null && (() => {
                const bmi = clin.weight_kg / (clin.height_cm / 100) ** 2;
                return <MetricRow label="BMI" value={bmi.toFixed(1)} />;
              })()}
              {clin.arm_span_cm != null && <MetricRow label="臂展" value={clin.arm_span_cm + ' cm'} />}
              {clin.seating_height_cm != null && <MetricRow label="坐高" value={clin.seating_height_cm + ' cm'} />}
              {clin.has_brace != null && <MetricRow label="支具" value={clin.has_brace === '1' ? '是' : '否'} />}
              {clin.xray_date && <MetricRow label="X光日期" value={clin.xray_date} />}
              {clin.recruit_date && <MetricRow label="招募日期" value={clin.recruit_date} />}
              {curveRows(clin).map((r, idx) => (
                <MetricRow key={idx} label={r.label} value={r.value} />
              ))}
              {clin.remarks && (
                <Box sx={{ py: 0.5, px: 0.5, fontSize: 11 }}>
                  <Typography variant="caption" sx={{ fontSize: 11 }}>📝 {clin.remarks}</Typography>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        )}
      </Box>
    </Box>
  );
}
