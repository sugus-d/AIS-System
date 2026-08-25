"""纯渲染函数 — 每个表一个函数，读取模板 + 填充数据 → 返回完整 HTML。

不加载数据，不做计算，只做字符串替换。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

# 模板目录：优先 AIS_EXPORT_DIR 环境变量（外部 docs/manuscript），
# 否则向上定位含 docs/manuscript/html 的祖先目录（自动适配主仓库与 worktree 深度）。
_TEMPLATE_DIR: Path | None = None
_explicit = os.environ.get("AIS_EXPORT_DIR")
if _explicit:
    _TEMPLATE_DIR = Path(_explicit) / "manuscript" / "html"
else:
    for _parent in Path(__file__).resolve().parents:
        if (_parent / "docs" / "manuscript" / "html").exists():
            _TEMPLATE_DIR = _parent / "docs" / "manuscript" / "html"
            break
if _TEMPLATE_DIR is None:
    # 兜底：找不到模板目录时指向项目根下相对路径（模板缺失时 _read 返回空串）
    _TEMPLATE_DIR = Path(__file__).resolve().parents[5] / "docs" / "manuscript" / "html"
TEMPLATE_DIR = _TEMPLATE_DIR


def _read(name: str) -> str:
    p = TEMPLATE_DIR / name
    return p.read_text() if p.exists() else ""


def _replace_tbody(html: str, rows: str) -> str:
    return re.sub(r"<tbody>.*?</tbody>", f"<tbody>{rows}</tbody>", html, flags=re.DOTALL)


def _replace_header_n(html: str, **counts: int) -> str:
    """Replace hardcoded n= values in <th> headers, e.g. Total (n=80) → Total (n=122)."""
    for label, n in counts.items():
        html = re.sub(rf"{label}\s*\(n=\d+\)", f"{label} (n={n})", html)
    return html


# ────────────────────────────────────
# Utilities
# ────────────────────────────────────


def _v(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return s if s not in ("—", "nan", "") else ""


def _r(v: object) -> str:
    s = _v(v)
    if not s or s == "-":
        return "-"
    try:
        return f"{float(s):.3f}"
    except (ValueError, TypeError):
        return s


def _ci(v: object) -> str:
    s = str(v).strip() if v else ""
    return s.strip('"') if s.startswith('"') else s


def _parse_ms(s: str) -> tuple[float, float]:
    m = re.match(r"([\d.]+)±([\d.]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


# ────────────────────────────────────
# Table 1
# ────────────────────────────────────


def render_table1(df: pd.DataFrame, clinical: dict | None = None) -> str:
    """Render Table 1 from demographic CSV and optional clinical stats."""
    # Unpack clinical stats (overall + by gender)
    bmi_d  = (clinical.get("bmi_detail") or {}) if clinical else {}
    ct_d   = (clinical.get("ct_detail") or {}) if clinical else {}
    sev_d  = (clinical.get("sev_detail") or {}) if clinical else {}
    # Build lookup: (gender, ais) -> row
    lu: dict[tuple[str, str], dict] = {}
    for _, r in df.iterrows():
        lu[(str(r["gender"]), str(r["ais"]))] = dict(r)

    def _ms(g: str, a: str, col: str) -> str:
        d = lu.get((g, a), {})
        v = d.get(col, "")
        return str(v) if pd.notna(v) and str(v).strip() not in ("", "-", "nan") else ""

    def _combine(col: str, *groups: tuple[str, str]) -> str:
        total_n, wsum = 0, 0.0
        for g, a in groups:
            s = _ms(g, a, col)
            if not s:
                continue
            try:
                m, _ = _parse_ms(s)
                n = int(lu.get((g, a), {}).get("n", 0))
                total_n += n
                wsum += m * n
            except (ValueError, TypeError):
                continue
        return f"{wsum / total_n:.1f}" if total_n else ""

    def _cell(g: str, a: str, col: str) -> str:
        v = _ms(g, a, col)
        return f"<td>{v}</td>" if v else "<td></td>"

    def _char_row(label: str, col: str) -> str:
        cells = (
            f"<td>{_combine(col, ('Total','Non-AIS'),('Total','AIS'))}</td>"
            + _cell("Total", "Non-AIS", col)
            + _cell("Total", "AIS", col)
            + f"<td>{_combine(col, ('Male','Non-AIS'),('Male','AIS'))}</td>"
            + _cell("Male", "Non-AIS", col)
            + _cell("Male", "AIS", col)
            + f"<td>{_combine(col, ('Female','Non-AIS'),('Female','AIS'))}</td>"
            + _cell("Female", "Non-AIS", col)
            + _cell("Female", "AIS", col)
        )
        return f"<tr><td>{label}</td>{cells}</tr>"

    rows = ""
    # Age — no data
    rows += "<tr><td>Age (years)</td>" + "<td></td>" * 9 + "</tr>"
    # Characteristics with data
    for label, col in [("Height (cm)", "height"), ("Weight (kg)", "weight"), ("BMI (kg/m²)", "bmi")]:
        rows += _char_row(label, col)
    # Helper: 9 cells for a category from detail data
    def cat_cells(detail: dict, label: str) -> str:
        data = detail.get(label, {})
        def _get(g: str, ais: str) -> int:
            return data.get(g, {}).get(ais, 0)
        # Total: overall, Non-AIS, AIS
        all_non = _get("Male", "Non-AIS") + _get("Female", "Non-AIS")
        all_ais = _get("Male", "AIS") + _get("Female", "AIS")
        ttl = all_non + all_ais
        # Male: total, Non-AIS, AIS
        m_non = _get("Male", "Non-AIS")
        m_ais = _get("Male", "AIS")
        m_ttl = m_non + m_ais
        # Female: total, Non-AIS, AIS
        f_non = _get("Female", "Non-AIS")
        f_ais = _get("Female", "AIS")
        f_ttl = f_non + f_ais
        cells = f"<td>{ttl}</td>"
        cells += f"<td>{all_non}</td>"
        cells += f"<td>{all_ais}</td>"
        cells += f"<td>{m_ttl}</td>"
        cells += f"<td>{m_non}</td>"
        cells += f"<td>{m_ais}</td>"
        cells += f"<td>{f_ttl}</td>"
        cells += f"<td>{f_non}</td>"
        cells += f"<td>{f_ais}</td>"
        return cells

    # BMI categories
    for label in ["Underweight", "Normal weight", "Overweight", "Obesity"]:
        rows += f"<tr><td>{label}</td>{cat_cells(bmi_d, label)}</tr>"
    # Cobb
    rows += _char_row("Cobb angle (°)", "max_cobb")
    # Curve Type
    rows += '<tr><td class="section-label">Curve Type</td>' + "<td></td>" * 9 + "</tr>"
    for ct_key, ct_label in [("Thoracic", "Thoracic curve"), ("Lumbar", "Lumbar curve"), ("Double", "Double Curves")]:
        rows += f'<tr><td class="sub-item">{ct_label}</td>{cat_cells(ct_d, ct_key)}</tr>'
    # Severity
    rows += '<tr><td class="section-label">Severity</td>' + "<td></td>" * 9 + "</tr>"
    for s in ["Normal", "Mild", "Moderate", "Severe"]:
        rows += f'<tr><td class="sub-item">{s}</td>{cat_cells(sev_d, s)}</tr>'

    fn = '<p style="font-size:11px;color:#888">Age data not collected in this cohort. BMI categories use adult cutoffs (Underweight &lt;18.5, Normal 18.5–25, Overweight 25–30, Obesity ≥30) since age data is unavailable for age-adjusted percentiles. Normal cases (max_cobb &lt;10°) count=7 from clinical cohort (n=122).</p>'

    # 从数据中提取实际 n 值并更新表头
    n_total = int(df[df["gender"] == "Total"]["n"].sum())
    n_male = int(df[df["gender"] == "Male"]["n"].sum())
    n_female = int(df[df["gender"] == "Female"]["n"].sum())
    html = _replace_tbody(_read("table1.html"), rows)
    html = _replace_header_n(html, Total=n_total, Male=n_male, Female=n_female)
    return html + fn


# ────────────────────────────────────
# Table 2 — no data
# ────────────────────────────────────


def render_table2(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return _read("table2.html")
    cols = ["Sh.IB", "Sh.A", "Sca.IB", "Sca.A", "ASIS.A", "Trunk.L", "Sh.W", "Sh.AI", "Pe.AI"]
    vals = {}
    for col in cols:
        v = df[col].dropna()
        vals[col] = f"{v.mean():.2f}±{v.std():.2f}" if len(v) > 0 else ""

    tmpl = _read("table2.html")
    rows = ""
    for param in cols:
        v = vals.get(param, "")
        if v:
            rows += f"<tr><td>{param}</td><td>{v}</td><td>{v}</td><td></td><td></td>" + "<td></td>" * 6 + "</tr>"
        else:
            rows += f"<tr><td>{param}</td>" + "<td></td>" * 10 + "</tr>"
    footnote = '<p style="font-size:11px;color:#888">Auto/Manual: values from automated landmark detection (same as GT). Observer 1/2: unavailable. MAE/RMSE/R²/r/ICC: unavailable (no Observer data).</p>'
    return _replace_tbody(tmpl, rows) + footnote


# ────────────────────────────────────
# Table 3
# ────────────────────────────────────


def render_table3(df: pd.DataFrame) -> str:
    lu = {(r["index"], r["severity"]): r["value"] for _, r in df.iterrows()}
    lu_p = {(r["index"], r["severity"]): r["value"] for _, r in df.iterrows()}
    name_map = {"Ai": "Asymmetric Index", "Curvature index": "Curvature Index", "Height index": "Height Index", "Nvi": "Normal Vector Index", "Ri": "Roughness Index"}
    # Template has 7 columns: th(empty) | Whole cohort | Normal | Mild | Moderate | Severe | P value
    # The first td (label) fills the empty header column, leaving 6 data cells
    col_order = ["Whole cohort", "Normal", "Mild", "Moderate", "Severe", "P value"]
    rows = ""
    for ridx, label in name_map.items():
        cells = ""
        for c in col_order:
            if c == "P value":
                cells += f"<td>{_v(lu_p.get((ridx, 'P value'), ''))}</td>"
            else:
                cells += f"<td>{_v(lu.get((ridx, c), ''))}</td>"
        rows += f"<tr><td>{label}</td>{cells}</tr>"
    fn = '<p style="font-size:11px;color:#888">Values are group means per severity; P value from ANOVA across severity groups.</p>'
    return _replace_tbody(_read("table3.html"), rows) + fn


# ────────────────────────────────────
# Table 4
# ────────────────────────────────────


def render_table4(df: pd.DataFrame) -> str:
    sev_map = {"all": "Whole cohort", "mild": "Mild cases", "moderate": "Moderate cases",
               "severe": "Severe cases", "normal": "Normal cases"}
    display = {"Ai": "Asymmetric Index", "Curvature index": "Curvature Index", "Height index": "Height Index", "Nvi": "Normal Vector Index", "Ri": "Roughness Index"}
    rows = ""
    for idx_name, disp in display.items():
        sub = df[df["index"] == idx_name]
        pairs = {}
        for _, r in sub.iterrows():
            pairs[r["severity"].lower()] = (_r(r.get("r", "")), _v(r.get("p", "")))
        rows += f"<tr><td rowspan='5'>{disp}</td>"
        for sk in ["all", "mild", "moderate", "severe", "normal"]:
            rr, pp = pairs.get(sk, ("", ""))
            rows += f"<td>{sev_map[sk]}</td><td>{rr}</td><td>{pp}</td></tr>"
        rows += "<tr><td colspan='4'></td></tr>"
    for sk in ["all", "mild", "moderate", "severe", "normal"]:
        rows += f"<td>{sev_map[sk]}</td><td></td><td></td></tr>"
    fn = '<p style="font-size:11px;color:#888">Pearson r between each index and max_cobb within the subgroup; P value of the correlation.</p>'
    return _replace_tbody(_read("table4.html"), rows) + fn


# ────────────────────────────────────
# Table 5
# ────────────────────────────────────


def render_table5(df: pd.DataFrame) -> str:
    pred = {}
    for _, r in df.iterrows():
        pred[r["subgroup"]] = r
    sub_map = {"Overall": "Whole cohort", "Normal": "Normal (0-10°)",
               "Mild": "Mild (10~20°)",
               "Moderate": "Moderate (20~40°)", "Severe": "Severe (≥40°)",
               "Thoracic": "Thoracic Curve", "Lumbar": "Lumbar Curve",
               "Double": "Double Curve"}
    order = ["Whole cohort", "Normal (0-10°)", "Mild (10~20°)",
             "Moderate (20~40°)", "Severe (≥40°)",
             "Thoracic Curve", "Lumbar Curve", "Double Curve"]
    rows = ""
    for name in order:
        rd = None
        for dk, dr in pred.items():
            if sub_map.get(dk, dk) == name or dk == name:
                rd = dr
                break
        if rd is not None:
            rows += (
                f"<tr><td>{name}</td>"
                f"<td>{_v(rd.get('gt_mean', ''))}</td>"
                f"<td>{_v(rd.get('grdf_mean', ''))}</td>"
                f"<td>{_ci(rd['mae_ci'])}</td>"
                f"<td>{_ci(rd['rmse_ci'])}</td>"
                f"<td>{_r(rd.get('r2', ''))}</td>"
                f"<td>{_r(rd.get('r', ''))}</td></tr>"
            )
        else:
            rows += f"<tr><td>{name}</td>" + "<td></td>" * 6 + "</tr>"
    return _replace_tbody(_read("table5.html"), rows)


# ────────────────────────────────────
# Table 6
# ────────────────────────────────────


def render_table6(df: pd.DataFrame) -> str:
    rows = ""
    for _, r in df.iterrows():
        sev = r["severity"]
        prec = _v(r["precision"])
        rec = _v(r["recall"])
        f1v = _v(r["f1"])
        bold = "<b>" if "Accuracy" in sev or "Weighted" in sev else ""
        close = "</b>" if bold else ""
        rows += f"<tr><td>{bold}{sev}{close}</td><td>{prec}</td><td>{rec}</td><td>{f1v}</td></tr>"
    fn = '<p style="font-size:11px;color:#888">Precision/Recall/F1 for Normal: 2/7 Normal subjects correctly classified, 5 misclassified as Mild. Overall/Weighted F1: aggregated metrics without per-class breakdown.</p>'
    return _replace_tbody(_read("table6.html"), rows) + fn
