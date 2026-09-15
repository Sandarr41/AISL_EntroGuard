"""Минимальные SVG-хелперы для графиков в make_report.py, без внешних
зависимостей.

Библиотека графиков намеренно не используется: отчёт — это один
статический HTML-файл, который должен открываться двойным кликом,
офлайн, на любой машине, куда его скопируют -- никаких CDN, никакого
matplotlib в браузере. Ровно столько SVG, сколько нужно для линейных и
столбчатых графиков с подсказками при наведении (нативный <title>, без JS)."""
import html

PALETTE = {
    "none": "#94a3b8",
    "gaussian": "#3b82f6",
    "pgd": "#f97316",
    "entroguard": "#22c55e",
    "entroguard_transformer(zero-shot)": "#14b8a6",
    "entroguard_lstm(native)": "#a855f7",
    "entroguard_cnn(native)": "#eab308",
    "static": "#94a3b8",
    "adaptive": "#ef4444",
}
FALLBACK_COLORS = ["#3b82f6", "#f97316", "#22c55e", "#a855f7", "#ef4444", "#14b8a6", "#eab308"]


def _color_for(name, idx):
    return PALETTE.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _esc(s):
    return html.escape(str(s))


def _nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    step = span / n
    # округляем шаг до "красивого" порядка величины
    mag = 10 ** (len(str(int(step))) - 1) if step >= 1 else 10 ** -(len(str(step).split(".")[1]) if "." in str(step) else 1)
    for m in (1, 2, 5, 10):
        if step <= m * mag:
            step = m * mag
            break
    ticks = []
    t = 0
    while t < hi:
        if t >= lo - step:
            ticks.append(round(t, 6))
        t += step
    if not ticks:
        ticks = [lo, hi]
    return ticks


def line_chart(series, title, width=640, height=280, y_fmt="{:.3f}", y_label=""):
    """series: dict[name] -> list[float], все серии используют x = 1..N (эпоха)."""
    series = {k: v for k, v in series.items() if v}
    if not series:
        return f'<div class="chart-empty">{_esc(title)}: нет данных</div>'

    ml, mr, mt, mb = 46, 16, 34, 34
    n = max(len(v) for v in series.values())
    all_vals = [v for vs in series.values() for v in vs]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.08 or (abs(hi) * 0.1 or 1.0)
    lo, hi = lo - pad, hi + pad

    def X(i):
        return ml + (i / max(n - 1, 1)) * (width - ml - mr)

    def Y(v):
        return height - mb - ((v - lo) / (hi - lo)) * (height - mt - mb)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    parts.append(f'<text x="{width/2}" y="18" text-anchor="middle" class="chart-title">{_esc(title)}</text>')

    for t in _nice_ticks(lo, hi):
        y = Y(t)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{ml-6}" y="{y+3:.1f}" text-anchor="end" class="tick">{y_fmt.format(t)}</text>')

    for i, name in enumerate(series):
        vals = series[name]
        color = _color_for(name, i)
        pts = " ".join(f"{X(j):.1f},{Y(v):.1f}" for j, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for j, v in enumerate(vals):
            parts.append(f'<circle cx="{X(j):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{color}">'
                         f'<title>{_esc(name)} · эпоха {j+1}: {y_fmt.format(v)}</title></circle>')

    parts.append(f'<text x="{ml}" y="{height-6}" class="tick">эпоха 1</text>')
    parts.append(f'<text x="{width-mr}" y="{height-6}" text-anchor="end" class="tick">эпоха {n}</text>')

    legend_x = width - mr
    for i, name in enumerate(series):
        color = _color_for(name, i)
        ly = mt + i * 16
        parts.append(f'<rect x="{legend_x-10}" y="{ly-8}" width="9" height="9" fill="{color}"/>')
        parts.append(f'<text x="{legend_x-14}" y="{ly}" text-anchor="end" class="legend">{_esc(name)}</text>')

    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart(categories, series, title, y_fmt="{:.3f}", log_scale=False,
                      higher_is_better=None, errors=None):
    """categories: list[str] (группы по оси x). series: dict[name] -> list[float]
    выровнен по categories (1+ серия на категорию, рисуются рядом).
    errors: опциональный dict[name] -> list[float] (той же формы, что и
    series) -- усы std-dev, например для мультисидового вывода
    aggregate_results.py."""
    if not categories or not series:
        return f'<div class="chart-empty">{_esc(title)}: нет данных</div>'

    long_labels = max(len(c) for c in categories) > 10
    width = max(560, 130 * len(categories))
    height = 330 if long_labels else 300
    ml, mr, mb = 50, 16, (100 if long_labels else 46)
    mt = 34 + (15 * len(series) if len(series) > 1 else 0)

    all_vals = [v for vs in series.values() for v in vs]
    if log_scale:
        floor = max(1e-4, min(v for v in all_vals if v > 0) * 0.5) if any(v > 0 for v in all_vals) else 1e-4
        disp_vals = [max(v, floor) for v in all_vals]
        import math
        lo, hi = math.log10(floor), math.log10(max(disp_vals))
        hi += (hi - lo) * 0.1 or 1

        def Y(v):
            v = max(v, floor)
            return height - mb - ((math.log10(v) - lo) / (hi - lo)) * (height - mt - mb)
    else:
        hi = max(all_vals) if all_vals else 1.0
        hi = hi * 1.15 if hi > 0 else 1.0
        lo = 0.0

        def Y(v):
            return height - mb - ((v - lo) / (hi - lo)) * (height - mt - mb)

    n_series = len(series)
    group_w = (width - ml - mr) / len(categories)
    bar_w = group_w / (n_series + 1)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    parts.append(f'<text x="{width/2}" y="18" text-anchor="middle" class="chart-title">{_esc(title)}</text>')

    base_y = height - mb
    parts.append(f'<line x1="{ml}" y1="{base_y}" x2="{width-mr}" y2="{base_y}" class="axis"/>')

    for gi, cat in enumerate(categories):
        gx = ml + gi * group_w
        for si, name in enumerate(series):
            v = series[name][gi]
            color = _color_for(name, si)
            bx = gx + (si + 0.5) * bar_w
            y = Y(v)
            bh = base_y - y
            err = errors.get(name, [0] * len(categories))[gi] if errors else 0
            tooltip = f"{_esc(name)} · {_esc(cat)}: {y_fmt.format(v)}"
            if err:
                tooltip += f" ± {y_fmt.format(err)}"
            parts.append(f'<rect x="{bx-bar_w*0.4:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" '
                         f'height="{max(bh,0):.1f}" fill="{color}" rx="2">'
                         f'<title>{tooltip}</title></rect>')
            if err:
                y_hi, y_lo = Y(v + err), Y(max(v - err, 0 if not log_scale else 1e-9))
                parts.append(f'<line x1="{bx:.1f}" y1="{y_hi:.1f}" x2="{bx:.1f}" y2="{y_lo:.1f}" '
                             f'stroke="black" stroke-width="1.2"/>')
                parts.append(f'<line x1="{bx-4:.1f}" y1="{y_hi:.1f}" x2="{bx+4:.1f}" y2="{y_hi:.1f}" '
                             f'stroke="black" stroke-width="1.2"/>')
                parts.append(f'<line x1="{bx-4:.1f}" y1="{y_lo:.1f}" x2="{bx+4:.1f}" y2="{y_lo:.1f}" '
                             f'stroke="black" stroke-width="1.2"/>')
            parts.append(f'<text x="{bx:.1f}" y="{y-4:.1f}" text-anchor="middle" class="bar-val">'
                         f'{y_fmt.format(v)}</text>')
        cx, label_y = gx + group_w / 2, base_y + 14
        if long_labels:
            parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="end" class="tick" '
                         f'transform="rotate(-30 {cx:.1f} {label_y})">{_esc(cat)}</text>')
        else:
            parts.append(f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="tick">{_esc(cat)}</text>')

    if n_series > 1:
        for si, name in enumerate(series):
            color = _color_for(name, si)
            ly = mt + si * 15
            lx = width - mr - 8
            parts.append(f'<rect x="{lx-9}" y="{ly-8}" width="9" height="9" fill="{color}"/>')
            parts.append(f'<text x="{lx-13}" y="{ly}" text-anchor="end" class="legend">{_esc(name)}</text>')

    if higher_is_better is not None:
        note = "выше = лучше" if higher_is_better else "ниже = лучше"
        parts.append(f'<text x="{width-mr}" y="{mt-16}" text-anchor="end" class="tick-note">{_esc(note)}</text>')

    parts.append("</svg>")
    return "".join(parts)
