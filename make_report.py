"""Рендерит results/*.json + results/history_*.json в понятный человеку
HTML-отчёт: кривые обучения (что происходило во время обучения) и
таблицы + графики сравнения по Экспериментам 1 / 2 / 3 (результаты).

Предпочитает мультисидовые агрегаты (results/<family>_metrics_agg.json,
от aggregate_results.py), если они есть, иначе — сид текущего процесса
(results/<family>_metrics_seed<config.SEED>.json), и в конце — старое имя
файла без суффикса, для обратной совместимости с временами до multi-seed.

Автономное использование (пишет один самодостаточный файл, который можно
открыть двойным кликом, без сервера):

    python make_report.py

Веб-панель app.py вызывает render_body() напрямую, чтобы встроить тот же
контент прямо в свою страницу -- оба используют этот модуль, так что
графики одинаковы что в статическом отчёте, что на сервере.
"""
import html
import json
import os

import config
from viz_svg import grouped_bar_chart, line_chart

REPORT_CSS = """
:root {
  --bg: #0b1220; --panel: #121a2b; --border: #223049; --text: #e5edf7;
  --muted: #93a4bf; --accent: #3b82f6;
}
* { box-sizing: border-box; }
body.report { background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system, Segoe UI, sans-serif; }
.section { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
           padding: 18px 20px; margin: 0 0 20px; }
.section h2 { margin: 0 0 4px; font-size: 17px; }
.section .sub { color: var(--muted); font-size: 12.5px; margin: 0 0 14px; }
.chart-grid { display: flex; flex-wrap: wrap; gap: 16px; }
.chart { background: #0e1526; border-radius: 8px; }
.chart-title { fill: var(--text); font-size: 12px; font-weight: 600; }
.grid { stroke: #223049; stroke-width: 1; }
.axis { stroke: #3a4a68; stroke-width: 1; }
.tick { fill: var(--muted); font-size: 9.5px; }
.tick-note { fill: var(--muted); font-size: 10px; font-style: italic; }
.bar-val { fill: var(--text); font-size: 9px; }
.legend { fill: var(--text); font-size: 10px; }
.chart-empty { color: var(--muted); font-style: italic; padding: 20px; }
.table-scroll { overflow-x: auto; margin-top: 10px; }
table.report-table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.report-table th, table.report-table td { border: 1px solid var(--border); padding: 6px 10px; text-align: right; }
table.report-table th:first-child, table.report-table td:first-child { text-align: left; }
table.report-table th { color: var(--muted); font-weight: 600; background: #0e1526; }
.takeaways { list-style: none; padding: 0; margin: 12px 0 0; }
.takeaways li { padding: 6px 0 6px 22px; position: relative; border-top: 1px solid var(--border); }
.takeaways li:first-child { border-top: none; }
.takeaways li::before { content: "→"; position: absolute; left: 0; color: var(--accent); }
"""


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _esc(s):
    return html.escape(str(s))


def _recall_key(row):
    return next((k for k in row if k.startswith("retrieval_recall")), None)


def _mval(x):
    """Значение метрики независимо от формы: обычный float (один прогон)
    или {"mean","std","n"} (мультисидовый вывод aggregate_results.py)."""
    return x["mean"] if isinstance(x, dict) else x


def _mstd(x):
    return x["std"] if isinstance(x, dict) else 0.0


def _fmt(x, fmt="{:.4f}"):
    if isinstance(x, dict):
        return f"{fmt.format(x['mean'])} ± {fmt.format(x['std'])}"
    return fmt.format(x)


def _load_family(results_dir, family):
    """Возвращает (metrics, mode, seeds). mode: "agg" (словари mean/std),
    "single" (обычные float, известный один сид) или "legacy" (обычные
    float, имя файла до multi-seed, сид неизвестен)."""
    agg = _load(os.path.join(results_dir, f"{family}_metrics_agg.json"))
    if agg:
        return agg["metrics"], "agg", agg["seeds"]
    single = _load(os.path.join(results_dir, f"{family}_metrics_seed{config.SEED}.json"))
    if single:
        return single, "single", [config.SEED]
    legacy = _load(os.path.join(results_dir, f"{family}_metrics.json"))
    if legacy:
        return legacy, "legacy", None
    return None, None, None


def _seed_note(mode, seeds):
    if mode == "agg":
        return f"по {len(seeds)} сидам: {seeds}"
    if mode == "single":
        return f"один прогон, seed={seeds[0]}"
    return "устаревший формат файла (до multi-seed), сид неизвестен"


def render_table(metrics, cols=None):
    if cols is None:
        cols = ["BLEU-2", "ROUGE-1", "EMR"]
        rkey = _recall_key(next(iter(metrics.values())))
        if rkey:
            cols.append(rkey)
        cols.append("ms_per_1k_queries")
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    rows = []
    for name, r in metrics.items():
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("<td>-</td>")
                continue
            fmt = "{:+.4f}" if c.startswith("delta") else ("{:.3g}" if c == "ms_per_1k_queries" else "{:.4f}")
            cells.append(f"<td>{_fmt(v, fmt)}</td>")
        rows.append(f"<tr><td>{_esc(name)}</td>{''.join(cells)}</tr>")
    table = f'<table class="report-table"><tr><th>защита</th>{head}</tr>{"".join(rows)}</table>'
    return f'<div class="table-scroll">{table}</div>'


def render_takeaways(metrics):
    defended = {k: v for k, v in metrics.items() if k != "none"}
    if not defended:
        return ""
    rkey = _recall_key(next(iter(metrics.values())))
    best_privacy = min(defended, key=lambda k: _mval(defended[k]["BLEU-2"]))
    best_utility = max(defended, key=lambda k: _mval(defended[k][rkey])) if rkey else None
    fastest = min(defended, key=lambda k: _mval(defended[k]["ms_per_1k_queries"]))
    items = [
        f"Меньше всего утечки (BLEU-2) даёт <b>{_esc(best_privacy)}</b> "
        f"({_fmt(defended[best_privacy]['BLEU-2'])} против {_fmt(metrics['none']['BLEU-2'])} без защиты).",
    ]
    if best_utility:
        items.append(f"Лучше всего сохраняет полезность retrieval — <b>{_esc(best_utility)}</b> "
                     f"({_fmt(defended[best_utility][rkey], '{:.3f}')}).")
    items.append(f"Быстрее всего применяется — <b>{_esc(fastest)}</b> "
                 f"({_fmt(defended[fastest]['ms_per_1k_queries'], '{:.3g}')} мс/1000 запросов).")
    return '<ul class="takeaways">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def render_family(family, title, subtitle, results_dir):
    metrics, mode, seeds = _load_family(results_dir, family)
    if not metrics:
        return (f'<div class="section"><h2>{_esc(title)}</h2>'
                f'<p class="chart-empty">нет results/{family}_metrics*.json — запустите '
                f'соответствующий evaluate_*.py (см. NOTES.md)</p></div>')

    categories = list(metrics.keys())
    rkey = _recall_key(next(iter(metrics.values())))
    use_errors = mode == "agg"

    bleu = [_mval(metrics[c]["BLEU-2"]) for c in categories]
    rouge = [_mval(metrics[c]["ROUGE-1"]) for c in categories]
    speed = [_mval(metrics[c]["ms_per_1k_queries"]) for c in categories]
    privacy_errors = None
    if use_errors:
        privacy_errors = {"BLEU-2": [_mstd(metrics[c]["BLEU-2"]) for c in categories],
                          "ROUGE-1": [_mstd(metrics[c]["ROUGE-1"]) for c in categories]}

    charts = [grouped_bar_chart(categories, {"BLEU-2": bleu, "ROUGE-1": rouge},
                                "Приватность (утечка)", higher_is_better=False, errors=privacy_errors)]
    if rkey:
        recall = [_mval(metrics[c][rkey]) for c in categories]
        recall_errors = {rkey: [_mstd(metrics[c][rkey]) for c in categories]} if use_errors else None
        charts.append(grouped_bar_chart(categories, {rkey: recall}, "Полезность (retrieval)",
                                        higher_is_better=True, errors=recall_errors))
    charts.append(grouped_bar_chart(categories, {"мс / 1000 запросов": speed}, "Скорость защиты",
                                    log_scale=True, higher_is_better=False))

    return f"""
    <div class="section">
      <h2>{_esc(title)}</h2>
      <p class="sub">{_esc(subtitle)} — {_esc(_seed_note(mode, seeds))}</p>
      <div class="chart-grid">{"".join(charts)}</div>
      {render_table(metrics)}
      {render_takeaways(metrics)}
    </div>"""


def render_exp3(results_dir):
    metrics, mode, seeds = _load_family(results_dir, "exp3")
    if not metrics:
        return ('<div class="section"><h2>Эксперимент 3 — адаптивный атакующий</h2>'
                '<p class="chart-empty">нет results/exp3_metrics*.json — запустите '
                'train_adaptive_attacker.py (по одному разу на защиту) и '
                'evaluate_adaptive.py</p></div>')

    categories = list(metrics.keys())
    static_bleu = [_mval(metrics[c]["static_BLEU-2"]) for c in categories]
    adaptive_bleu = [_mval(metrics[c]["adaptive_BLEU-2"]) for c in categories]
    static_rouge = [_mval(metrics[c]["static_ROUGE-1"]) for c in categories]
    adaptive_rouge = [_mval(metrics[c]["adaptive_ROUGE-1"]) for c in categories]

    charts = [
        grouped_bar_chart(categories, {"static": static_bleu, "adaptive": adaptive_bleu},
                          "BLEU-2: статический vs адаптивный атакующий", higher_is_better=False),
        grouped_bar_chart(categories, {"static": static_rouge, "adaptive": adaptive_rouge},
                          "ROUGE-1: статический vs адаптивный атакующий", higher_is_better=False),
    ]

    biggest = max(categories, key=lambda c: _mval(metrics[c]["delta_BLEU-2"]))
    takeaway = (f'<ul class="takeaways"><li>Сильнее всего теряет защиту под адаптивной атакой — '
               f'<b>{_esc(biggest)}</b> (BLEU-2 растёт на '
               f'{_mval(metrics[biggest]["delta_BLEU-2"]):+.4f} по сравнению со статическим '
               f'атакующим).</li></ul>')

    cols = ["static_BLEU-2", "adaptive_BLEU-2", "delta_BLEU-2",
            "static_ROUGE-1", "adaptive_ROUGE-1", "delta_ROUGE-1"]

    return f"""
    <div class="section">
      <h2>Эксперимент 3 — адаптивный атакующий</h2>
      <p class="sub">Насколько защита проседает против атакующего, дообученного именно под неё
      (transformer) — {_esc(_seed_note(mode, seeds))}</p>
      <div class="chart-grid">{"".join(charts)}</div>
      {render_table(metrics, cols)}
      {takeaway}
    </div>"""


def render_history(results_dir):
    seed = config.SEED
    specs = []
    for arch in ["transformer", "lstm", "cnn"]:
        specs.append((f"history_attacker_{arch}_seed{seed}.json",
                      f"Атакующий ({arch}) — обучение на чистых эмбеддингах",
                      ["train_ce", "loss"], ["BLEU-2", "ROUGE-1"]))
    for target in ["transformer", "lstm", "cnn"]:
        specs.append((f"history_entroguard_{target}_seed{seed}.json",
                      f"EntroGuard vs {target} — обучение генератора",
                      ["sim_loss", "entropy", "ce"], ["BLEU-2", "ROUGE-1"]))
    for defense in ["gaussian", "pgd", "entroguard"]:
        specs.append((f"history_adaptive_transformer_{defense}_seed{seed}.json",
                      f"Адаптивный атакующий vs {defense} — дообучение",
                      ["train_ce"], ["BLEU-2", "ROUGE-1"]))

    blocks = []
    for fname, title, loss_keys, val_keys in specs:
        data = _load(os.path.join(results_dir, fname))
        if not data or not data.get("history"):
            continue
        hist = data["history"]
        loss_series = {k: [row[k] for row in hist] for k in loss_keys if k in hist[0]}
        val_series = {f"val_{k}": [row[k] for row in hist] for k in val_keys if k in hist[0]}
        charts = [
            line_chart(loss_series, f"{title} — loss", y_fmt="{:.3f}"),
            line_chart(val_series, f"{title} — метрики на val", y_fmt="{:.3f}"),
        ]
        blocks.append(f'<div class="chart-grid">{"".join(charts)}</div>')

    if not blocks:
        return ('<div class="section"><h2>Ход обучения</h2>'
                f'<p class="chart-empty">История по эпохам для seed={seed} не найдена — '
                'запустите train_attacker.py / train_entroguard.py / '
                'train_adaptive_attacker.py.</p></div>')

    return (f'<div class="section"><h2>Ход обучения</h2>'
            f'<p class="sub">Метрики по эпохам для текущего seed={seed}, из results/history_*.json'
            f'</p>{"".join(blocks)}</div>')


def render_body(results_dir=None):
    results_dir = results_dir or config.RESULTS_DIR
    body = render_history(results_dir)
    body += render_family("exp1", "Эксперимент 1", "none / gaussian / pgd / entroguard, атакующий — transformer",
                          results_dir)
    for arch in ["lstm", "cnn"]:
        body += render_family(f"exp2_{arch}", f"Эксперимент 2 — перенос на {arch}",
                              f"то же + zero-shot/native entroguard, атакующий — {arch}", results_dir)
    body += render_exp3(results_dir)
    return body


def render_page(results_dir=None, title="EntroGuard — отчёт"):
    body = render_body(results_dir)
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>{title}</title>
<style>{REPORT_CSS} body{{margin:0;padding:24px;max-width:1100px}}</style>
</head><body class="report"><h1>{title}</h1>{body}</body></html>"""


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_page())
    print(f"записал {path} -- откройте в браузере")


if __name__ == "__main__":
    main()
