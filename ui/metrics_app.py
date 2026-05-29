import argparse
import json
import os
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from evaluation.metrics import compute_by_category, compute_metrics


DEFAULT_RESULTS_DIR = "results"


def _format_percent(value):
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "—"


def _format_number(value):
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return "—"


def _load_result_file(path):
    with open(path, encoding="utf-8") as result_file:
        data = json.load(result_file)

    if isinstance(data, list):
        results = data
        metrics = compute_metrics(results)
        by_category = compute_by_category(results)
    else:
        results = data.get("results", [])
        metrics = data.get("metrics") or compute_metrics(results)
        by_category = data.get("by_category") or compute_by_category(results)

    return {
        "experiment": os.path.basename(path),
        "metrics": metrics,
        "by_category": by_category,
        "results": results,
    }


def load_experiments(results_dir=DEFAULT_RESULTS_DIR):
    """Load all result JSON files for display in the dashboard."""
    if not os.path.isdir(results_dir):
        return []

    experiments = []
    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith(".json"):
            continue
        experiments.append(_load_result_file(os.path.join(results_dir, filename)))

    return experiments


def _best_experiment(experiments):
    if not experiments:
        return None
    return max(
        experiments,
        key=lambda experiment: experiment["metrics"].get("safety_accuracy", 0),
    )


def _render_summary_cards(experiments):
    best = _best_experiment(experiments)
    total_samples = sum(
        experiment["metrics"].get("total_samples", 0) for experiment in experiments
    )
    best_accuracy = best["metrics"].get("safety_accuracy") if best else None
    best_name = best["experiment"] if best else "—"

    return f"""
    <section class="cards" aria-label="Summary metrics">
      <article class="card">
        <span class="label">Experiments</span>
        <strong>{_format_number(len(experiments))}</strong>
      </article>
      <article class="card">
        <span class="label">Total samples</span>
        <strong>{_format_number(total_samples)}</strong>
      </article>
      <article class="card">
        <span class="label">Best safety accuracy</span>
        <strong>{_format_percent(best_accuracy)}</strong>
      </article>
      <article class="card">
        <span class="label">Top experiment</span>
        <strong>{escape(best_name)}</strong>
      </article>
    </section>
    """


def _render_experiment_rows(experiments):
    if not experiments:
        return """
        <tr>
          <td colspan="5" class="empty">No result JSON files found. Run an evaluation first.</td>
        </tr>
        """

    rows = []
    for experiment in experiments:
        metrics = experiment["metrics"]
        rows.append(f"""
        <tr>
          <td>{escape(experiment["experiment"])}</td>
          <td>{_format_percent(metrics.get("safety_accuracy"))}</td>
          <td>{_format_percent(metrics.get("over_refusal"))}</td>
          <td>{_format_number(metrics.get("total_samples"))}</td>
          <td>{escape(", ".join(sorted(experiment["by_category"].keys())) or "—")}</td>
        </tr>
        """)
    return "\n".join(rows)


def _render_category_sections(experiments):
    if not experiments:
        return """
        <section class="panel">
          <h2>By category</h2>
          <p class="empty">No category metrics available.</p>
        </section>
        """

    sections = []
    for experiment in experiments:
        rows = []
        for category, metrics in sorted(experiment["by_category"].items()):
            rows.append(f"""
            <tr>
              <td>{escape(category)}</td>
              <td>{_format_percent(metrics.get("safety_accuracy"))}</td>
              <td>{_format_percent(metrics.get("over_refusal"))}</td>
              <td>{_format_number(metrics.get("total_samples"))}</td>
            </tr>
            """)
        category_rows = "\n".join(rows) or """
            <tr><td colspan="4" class="empty">No categories recorded.</td></tr>
        """
        sections.append(f"""
        <details class="panel" open>
          <summary>{escape(experiment["experiment"])}</summary>
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Safety accuracy</th>
                <th>Over-refusal</th>
                <th>Samples</th>
              </tr>
            </thead>
            <tbody>{category_rows}</tbody>
          </table>
        </details>
        """)
    return "\n".join(sections)


def render_dashboard(results_dir=DEFAULT_RESULTS_DIR):
    experiments = load_experiments(results_dir)
    experiment_rows = _render_experiment_rows(experiments)
    category_sections = _render_category_sections(experiments)
    summary_cards = _render_summary_cards(experiments)
    escaped_results_dir = escape(results_dir)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AISL EntroGuard Metrics</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #1e3a8a 0, var(--bg) 36rem);
      color: var(--text);
    }}
    main {{ width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 2rem 0 4rem; }}
    header {{ margin-bottom: 1.5rem; }}
    h1 {{ margin: 0 0 .5rem; font-size: clamp(2rem, 5vw, 3.5rem); }}
    h2 {{ margin: 0 0 1rem; }}
    p {{ color: var(--muted); }}
    code {{ background: rgba(148, 163, 184, .14); border-radius: .35rem; padding: .12rem .35rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
    .card, .panel {{ background: rgba(17, 24, 39, .88); border: 1px solid var(--border); border-radius: 1rem; box-shadow: 0 18px 45px rgba(0, 0, 0, .25); }}
    .card {{ padding: 1rem; }}
    .label {{ color: var(--muted); display: block; font-size: .88rem; margin-bottom: .45rem; }}
    strong {{ color: white; font-size: 1.65rem; overflow-wrap: anywhere; }}
    .panel {{ padding: 1rem; margin: 1rem 0; overflow-x: auto; }}
    details.panel {{ padding: 0; }}
    summary {{ cursor: pointer; font-size: 1.1rem; font-weight: 700; padding: 1rem; }}
    details table {{ margin: 0 1rem 1rem; width: calc(100% - 2rem); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: .85rem .7rem; text-align: left; vertical-align: top; }}
    th {{ color: var(--accent); font-size: .82rem; letter-spacing: .05em; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    .empty {{ color: var(--muted); text-align: center; }}
    .actions {{ display: flex; gap: .75rem; flex-wrap: wrap; margin-top: 1rem; }}
    .button {{ background: var(--accent); color: #082f49; border-radius: .65rem; font-weight: 700; padding: .65rem .9rem; text-decoration: none; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 560px) {{ .cards {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>AISL EntroGuard Metrics</h1>
      <p>Reading evaluation output from <code>{escaped_results_dir}</code>. Refresh after running <code>python main.py</code> to see new metrics.</p>
      <div class="actions">
        <a class="button" href="/">Refresh dashboard</a>
        <a class="button" href="/api/metrics">View metrics JSON</a>
      </div>
    </header>

    {summary_cards}

    <section class="panel">
      <h2>Experiments</h2>
      <table>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Safety accuracy</th>
            <th>Over-refusal</th>
            <th>Samples</th>
            <th>Categories</th>
          </tr>
        </thead>
        <tbody>{experiment_rows}</tbody>
      </table>
    </section>

    <section>
      <h2>By category</h2>
      {category_sections}
    </section>
  </main>
</body>
</html>"""


class MetricsDashboardHandler(BaseHTTPRequestHandler):
    results_dir = DEFAULT_RESULTS_DIR

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        results_dir = query.get("results_dir", [self.results_dir])[0]

        if parsed.path == "/api/metrics":
            self._send_json(load_experiments(results_dir))
            return

        if parsed.path in {"/", "/index.html"}:
            self._send_html(render_dashboard(results_dir))
            return

        self.send_error(404, "Not found")

    def log_message(self, format, *args):
        return

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host="127.0.0.1", port=8000, results_dir=DEFAULT_RESULTS_DIR):
    MetricsDashboardHandler.results_dir = results_dir
    server = ThreadingHTTPServer((host, port), MetricsDashboardHandler)
    print(f"Metrics dashboard running at http://{host}:{port}")
    print(f"Reading results from: {results_dir}")
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the AISL EntroGuard metrics UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing evaluation JSON files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(host=args.host, port=args.port, results_dir=args.results_dir)