"""Локальная веб-панель для проекта EntroGuard: запуск скриптов пайплайна
(с выбором источника данных / переопределением гиперпараметров), просмотр
их логов вживую и результатов -- всё из браузера, без терминала. Только
stdlib (http.server + subprocess), для одного пользователя, рассчитана
только на http://localhost.

Использование:
    python app.py            # затем открыть http://localhost:8000
    python app.py --port 8001
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import config
from make_report import REPORT_CSS, render_body

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# "needs" перечисляет БАЗОВЫЕ имена чекпоинтов (без расширения, без
# суффикса сида -- checkpoint_status() сама добавляет "_seed<N>.pt" под
# тот сид, что сейчас введён в поле "Сид" на панели).
SCRIPTS = {
    "download_data": {"file": "download_data.py", "args": [],
                       "label": "0. Скачать данные", "needs": []},
    "train_attacker_transformer": {"file": "train_attacker.py", "args": ["--arch", "transformer"],
                                    "label": "1a. Атакующий: transformer", "needs": []},
    "train_attacker_lstm": {"file": "train_attacker.py", "args": ["--arch", "lstm"],
                             "label": "1b. Атакующий: LSTM", "needs": []},
    "train_attacker_cnn": {"file": "train_attacker.py", "args": ["--arch", "cnn"],
                            "label": "1c. Атакующий: CNN", "needs": []},
    "train_entroguard_transformer": {"file": "train_entroguard.py", "args": ["--target", "transformer"],
                                      "label": "2a. EntroGuard vs transformer", "needs": ["attacker_transformer"]},
    "train_entroguard_lstm": {"file": "train_entroguard.py", "args": ["--target", "lstm"],
                               "label": "2b. EntroGuard vs LSTM", "needs": ["attacker_lstm"]},
    "train_entroguard_cnn": {"file": "train_entroguard.py", "args": ["--target", "cnn"],
                              "label": "2c. EntroGuard vs CNN", "needs": ["attacker_cnn"]},
    "evaluate_defense": {"file": "evaluate_defense.py", "args": [],
                          "label": "3. Эксперимент 1 (transformer)",
                          "needs": ["attacker_transformer", "entroguard_transformer"]},
    "evaluate_transfer": {"file": "evaluate_transfer.py", "args": [],
                           "label": "4. Эксперимент 2 (перенос на LSTM/CNN)",
                           "needs": ["attacker_lstm", "attacker_cnn", "entroguard_transformer",
                                    "entroguard_lstm", "entroguard_cnn"]},
    "train_adaptive_gaussian": {"file": "train_adaptive_attacker.py", "args": ["--defense", "gaussian"],
                                 "label": "5a. Адаптивный атакующий vs gaussian", "needs": ["attacker_transformer"]},
    "train_adaptive_pgd": {"file": "train_adaptive_attacker.py", "args": ["--defense", "pgd"],
                            "label": "5b. Адаптивный атакующий vs pgd", "needs": ["attacker_transformer"]},
    "train_adaptive_entroguard": {"file": "train_adaptive_attacker.py", "args": ["--defense", "entroguard"],
                                   "label": "5c. Адаптивный атакующий vs entroguard",
                                   "needs": ["attacker_transformer", "entroguard_transformer"]},
    "evaluate_adaptive": {"file": "evaluate_adaptive.py", "args": [],
                          "label": "6. Эксперимент 3 (адаптивный атакующий)",
                          "needs": ["attacker_transformer", "entroguard_transformer"]},
    "run_multiseed": {"file": "run_multiseed.py", "args": [],
                       "label": "Прогнать всё по нескольким сидам (см. поле «Сиды»)", "needs": []},
    "aggregate_results": {"file": "aggregate_results.py", "args": [],
                          "label": "Агрегировать результаты по сидам (mean ± std)", "needs": []},
}

ENV_OVERRIDE_KEYS = ["SEED", "SEEDS", "DATA_SOURCE", "ATTACKER_EPOCHS", "ENTROGUARD_EPOCHS",
                     "ADAPTIVE_ATTACKER_EPOCHS", "EPSILON", "ALPHA", "BETA", "GAMMA",
                     "PERSONACHAT_MAX_SENTENCES", "MSMARCO_MAX_QUERIES", "MSMARCO_MAX_PASSAGES"]

CHECKPOINT_BASENAMES = ["attacker_transformer", "attacker_lstm", "attacker_cnn",
                        "entroguard_transformer", "entroguard_lstm", "entroguard_cnn"]

JOBS = {}
LOCK = threading.Lock()


def checkpoint_status(seed):
    ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints")
    return {n: os.path.exists(os.path.join(ckpt_dir, f"{n}_seed{seed}.pt"))
            for n in CHECKPOINT_BASENAMES}


def start_job(key, env_overrides):
    if key not in SCRIPTS:
        raise ValueError(f"неизвестный шаг: {key}")
    with LOCK:
        for j in JOBS.values():
            if j["key"] == key and j["proc"].poll() is None:
                raise RuntimeError(f"{key} уже выполняется (job {j['id']})")

    spec = SCRIPTS[key]
    job_id = uuid.uuid4().hex[:10]
    log_path = os.path.join(RUNS_DIR, f"{job_id}.log")
    env = os.environ.copy()
    for k, v in env_overrides.items():
        if k in ENV_OVERRIDE_KEYS and v not in (None, ""):
            env[f"ENTRO_{k}"] = str(v)

    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    log_file.write(f"$ python {spec['file']} {' '.join(spec['args'])}\n"
                    f"  env overrides: {json.dumps({k: v for k, v in env_overrides.items() if v})}\n\n")
    log_file.flush()
    proc = subprocess.Popen(
        [sys.executable, "-u", spec["file"], *spec["args"]],
        stdout=log_file, stderr=subprocess.STDOUT, cwd=PROJECT_DIR, env=env,
    )
    with LOCK:
        JOBS[job_id] = {"id": job_id, "key": key, "proc": proc, "log_path": log_path,
                        "log_file": log_file, "started": time.time(), "finished": None}
    return job_id


def job_status(job_id):
    j = JOBS.get(job_id)
    if not j:
        return None
    rc = j["proc"].poll()
    if rc is not None and j["finished"] is None:
        j["finished"] = time.time()
        j["log_file"].close()
    return {"id": job_id, "key": j["key"], "running": rc is None, "returncode": rc,
            "started": j["started"], "finished": j["finished"]}


def read_log(job_id, offset):
    j = JOBS.get(job_id)
    if not j:
        return "", 0, False
    path = j["log_path"]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        text = f.read()
        new_offset = f.tell()
    running = j["proc"].poll() is None
    return text, new_offset, running


PAGE_TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>EntroGuard — панель управления</title>
<style>
%(css)s
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; max-width: 1100px; }
.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; flex-wrap: wrap; gap: 10px; }
h1 { margin: 0; font-size: 22px; }
.tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.tab-btn { background: #121a2b; border: 1px solid #223049; color: var(--muted); padding: 8px 16px;
           border-radius: 8px; cursor: pointer; font-size: 13px; }
.tab-btn.active { color: var(--text); border-color: var(--accent); background: #16223a; }
.tab { display: none; }
.tab.active { display: block; }
.config-row { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }
.config-row label { font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 4px; }
.config-row input, .config-row select { background: #0e1526; border: 1px solid var(--border); color: var(--text);
                                          padding: 6px 8px; border-radius: 6px; font-size: 13px; width: 110px; }
.config-row select { width: 150px; }
.job-card { display: flex; flex-direction: column; gap: 8px; padding: 14px 16px; border: 1px solid var(--border);
            border-radius: 8px; margin-bottom: 10px; background: #0e1526; }
.job-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.job-title { font-weight: 600; }
.job-need { font-size: 11px; color: var(--muted); }
.badge { font-size: 11px; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--border); }
.badge.idle { color: var(--muted); }
.badge.running { color: #fbbf24; border-color: #fbbf24; }
.badge.ok { color: #22c55e; border-color: #22c55e; }
.badge.fail { color: #ef4444; border-color: #ef4444; }
button.run { background: var(--accent); color: white; border: none; padding: 7px 16px; border-radius: 6px;
             cursor: pointer; font-size: 13px; }
button.run:disabled { background: #334155; cursor: not-allowed; }
button.stop { background: transparent; color: #ef4444; border: 1px solid #ef4444; padding: 7px 12px;
              border-radius: 6px; cursor: pointer; font-size: 12px; }
pre.log { background: #060a14; color: #cbd5e1; padding: 10px; border-radius: 6px; max-height: 260px;
          overflow: auto; font-size: 11.5px; margin: 0; white-space: pre-wrap; word-break: break-word; }
.hint { color: var(--muted); font-size: 12px; }
#report-refresh { background: transparent; border: 1px solid var(--border); color: var(--text);
                   padding: 6px 12px; border-radius: 6px; cursor: pointer; margin-bottom: 12px; }
</style></head>
<body class="report">
  <div class="topbar">
    <h1>EntroGuard — панель управления</h1>
    <div class="hint">процессы выполняются на этой машине (CUDA, если доступна) — окно можно закрыть, обучение продолжится в фоне</div>
  </div>

  <div class="section">
    <h2>Настройки запуска</h2>
    <p class="sub">Применяются к следующему запуску любого шага (передаются как ENTRO_* переменные окружения)</p>
    <div class="config-row">
      <label>Сид <input id="cfg-SEED" type="number" placeholder="%(default_seed)s"
                        oninput="refreshNeeds()"></label>
      <label>Сиды (для «Прогнать всё») <input id="cfg-SEEDS" type="text" placeholder="1024,7,42"></label>
      <label>Источник данных
        <select id="cfg-DATA_SOURCE">
          <option value="personachat">personachat (реальные)</option>
          <option value="synthetic">synthetic (офлайн)</option>
        </select>
      </label>
      <label>Эпохи attacker <input id="cfg-ATTACKER_EPOCHS" type="number" placeholder="40"></label>
      <label>Эпохи entroguard <input id="cfg-ENTROGUARD_EPOCHS" type="number" placeholder="40"></label>
      <label>Эпохи adaptive attacker <input id="cfg-ADAPTIVE_ATTACKER_EPOCHS" type="number" placeholder="20"></label>
      <label>epsilon <input id="cfg-EPSILON" type="number" step="0.01" placeholder="0.15"></label>
      <label>alpha <input id="cfg-ALPHA" type="number" step="0.5" placeholder="6.0"></label>
      <label>beta <input id="cfg-BETA" type="number" step="0.5" placeholder="1.0"></label>
      <label>gamma <input id="cfg-GAMMA" type="number" step="0.5" placeholder="1.0"></label>
    </div>
    <p class="hint">«Сид» переопределяет ENTRO_SEED для отдельных шагов (и определяет,
      какие чекпоинты ищутся ниже); «Сиды» — только для кнопки «Прогнать всё по
      нескольким сидам».</p>
  </div>

  <div class="tabs">
    <button class="tab-btn active" data-tab="pipeline">Пайплайн</button>
    <button class="tab-btn" data-tab="results">Результаты</button>
  </div>

  <div id="tab-pipeline" class="tab active">
    <div class="section" id="jobs"></div>
  </div>

  <div id="tab-results" class="tab">
    <button id="report-refresh">↻ Обновить результаты</button>
    <div id="report-body">%(report)s</div>
  </div>

<script>
const SCRIPTS = %(scripts_json)s;
const CFG_KEYS = %(cfg_keys_json)s;
const DEFAULT_SEED = %(default_seed)s;
let ACTIVE = {};   // job_key -> {id, offset}

function cfgOverrides() {
  const o = {};
  for (const k of CFG_KEYS) {
    const el = document.getElementById('cfg-' + k);
    if (el && el.value !== '') o[k] = el.value;
  }
  return o;
}

function badge(state, text) {
  return `<span class="badge ${state}">${text}</span>`;
}

function renderJobs() {
  const el = document.getElementById('jobs');
  el.innerHTML = Object.entries(SCRIPTS).map(([key, s]) => `
    <div class="job-card" id="card-${key}">
      <div class="job-head">
        <span class="job-title">${s.label}</span>
        <span id="badge-${key}">${badge('idle','не запущено')}</span>
      </div>
      <div class="job-need">${s.file} ${s.args.join(' ')}</div>
      <div class="job-need" id="need-${key}"></div>
      <div>
        <button class="run" id="run-${key}" onclick="runJob('${key}')">Запустить</button>
        <button class="stop" id="stop-${key}" style="display:none" onclick="stopJob('${key}')">Остановить</button>
      </div>
      <pre class="log" id="log-${key}" style="display:none"></pre>
    </div>`).join('');
  refreshNeeds();
}

async function refreshNeeds() {
  const seedEl = document.getElementById('cfg-SEED');
  const seed = (seedEl && seedEl.value) || DEFAULT_SEED;
  const res = await fetch(`/api/checkpoints?seed=${seed}`);
  const ckpts = await res.json();
  for (const [key, s] of Object.entries(SCRIPTS)) {
    const el = document.getElementById(`need-${key}`);
    if (!el || !s.needs.length) continue;
    el.innerHTML = 'требует: ' + s.needs.map(n =>
      `<span style="color:${ckpts[n] ? '#22c55e' : '#ef4444'}">${n}${ckpts[n] ? ' ✓' : ' ✗'}</span>`
    ).join(', ');
  }
}

async function runJob(key) {
  const res = await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key, env: cfgOverrides()})});
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  ACTIVE[key] = {id: data.job_id, offset: 0};
  document.getElementById(`log-${key}`).style.display = 'block';
  document.getElementById(`log-${key}`).textContent = '';
  document.getElementById(`run-${key}`).disabled = true;
  document.getElementById(`stop-${key}`).style.display = 'inline-block';
  document.getElementById(`badge-${key}`).innerHTML = badge('running', 'выполняется…');
  pollLog(key);
}

async function stopJob(key) {
  const a = ACTIVE[key];
  if (!a) return;
  await fetch('/api/stop', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({job_id: a.id})});
}

async function pollLog(key) {
  const a = ACTIVE[key];
  if (!a) return;
  const res = await fetch(`/api/log?job=${a.id}&offset=${a.offset}`);
  const data = await res.json();
  if (data.text) {
    const pre = document.getElementById(`log-${key}`);
    pre.textContent += data.text;
    pre.scrollTop = pre.scrollHeight;
  }
  a.offset = data.offset;
  if (data.running) {
    setTimeout(() => pollLog(key), 1200);
  } else {
    document.getElementById(`run-${key}`).disabled = false;
    document.getElementById(`stop-${key}`).style.display = 'none';
    const ok = data.returncode === 0;
    document.getElementById(`badge-${key}`).innerHTML = ok ? badge('ok', 'готово') : badge('fail', `ошибка (код ${data.returncode})`);
    delete ACTIVE[key];
    refreshReport();
    refreshNeeds();
  }
}

async function refreshReport() {
  const res = await fetch('/api/report');
  document.getElementById('report-body').innerHTML = await res.text();
}

document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  if (btn.dataset.tab === 'results') refreshReport();
}));
document.getElementById('report-refresh').addEventListener('click', refreshReport);

renderJobs();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # чтобы не шуметь в терминале; логи каждой задачи -- в runs/*.log

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            scripts_meta = {k: {"file": v["file"], "args": v["args"], "label": v["label"], "needs": v["needs"]}
                            for k, v in SCRIPTS.items()}
            page = PAGE_TEMPLATE % {
                "css": REPORT_CSS,
                "report": render_body(),
                "scripts_json": json.dumps(scripts_meta, ensure_ascii=False),
                "cfg_keys_json": json.dumps(ENV_OVERRIDE_KEYS),
                "default_seed": config.SEED,
            }
            self._html(page)
        elif parsed.path == "/api/report":
            self._html(render_body())
        elif parsed.path == "/api/checkpoints":
            qs = parse_qs(parsed.query)
            seed = qs.get("seed", [str(config.SEED)])[0]
            self._json(checkpoint_status(seed))
        elif parsed.path == "/api/log":
            qs = parse_qs(parsed.query)
            job_id = qs.get("job", [""])[0]
            offset = int(qs.get("offset", ["0"])[0])
            text, new_offset, running = read_log(job_id, offset)
            st = job_status(job_id) or {"returncode": None}
            self._json({"text": text, "offset": new_offset, "running": running,
                        "returncode": st.get("returncode")})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if parsed.path == "/api/start":
            try:
                job_id = start_job(body["key"], body.get("env", {}))
                self._json({"job_id": job_id})
            except Exception as e:
                self._json({"error": str(e)}, status=400)
        elif parsed.path == "/api/stop":
            j = JOBS.get(body.get("job_id"))
            if j and j["proc"].poll() is None:
                j["proc"].terminate()
            self._json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1",
                        help="используйте 0.0.0.0, чтобы панель была доступна снаружи контейнера/ВМ")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"панель EntroGuard: http://{args.host}:{args.port}  (Ctrl+C для остановки)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in JOBS.values():
            if j["proc"].poll() is None:
                j["proc"].terminate()


if __name__ == "__main__":
    main()
