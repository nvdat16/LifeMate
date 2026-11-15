"""Zero-dependency Trajectory Inspector Dashboard for LifeMate Voice Agent."""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Set up paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = Path(os.environ.get("ZAIC_DATA_ROOT", WORKSPACE_ROOT)).expanduser().resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(DATA_ROOT / "train"))

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LifeMate — Agent Trajectory Inspector</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #38bdf8;
            --accent: #818cf8;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; }
        header { background: #0b1120; border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; }
        .logo { font-size: 1.25rem; font-weight: 700; color: var(--primary); display: flex; align-items: center; gap: 8px; }
        .badge { background: #1e3a8a; color: #93c5fd; font-size: 0.75rem; padding: 3px 8px; border-radius: 9999px; }
        .layout { display: grid; grid-template-columns: 320px 1fr 400px; flex: 1; overflow: hidden; }
        .sidebar { background: #0b1120; border-right: 1px solid var(--border); padding: 16px; overflow-y: auto; }
        .main-pane { padding: 24px; overflow-y: auto; background: var(--bg); }
        .right-pane { background: #0b1120; border-left: 1px solid var(--border); padding: 16px; overflow-y: auto; }
        .sample-item { padding: 10px 14px; border-radius: 8px; margin-bottom: 6px; cursor: pointer; border: 1px solid transparent; background: var(--card-bg); transition: all 0.15s; }
        .sample-item:hover { border-color: var(--primary); }
        .sample-item.active { border-color: var(--primary); background: #1e293b; box-shadow: 0 0 12px rgba(56, 189, 248, 0.2); }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
        .card-header { font-weight: 600; font-size: 0.95rem; color: var(--text-muted); margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
        .timeline-step { position: relative; padding-left: 28px; margin-bottom: 20px; }
        .timeline-step::before { content: ""; position: absolute; left: 8px; top: 8px; bottom: -20px; width: 2px; background: var(--border); }
        .timeline-step:last-child::before { display: none; }
        .timeline-dot { position: absolute; left: 0; top: 4px; width: 18px; height: 18px; border-radius: 50%; background: var(--card-bg); border: 3px solid var(--primary); }
        .step-agent { font-weight: 700; color: var(--primary); font-size: 0.85rem; text-transform: uppercase; }
        .step-thought { background: #0b1120; padding: 10px; border-radius: 6px; margin-top: 6px; font-size: 0.9rem; line-height: 1.4; }
        .step-call { background: #172554; border-left: 3px solid var(--primary); padding: 8px 12px; margin-top: 8px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; }
        .step-obs { background: #064e3b; border-left: 3px solid var(--success); padding: 8px 12px; margin-top: 8px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; }
        pre { background: #0b1120; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; color: #a5f3fc; }
        audio { width: 100%; margin-top: 8px; }
        .status-pill { font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; font-weight: 600; }
        .status-pill.success { background: rgba(52, 211, 153, 0.2); color: var(--success); }
        .status-pill.warning { background: rgba(251, 191, 36, 0.2); color: var(--warning); }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span>🎙️ LifeMate Inspector</span>
            <span class="badge">Zalo AI Challenge 2026</span>
        </div>
        <div style="display:flex; gap:12px; align-items:center;">
            <select id="engineSelect" onchange="selectSample(currentSampleId)" style="background:#1e293b; color:#38bdf8; font-weight:600; padding:6px 12px; border:1px solid #38bdf8; border-radius:6px;">
                <option value="ollama">🤖 LLM: Ollama (qwen2.5:1.5b)</option>
                <option value="mock">⚡ LLM: Fast Mock</option>
            </select>
            <select id="splitSelect" onchange="loadSamples()" style="background:#1e293b; color:#fff; padding:6px 12px; border:1px solid #334155; border-radius:6px;">
                <option value="train">Train Split (1,545 mẫu)</option>
                <option value="public_test">Public Test Split</option>
            </select>
        </div>
    </header>
    <div class="layout">
        <div class="sidebar">
            <h3 style="font-size:0.85rem; text-transform:uppercase; color:var(--text-muted); margin-bottom:12px;">Samples List</h3>
            <div id="sampleList">Đang tải danh sách...</div>
        </div>
        <div class="main-pane">
            <div id="mainContent">
                <p style="color:var(--text-muted);">Vui lòng chọn một sample từ cột bên trái để xem Trajectory chi tiết.</p>
            </div>
        </div>
        <div class="right-pane">
            <h3 style="font-size:0.85rem; text-transform:uppercase; color:var(--text-muted); margin-bottom:12px;">Sandbox State & Submission</h3>
            <div id="rightContent">
                <p style="color:var(--text-muted); font-size:0.85rem;">Thông tin Hash và State sẽ hiển thị ở đây.</p>
            </div>
        </div>
    </div>

    <script>
        let currentSplit = 'train';
        let currentSampleId = null;

        async function loadSamples() {
            currentSplit = document.getElementById('splitSelect').value;
            const res = await fetch(`/api/samples?split=${currentSplit}&limit=50`);
            const data = await res.json();
            const listEl = document.getElementById('sampleList');
            listEl.innerHTML = '';
            data.samples.forEach(s => {
                const item = document.createElement('div');
                item.className = 'sample-item';
                item.id = `item-${s.id}`;
                item.innerHTML = `
                    <div style="font-weight:600; font-size:0.9rem;">${s.id}</div>
                    <div style="font-size:0.75rem; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${s.preview || 'Audio request'}</div>
                `;
                item.onclick = () => selectSample(s.id);
                listEl.appendChild(item);
            });
            if (data.samples.length > 0) {
                selectSample(data.samples[0].id);
            }
        }

        async function selectSample(sampleId) {
            if (!sampleId) return;
            currentSampleId = sampleId;
            const engine = document.getElementById('engineSelect').value;
            document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
            const activeEl = document.getElementById(`item-${sampleId}`);
            if (activeEl) activeEl.classList.add('active');

            const res = await fetch(`/api/sample?split=${currentSplit}&id=${sampleId}&engine=${engine}`);
            const data = await res.json();

            // Render main pane
            const mainEl = document.getElementById('mainContent');
            mainEl.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <span>AUDIO INPUT & CONTEXT</span>
                        <span class="status-pill success">${data.id}</span>
                    </div>
                    ${data.audio_url ? `<audio controls src="${data.audio_url}"></audio>` : ''}
                    <div style="margin-top:10px; font-weight:600; color:var(--primary); font-size:1.05rem;">
                        "${data.transcript || data.audio_text || 'Chưa có transcript'}"
                    </div>
                    <div style="margin-top:8px; font-size:0.8rem; color:var(--text-muted);">
                        🕒 Current time: ${data.context.current_time} | 📍 Location: ${JSON.stringify(data.context.location)}
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span>AGENT EXECUTION TRAJECTORY</span>
                        <span class="badge">${data.trajectory ? data.trajectory.steps.length : 0} Steps</span>
                    </div>
                    <div id="trajectorySteps">
                        ${renderTrajectory(data.trajectory)}
                    </div>
                </div>
            `;

            // Render right pane
            const rightEl = document.getElementById('rightContent');
            rightEl.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <span>SUBMISSION RESULT</span>
                    </div>
                    <div style="font-size:0.8rem; margin-bottom:6px; color:var(--text-muted);">STATE HASH (SHA-256):</div>
                    <div style="font-family:monospace; font-size:0.75rem; color:#fde047; word-break:break-all; background:#0b1120; padding:8px; border-radius:6px;">
                        ${data.state_hash || 'Chưa khởi tạo'}
                    </div>
                    <div style="font-size:0.8rem; margin-top:12px; margin-bottom:6px; color:var(--text-muted);">FINAL TOOL CALLS:</div>
                    <pre>${JSON.stringify(data.tool_calls || [], null, 2)}</pre>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span>INITIAL STATE.JSON</span>
                    </div>
                    <pre>${JSON.stringify(data.initial_state || {}, null, 2)}</pre>
                </div>
            `;
        }

        function renderTrajectory(traj) {
            if (!traj || !traj.steps || traj.steps.length === 0) {
                return '<p style="color:var(--text-muted); font-size:0.85rem;">Không có bước thực thi nào.</p>';
            }
            return traj.steps.map(st => `
                <div class="timeline-step">
                    <div class="timeline-dot"></div>
                    <div class="step-agent">${st.agent_name} <span style="color:var(--text-muted); font-size:0.75rem;">Step #${st.step_index}</span></div>
                    <div class="step-thought">${st.thought}</div>
                    ${st.action ? `<div class="step-call">⚡ Action: <b>${st.action.name}</b>(${JSON.stringify(st.action.arguments)})</div>` : ''}
                    ${st.observation ? `<div class="step-obs">👁️ Observation: ${JSON.stringify(st.observation.result)}</div>` : ''}
                </div>
            `).join('');
        }

        window.onload = loadSamples;
    </script>
</body>
</html>
"""

class LifeMateUIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif url.path == "/api/samples":
            query = parse_qs(url.query)
            split = query.get("split", ["train"])[0]
            limit = int(query.get("limit", [50])[0])

            jsonl_file = DATA_ROOT / split / f"{split}.jsonl"
            samples = []
            if jsonl_file.is_file():
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= limit:
                            break
                        if line.strip():
                            d = json.loads(line)
                            samples.append({
                                "id": d["id"],
                                "preview": d.get("audio_text", "Sample audio request")
                            })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"samples": samples}).encode("utf-8"))
            return

        elif url.path == "/api/sample":
            query = parse_qs(url.query)
            split = query.get("split", ["train"])[0]
            sample_id = query.get("id", [""])[0]
            engine_choice = query.get("engine", ["ollama"])[0]

            # Read sample
            from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
            from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory
            from lifemate.adapters.asr.mock_asr import MockASREngine
            from lifemate.adapters.llm.mock_llm import MockLLMEngine
            from lifemate.adapters.llm.local_llm_adapter import LocalLLMEngine
            from lifemate.workflow.orchestrator import LifeMatePipeline
            from lifemate.workflow.graph_orchestrator import LifeMateGraphPipeline

            pipeline_choice = query.get("pipeline", ["graph"])[0]

            reader = JsonlDatasetReader(DATA_ROOT)
            sample = reader.get_sample_by_id(split, sample_id)

            catalog_path = DATA_ROOT / split / f"{split}_tools_catalog.json"
            catalog = json.load(open(catalog_path, encoding="utf-8")) if catalog_path.is_file() else {}

            sandbox_factory = ZaicSandboxEnvFactory(DATA_ROOT)

            if engine_choice == "ollama":
                llm_engine = LocalLLMEngine(
                    base_url="http://localhost:11434/v1",
                    model="qwen2.5:1.5b",
                    timeout=20.0
                )
            else:
                llm_engine = MockLLMEngine()

            if pipeline_choice == "classic":
                pipeline = LifeMatePipeline(
                    asr_engine=MockASREngine(),
                    llm_engine=llm_engine,
                    sandbox_factory=sandbox_factory,
                    catalog=catalog
                )
            else:
                pipeline = LifeMateGraphPipeline(
                    asr_engine=MockASREngine(),
                    llm_engine=llm_engine,
                    sandbox_factory=sandbox_factory,
                    catalog=catalog
                )

            result = pipeline.process_sample(sample, split=split)

            # Get initial state
            state_file = DATA_ROOT / split / f"{split}_environment" / sample_id / "state.json"
            initial_state = json.load(open(state_file, encoding="utf-8")) if state_file.is_file() else {}

            # Prepare trajectory dict
            traj_dict = None
            if result.trajectory:
                traj_dict = {
                    "sample_id": result.trajectory.sample_id,
                    "scenario": result.trajectory.scenario.value,
                    "transcript": result.trajectory.transcript,
                    "steps": [
                        {
                            "step_index": s.step_index,
                            "agent_name": s.agent_name,
                            "thought": s.thought,
                            "action": s.action.to_dict() if s.action else None,
                            "guardrail_status": s.guardrail_status,
                            "observation": {"tool_name": s.observation.tool_name, "result": s.observation.result} if s.observation else None
                        }
                        for s in result.trajectory.steps
                    ]
                }

            payload = {
                "id": sample.id,
                "transcript": result.trajectory.transcript if result.trajectory else sample.audio_text,
                "audio_text": sample.audio_text,
                "audio_url": f"/audio/{split}/{sample.id}",
                "context": {
                    "current_time": sample.context.current_time,
                    "location": sample.context.location,
                    "user_profile": sample.context.user_profile
                },
                "state_hash": result.state_hash,
                "tool_calls": result.tool_calls,
                "trajectory": traj_dict,
                "initial_state": initial_state
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            return

        elif url.path.startswith("/audio/"):
            # Stream audio wav file
            parts = url.path.strip("/").split("/")
            if len(parts) >= 3:
                split = parts[1]
                s_id = parts[2]
                wav_path = DATA_ROOT / split / "audio" / s_id / f"{s_id}.wav"
                if wav_path.is_file():
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(wav_path.stat().st_size))
                    self.end_headers()
                    with open(wav_path, "rb") as af:
                        self.wfile.write(af.read())
                    return

        self.send_response(404)
        self.end_headers()

def run_server(port=8501):
    server = HTTPServer(("localhost", port), LifeMateUIHandler)
    print(f"\n=======================================================")
    print(f"🚀 LifeMate Trajectory Inspector UI is running at:")
    print(f"   👉 http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

if __name__ == "__main__":
    run_server()

