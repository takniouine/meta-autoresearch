"""
dashboard/api.py — Flask backend for the meta-autoresearch dashboard.

Run with: uv run dashboard/api.py
Access at: http://localhost:5000

Endpoints:
    GET /                → serve index.html
    GET /api/status      → current state (state.json + live results.tsv)
    GET /api/history     → all completed batch results
    GET /api/programs    → all program.md versions with scores
"""

import json
import sys
from pathlib import Path

import yaml
from flask import Flask, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# The dashboard lives in dashboard/ but imports from the project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from logger import load_history

app = Flask(__name__, static_folder=str(Path(__file__).parent))

DASHBOARD_DIR = Path(__file__).parent
STATE_FILE    = ROOT / "state.json"
RESULTS_TSV   = ROOT / "results.tsv"


def _read_live_experiments():
    """
    Read results.tsv (current batch in progress) and return a list of live experiments.
    Called on every /api/status poll so the dashboard updates after each training run.
    """
    if not RESULTS_TSV.exists():
        return []
    experiments = []
    try:
        lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            try:
                experiments.append({
                    "commit":      parts[0],
                    "val_bpb":     float(parts[1]),
                    "memory_gb":   float(parts[2]),
                    "status":      parts[3],
                    "description": parts[4],
                })
            except ValueError:
                continue
    except Exception:
        pass
    return experiments


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the dashboard HTML page."""
    return send_from_directory(str(DASHBOARD_DIR), "index.html")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """
    Return the current state of the meta loop.
    Reads state.json if it exists, otherwise returns an empty state.
    Also includes live_experiments from results.tsv for the current batch.
    """
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = {
            "goal":                 None,
            "batches_done":         0,
            "best_val_bpb":         None,
            "no_improvement_count": 0,
            "started_at":           None,
            "last_updated":         None,
        }

    # Load history for aggregate statistics
    try:
        history = load_history()
    except Exception:
        history = {"num_batches": 0, "results": [], "programs": []}

    # Compute aggregate statistics
    total_experiments = sum(
        len(batch.get("experiments", []))
        for batch in history.get("results", [])
    )
    total_crashes = sum(
        batch.get("summary", {}).get("num_crash", 0)
        for batch in history.get("results", [])
    )
    crash_rate = round(total_crashes / max(total_experiments, 1) * 100, 1)

    try:
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        max_batches = config.get("max_batches", 50)
    except Exception:
        max_batches = 50

    live_experiments = _read_live_experiments()

    return jsonify({
        **state,
        "total_experiments": total_experiments,
        "total_programs":    len(history.get("programs", [])),
        "crash_rate_pct":    crash_rate,
        "max_batches":       max_batches,
        "running":           STATE_FILE.exists(),
        "live_experiments":  live_experiments,
    })


@app.route("/api/history")
def api_history():
    """
    Return the full history of completed batches.
    Used to plot the val_bpb curve and populate the experiments table.
    """
    try:
        history = load_history()
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e), "results": [], "analyses": [], "programs": []})


@app.route("/api/programs")
def api_programs():
    """
    Return all program.md versions with their scores.
    """
    try:
        history = load_history()
        programs = history.get("programs", [])

        # Associate each program version with its best score
        scores = {}
        for batch in history.get("results", []):
            v   = batch.get("program_version")
            bpb = batch.get("summary", {}).get("best_val_bpb")
            cr  = batch.get("summary", {}).get("crash_rate", 1.0)
            if v is not None and bpb is not None:
                score = round(bpb * 0.7 + cr * 0.3, 6)
                if v not in scores or score < scores[v]:
                    scores[v] = score

        result = []
        for p in programs:
            v = p["version"]
            result.append({
                "version":  v,
                "score":    scores.get(v),
                "preview":  p["content"][:200].replace("\n", " "),
            })

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "programs": []})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Dashboard running at http://localhost:5000")
    app.run(port=5000, debug=False)
