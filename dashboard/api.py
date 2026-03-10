"""
dashboard/api.py — Backend Flask du dashboard meta-autoresearch.

Lance avec : uv run dashboard/api.py
Accès sur   : http://localhost:5000

Endpoints :
    GET /                → sert index.html
    GET /api/status      → état courant (state.json)
    GET /api/history     → tous les résultats de batches
    GET /api/programs    → toutes les versions de program.md
"""

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# Le dashboard est dans dashboard/ mais importe depuis la racine du projet
# On ajoute la racine au path Python
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from logger import load_history

app = Flask(__name__, static_folder=str(Path(__file__).parent))

DASHBOARD_DIR = Path(__file__).parent
STATE_FILE    = ROOT / "state.json"


# ---------------------------------------------------------------------------
# Serveur du frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Sert la page HTML du dashboard."""
    return send_from_directory(str(DASHBOARD_DIR), "index.html")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """
    Retourne l'état courant de la boucle meta.
    Lit state.json s'il existe, sinon retourne un état vide.
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

    # Charge l'historique pour les statistiques globales
    try:
        history = load_history()
    except Exception:
        history = {"num_batches": 0, "results": [], "programs": []}

    # Calcule les statistiques agrégées
    total_experiments = sum(
        len(batch.get("experiments", []))
        for batch in history.get("results", [])
    )
    total_crashes = sum(
        batch.get("summary", {}).get("num_crash", 0)
        for batch in history.get("results", [])
    )
    crash_rate = round(total_crashes / max(total_experiments, 1) * 100, 1)

    return jsonify({
        **state,
        "total_experiments": total_experiments,
        "total_programs":    len(history.get("programs", [])),
        "crash_rate_pct":    crash_rate,
        "running":           STATE_FILE.exists(),
    })


@app.route("/api/history")
def api_history():
    """
    Retourne l'historique complet des batches.
    Utilisé pour tracer la courbe val_bpb et remplir la table d'expériences.
    """
    try:
        history = load_history()
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e), "results": [], "analyses": [], "programs": []})


@app.route("/api/programs")
def api_programs():
    """
    Retourne la liste de toutes les versions de program.md avec leurs scores.
    """
    try:
        history = load_history()
        programs = history.get("programs", [])

        # Associe le score de chaque program à sa version
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
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Dashboard running at http://localhost:5000")
    app.run(port=5000, debug=False)
