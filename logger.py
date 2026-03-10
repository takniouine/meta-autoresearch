"""
logger.py — Système de logging et d'historique pour meta-autoresearch.

Responsabilités :
- Sauvegarder chaque version de program.md testée
- Enregistrer les résultats de chaque batch d'expériences
- Enregistrer les analyses du meta-agent
- Fournir l'historique complet pour que le meta-agent puisse apprendre
"""

import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins des dossiers d'historique (relatifs à la racine du projet)
# ---------------------------------------------------------------------------

HISTORY_DIR  = Path("history")
PROGRAMS_DIR = HISTORY_DIR / "programs"
RESULTS_DIR  = HISTORY_DIR / "results"
ANALYSIS_DIR = HISTORY_DIR / "analysis"


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def get_next_ids():
    """
    Calcule le prochain batch_id et program_version disponibles.
    Regarde les fichiers existants et retourne les prochains numéros.

    Retourne : (next_batch_id, next_program_version)
    """
    existing_batches = list(RESULTS_DIR.glob("batch_*.json"))
    next_batch_id = len(existing_batches) + 1

    existing_programs = list(PROGRAMS_DIR.glob("program_v*.md"))
    next_program_version = len(existing_programs) + 1

    return next_batch_id, next_program_version


# ---------------------------------------------------------------------------
# Fonctions de sauvegarde
# ---------------------------------------------------------------------------

def save_program(content):
    """
    Sauvegarde une version de program.md dans history/programs/.

    Arguments :
        content (str) : le contenu complet du program.md généré

    Retourne :
        version (int) : le numéro de version assigné (ex: 1, 2, 3...)
    """
    _, version = get_next_ids()
    filename = PROGRAMS_DIR / f"program_v{version:03d}.md"
    filename.write_text(content, encoding="utf-8")
    print(f"[logger] Saved program version {version} → {filename}")
    return version


def save_results(batch_id, program_version, experiments):
    """
    Sauvegarde les résultats d'un batch d'expériences dans history/results/.

    Arguments :
        batch_id         (int)  : identifiant du batch (ex: 1, 2, 3...)
        program_version  (int)  : version de program.md utilisée pour ce batch
        experiments      (list) : liste de dicts, un par expérience.
            Chaque dict contient :
                experiment_id    (int)   : numéro de l'expérience dans le batch
                commit           (str)   : hash git court (7 chars)
                val_bpb          (float) : bits per byte obtenu (0.0 si crash)
                memory_gb        (float) : VRAM utilisée en GB (0.0 si crash)
                training_seconds (float) : durée réelle d'entraînement
                status           (str)   : "keep", "discard", ou "crash"
                description      (str)   : description courte de l'idée testée
                timestamp_start  (str)   : heure de début (isoformat)

    Retourne :
        filename (Path) : chemin du fichier JSON créé
    """
    valid   = [e for e in experiments if e["status"] != "crash"]
    crashed = [e for e in experiments if e["status"] == "crash"]
    kept    = [e for e in experiments if e["status"] == "keep"]

    best_val_bpb = min(
        (e["val_bpb"] for e in valid),
        default=None,
    )

    data = {
        "batch_id": batch_id,
        "program_version": program_version,
        "timestamp_start": experiments[0].get("timestamp_start", "") if experiments else "",
        "timestamp_end": datetime.now().isoformat(timespec="seconds"),
        "experiments": experiments,
        "summary": {
            "num_experiments": len(experiments),
            "num_keep":        len(kept),
            "num_discard":     len([e for e in experiments if e["status"] == "discard"]),
            "num_crash":       len(crashed),
            "best_val_bpb":    best_val_bpb,
            "crash_rate":      round(len(crashed) / len(experiments), 2) if experiments else 0,
        },
    }

    filename = RESULTS_DIR / f"batch_{batch_id:03d}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[logger] Saved batch {batch_id} results ({len(experiments)} experiments) → {filename}")
    return filename


def save_analysis(batch_id, analysis_data):
    """
    Sauvegarde l'analyse du meta-agent après un batch dans history/analysis/.

    Arguments :
        batch_id       (int)  : identifiant du batch analysé
        analysis_data  (dict) : résultat de l'analyse du meta-agent.
            Clés attendues :
                observations        (str)  : résumé de ce qui a été observé
                successful_patterns (list) : idées qui ont bien marché
                failed_patterns     (list) : idées qui n'ont pas marché
                next_directions     (list) : directions à explorer au prochain batch
                program_version_next (int) : version du prochain program.md

    Retourne :
        filename (Path) : chemin du fichier JSON créé
    """
    data = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **analysis_data,
    }

    filename = ANALYSIS_DIR / f"analysis_{batch_id:03d}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[logger] Saved analysis for batch {batch_id} → {filename}")
    return filename


# ---------------------------------------------------------------------------
# Fonction de lecture
# ---------------------------------------------------------------------------

def load_history():
    """
    Charge l'historique complet : tous les résultats, analyses et programs.
    Utilisé par le meta-agent pour analyser ce qui a été fait.

    Retourne un dict avec :
        results               (list) : tous les batch results (triés par batch_id)
        analyses              (list) : toutes les analyses (triées par batch_id)
        programs              (list) : tous les programs (triés par version)
        num_batches           (int)  : nombre total de batches effectués
        best_val_bpb_overall  (float|None) : meilleur val_bpb sur tous les batches
    """
    results = []
    for path in sorted(RESULTS_DIR.glob("batch_*.json")):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))

    analyses = []
    for path in sorted(ANALYSIS_DIR.glob("analysis_*.json")):
        with open(path, encoding="utf-8") as f:
            analyses.append(json.load(f))

    programs = []
    for path in sorted(PROGRAMS_DIR.glob("program_v*.md")):
        programs.append({
            "version": int(path.stem.split("v")[1]),
            "content": path.read_text(encoding="utf-8"),
        })

    best_val_bpb_overall = min(
        (r["summary"]["best_val_bpb"] for r in results
         if r["summary"]["best_val_bpb"] is not None),
        default=None,
    )

    return {
        "results":              results,
        "analyses":             analyses,
        "programs":             programs,
        "num_batches":          len(results),
        "best_val_bpb_overall": best_val_bpb_overall,
    }
