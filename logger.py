"""
logger.py — Logging and history system for meta-autoresearch.

Responsibilities:
- Save each version of program.md that was tested
- Record the results of each experiment batch
- Record meta-agent analyses
- Provide the full history so the meta-agent can learn from past runs
"""

import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# History directory paths (relative to project root)
# ---------------------------------------------------------------------------

HISTORY_DIR  = Path("history")
PROGRAMS_DIR = HISTORY_DIR / "programs"
RESULTS_DIR  = HISTORY_DIR / "results"
ANALYSIS_DIR = HISTORY_DIR / "analysis"


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def get_next_ids():
    """
    Compute the next available batch_id and program_version.
    Uses max existing ID + 1 (robust to deleted files).

    Returns: (next_batch_id, next_program_version)
    """
    batch_ids = []
    for p in RESULTS_DIR.glob("batch_*.json"):
        try:
            batch_ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_batch_id = max(batch_ids, default=0) + 1

    program_versions = []
    for p in PROGRAMS_DIR.glob("program_v*.md"):
        try:
            program_versions.append(int(p.stem.split("v")[1]))
        except (IndexError, ValueError):
            pass
    next_program_version = max(program_versions, default=0) + 1

    return next_batch_id, next_program_version


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_program(content):
    """
    Save a version of program.md to history/programs/.

    Args:
        content (str): full content of the generated program.md

    Returns:
        version (int): assigned version number (e.g. 1, 2, 3...)
    """
    _, version = get_next_ids()
    filename = PROGRAMS_DIR / f"program_v{version:03d}.md"
    filename.write_text(content, encoding="utf-8")
    print(f"[logger] Saved program version {version} → {filename}")
    return version


def save_results(batch_id, program_version, experiments):
    """
    Save the results of an experiment batch to history/results/.

    Args:
        batch_id         (int)  : batch identifier (e.g. 1, 2, 3...)
        program_version  (int)  : program.md version used for this batch
        experiments      (list) : list of dicts, one per experiment.
            Each dict contains:
                experiment_id    (int)   : experiment number within the batch
                commit           (str)   : short git hash (7 chars)
                val_bpb          (float) : bits per byte achieved (0.0 if crash)
                memory_gb        (float) : VRAM used in GB (0.0 if crash)
                training_seconds (float) : actual training duration
                status           (str)   : "keep", "discard", or "crash"
                description      (str)   : short description of the idea tested
                timestamp_start  (str)   : start time (isoformat)

    Returns:
        filename (Path): path to the created JSON file
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
    Save the meta-agent analysis after a batch to history/analysis/.

    Args:
        batch_id       (int)  : identifier of the analyzed batch
        analysis_data  (dict) : meta-agent analysis result.
            Expected keys:
                observations        (str)  : summary of what was observed
                successful_patterns (list) : ideas that worked well
                failed_patterns     (list) : ideas that did not work
                next_directions     (list) : directions to explore in the next batch

    Returns:
        filename (Path): path to the created JSON file
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
# Load function
# ---------------------------------------------------------------------------

def load_history():
    """
    Load the full history: all results, analyses, and programs.
    Used by the meta-agent to understand what has been tried before.

    Returns a dict with:
        results               (list)       : all batch results (sorted by batch_id)
        analyses              (list)       : all analyses (sorted by batch_id)
        programs              (list)       : all programs (sorted by version)
        num_batches           (int)        : total number of batches completed
        best_val_bpb_overall  (float|None) : best val_bpb across all batches
    """
    results = []
    for path in sorted(RESULTS_DIR.glob("batch_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[logger] Warning: could not load {path}: {e} — skipping")

    analyses = []
    for path in sorted(ANALYSIS_DIR.glob("analysis_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                analyses.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[logger] Warning: could not load {path}: {e} — skipping")

    programs = []
    for path in sorted(PROGRAMS_DIR.glob("program_v*.md")):
        try:
            version = int(path.stem.split("v")[1])
        except (IndexError, ValueError):
            continue
        programs.append({
            "version": version,
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
