"""
run_meta.py — Point d'entrée de meta-autoresearch.

Usage :
    uv run run_meta.py --goal "find the best LLM architecture for TinyStories"

Ce script :
1. Lit config.yaml pour les paramètres
2. Reprend là où on s'est arrêté si state.json existe
3. Lance la boucle : analyze → generate_program → run_batch → repeat
4. S'arrête si convergence détectée ou max_batches atteint
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from logger import load_history, save_analysis
from meta_agent import MetaAgent


# ---------------------------------------------------------------------------
# Chargement de la configuration et de l'état
# ---------------------------------------------------------------------------

def load_config():
    """Lit config.yaml et retourne un dict."""
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(goal):
    """
    Charge state.json si le fichier existe ET si le goal correspond.
    Sinon, crée un état vide pour repartir de zéro.

    state.json est dans .gitignore — il reste local.
    """
    path = Path("state.json")
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("goal") == goal:
            print(f"[run_meta] Resuming from batch {state['batches_done'] + 1}")
            print(f"[run_meta] Best val_bpb so far: {state['best_val_bpb']}")
            return state
        else:
            print(f"[run_meta] New goal detected — starting fresh (old goal: '{state.get('goal')}')")

    return {
        "goal":                 goal,
        "batches_done":         0,
        "best_val_bpb":         None,
        "no_improvement_count": 0,
        "started_at":           datetime.now().isoformat(timespec="seconds"),
        "last_updated":         datetime.now().isoformat(timespec="seconds"),
    }


def save_state(state):
    """Sauvegarde l'état courant dans state.json."""
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    Path("state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def run(goal, config):
    """
    Lance la boucle meta-research.

    Pour chaque batch :
      1. Charger l'historique complet
      2. Analyser les résultats (identifier les patterns)
      3. Générer un nouveau program.md
      4. Lancer le batch d'expériences (inner agent)
      5. Sauvegarder l'analyse et l'état
      6. Vérifier la convergence
    """
    state = load_state(goal)
    agent = MetaAgent(goal=goal)

    n_exp        = config["n_experiments"]
    max_batches  = config["max_batches"]
    threshold    = config["convergence_threshold"]

    print(f"\n[run_meta] Goal: {goal}")
    print(f"[run_meta] {n_exp} experiments/batch — max {max_batches} batches — stop after {threshold} batches without improvement")

    # -----------------------------------------------------------------------
    # Boucle principale
    # -----------------------------------------------------------------------

    for batch_num in range(state["batches_done"] + 1, max_batches + 1):

        print(f"\n[run_meta] {'═' * 60}")
        print(f"[run_meta]  BATCH {batch_num} / {max_batches}")
        print(f"[run_meta] {'═' * 60}")

        # 1. Charger l'historique
        history = load_history()

        # 2. Analyser
        print(f"[run_meta] Analyzing {history['num_batches']} batch(es) of history...")
        analysis = agent.analyze_results(history)
        print(f"[run_meta] Observations: {str(analysis.get('observations', ''))[:120]}...")
        print(f"[run_meta] Next directions: {analysis.get('next_directions', [])}")

        # 3. Générer program.md
        print(f"[run_meta] Generating program.md...")
        program_content = agent.generate_program(history, analysis)
        print(f"[run_meta] program.md ready ({len(program_content)} chars)")

        # 4. Lancer le batch
        experiments, batch_id, program_version = agent.run_batch(
            program_content,
            n_experiments=n_exp,
        )

        # 5. Sauvegarder l'analyse dans history/analysis/
        save_analysis(batch_id, analysis)

        # 6. Calculer le meilleur val_bpb de ce batch
        valid = [e for e in experiments if e["status"] != "crash"]
        batch_best = min((e["val_bpb"] for e in valid), default=None)

        # 7. Mettre à jour l'état et vérifier la convergence
        if batch_best is not None:
            if state["best_val_bpb"] is None or batch_best < state["best_val_bpb"]:
                print(f"[run_meta] New best val_bpb: {batch_best:.6f}  (was {state['best_val_bpb']})")
                state["best_val_bpb"]         = batch_best
                state["no_improvement_count"] = 0
            else:
                state["no_improvement_count"] += 1
                print(f"[run_meta] No improvement {state['no_improvement_count']}/{threshold}  (best: {state['best_val_bpb']:.6f})")
        else:
            state["no_improvement_count"] += 1
            print(f"[run_meta] All experiments crashed — no_improvement_count={state['no_improvement_count']}")

        state["batches_done"] = batch_num
        save_state(state)

        # Convergence : arrêt anticipé
        if state["no_improvement_count"] >= threshold:
            print(f"\n[run_meta] Convergence detected — {threshold} batches without improvement.")
            break

    # -----------------------------------------------------------------------
    # Résumé final
    # -----------------------------------------------------------------------

    history = load_history()
    print(f"\n[run_meta] {'═' * 60}")
    print(f"[run_meta]  DONE")
    print(f"[run_meta] {'═' * 60}")
    print(f"[run_meta]  Batches completed : {state['batches_done']}")
    print(f"[run_meta]  Programs tested   : {len(history['programs'])}")
    print(f"[run_meta]  Best val_bpb      : {state['best_val_bpb']}")
    print(f"[run_meta]  Started           : {state['started_at']}")
    print(f"[run_meta]  Last updated      : {state['last_updated']}")
    print(f"[run_meta] {'═' * 60}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="meta-autoresearch — autonomous optimization of LLM research instructions"
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help='Research objective in one sentence. Ex: "find the best LLM for TinyStories"',
    )
    args = parser.parse_args()

    config = load_config()
    run(goal=args.goal, config=config)


if __name__ == "__main__":
    main()
