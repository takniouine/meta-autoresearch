"""
run_meta.py — Entry point for meta-autoresearch.

Usage:
    uv run run_meta.py --goal "find the best LLM architecture for TinyStories"

This script:
1. Reads config.yaml for parameters
2. Resumes from state.json if it exists
3. Runs the loop: analyze → generate_program → run_batch → repeat
4. Stops on convergence or when max_batches is reached
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from logger import load_history, save_analysis
from meta_agent import MetaAgent


# ---------------------------------------------------------------------------
# Configuration and state loading
# ---------------------------------------------------------------------------

def load_config():
    """Read config.yaml and return a dict."""
    path = Path("config.yaml")
    if not path.exists():
        raise FileNotFoundError(
            "config.yaml not found — make sure you run from the project root directory."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(goal):
    """
    Load state.json if it exists AND the goal matches.
    Otherwise create a fresh state to start from scratch.

    state.json is in .gitignore — it stays local.
    """
    path = Path("state.json")
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[run_meta] Warning: could not read state.json ({e}) — starting fresh")
            state = {}
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
        "consecutive_crashes":  0,
        "started_at":           datetime.now().isoformat(timespec="seconds"),
        "last_updated":         datetime.now().isoformat(timespec="seconds"),
    }


def save_state(state):
    """Persist current state to state.json."""
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    Path("state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(goal, config):
    """
    Run the meta-research loop.

    For each batch:
      1. Load the full history
      2. Analyze results (identify patterns)
      3. Generate a new program.md
      4. Run the experiment batch (inner agent)
      5. Save the analysis and update state
      6. Check for convergence
    """
    state = load_state(goal)
    agent = MetaAgent(
        goal=goal,
        model=config.get("model", "qwen2.5:7b"),
        ollama_base_url=config.get("ollama_base_url", "http://localhost:11434/v1"),
    )

    n_exp            = config.get("n_experiments", 5)
    max_batches      = config.get("max_batches", 50)
    threshold        = config.get("convergence_threshold", 3)
    max_crashes      = config.get("max_consecutive_crashes", 3)

    print(f"\n[run_meta] Goal: {goal}")
    print(f"[run_meta] {n_exp} experiments/batch — max {max_batches} batches — stop after {threshold} batches without improvement")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    for batch_num in range(state["batches_done"] + 1, max_batches + 1):

        print(f"\n[run_meta] {'═' * 60}")
        print(f"[run_meta]  BATCH {batch_num} / {max_batches}")
        print(f"[run_meta] {'═' * 60}")

        # 1. Load history
        history = load_history()

        # 2. Analyze
        print(f"[run_meta] Analyzing {history['num_batches']} batch(es) of history...")
        analysis = agent.analyze_results(history)
        print(f"[run_meta] Observations: {str(analysis.get('observations', ''))[:120]}...")
        print(f"[run_meta] Next directions: {analysis.get('next_directions', [])}")

        # 3. Generate program.md
        print(f"[run_meta] Generating program.md...")
        program_content = agent.generate_program(history, analysis)
        print(f"[run_meta] program.md ready ({len(program_content)} chars)")

        # 4. Run the batch
        experiments, batch_id, program_version = agent.run_batch(
            program_content,
            n_experiments=n_exp,
        )

        # 5. Save analysis to history/analysis/
        save_analysis(batch_id, analysis)

        # 6. Compute best val_bpb for this batch
        valid = [e for e in experiments if e["status"] != "crash"]
        batch_best = min((e["val_bpb"] for e in valid), default=None)

        # 7. Update state and check convergence
        if batch_best is not None:
            # Valid batch — reset consecutive crash counter
            state["consecutive_crashes"] = 0
            if state["best_val_bpb"] is None or batch_best < state["best_val_bpb"]:
                print(f"[run_meta] New best val_bpb: {batch_best:.6f}  (was {state['best_val_bpb']})")
                state["best_val_bpb"]         = batch_best
                state["no_improvement_count"] = 0
            else:
                state["no_improvement_count"] += 1
                print(f"[run_meta] No improvement {state['no_improvement_count']}/{threshold}  (best: {state['best_val_bpb']:.6f})")
        else:
            # All experiments crashed — do NOT count as a convergence plateau.
            # The program.md may be bad; that is separate from the model converging.
            state["consecutive_crashes"] = state.get("consecutive_crashes", 0) + 1
            print(f"[run_meta] All experiments crashed ({state['consecutive_crashes']} consecutive crash batch(es))")

        state["batches_done"] = batch_num
        save_state(state)

        # Stop on true convergence (plateau with valid experiments)
        if state["no_improvement_count"] >= threshold:
            print(f"\n[run_meta] Convergence detected — {threshold} batches without improvement.")
            break

        # Stop on too many consecutive crash batches (program.md is unusable)
        if state.get("consecutive_crashes", 0) >= max_crashes:
            print(f"\n[run_meta] Stopping — {max_crashes} consecutive crash batches. Check program.md quality.")
            break

    # -----------------------------------------------------------------------
    # Final summary
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
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="meta-autoresearch — autonomous optimization of LLM research instructions"
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help='Research objective in one sentence. e.g. "find the best LLM for TinyStories"',
    )
    args = parser.parse_args()

    config = load_config()
    run(goal=args.goal, config=config)


if __name__ == "__main__":
    main()
