"""
meta_agent.py — The brain of meta-autoresearch.

MetaAgent manages the high-level research loop:
1. Analyze results from previous batches
2. Generate a better program.md
3. Run a batch of experiments (via inner_agent)
4. Save results and analysis
5. Repeat
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import openai

from logger import save_program, save_results, save_analysis, load_history, get_next_ids
from inner_agent import run_inner_agent

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:7b"

# Base template: start from the original autoresearch program.md
# Falls back to empty string if not available locally
_autoresearch_program = Path("autoresearch/program.md")
BASE_PROGRAM = _autoresearch_program.read_text(encoding="utf-8") if _autoresearch_program.exists() else ""

# Path where we persist the best train.py found across all batches
BEST_TRAIN_PY = Path("history") / "best_train.py"


# ---------------------------------------------------------------------------
# MetaAgent
# ---------------------------------------------------------------------------

class MetaAgent:
    """
    The level-1 agent: optimizes research instructions (program.md).

    It never trains models directly — it generates the instructions
    that guide the inner agent, which actually runs training.
    """

    def __init__(self, goal):
        """
        Args:
            goal (str): research objective in natural language.
                        e.g. "find the best LLM architecture for TinyStories"
        """
        self.client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.goal = goal
        print(f"[MetaAgent] Initialized. Goal: {self.goal}")

    # -----------------------------------------------------------------------
    # analyze_results — understand what happened
    # -----------------------------------------------------------------------

    def analyze_results(self, history):
        """
        Call the meta-agent LLM to analyze history and identify patterns.

        Args:
            history (dict): returned by load_history()

        Returns a dict with keys:
            observations        (str)  : observed patterns
            successful_patterns (list) : ideas that improved val_bpb
            failed_patterns     (list) : ideas that made things worse or crashed
            next_directions     (list) : directions to explore in the next batch
        """
        # Special case: no history yet
        if not history["results"]:
            return {
                "observations":        "No experiments run yet. Starting from scratch.",
                "successful_patterns": [],
                "failed_patterns":     [],
                "next_directions":     ["establish baseline", "explore learning rates", "try different depths"],
            }

        prompt = f"""You are analyzing results from an autonomous LLM research system.

Goal: {self.goal}

Experiment history (all batches):
{json.dumps(history["results"], indent=2)}

Previous meta-agent analyses:
{json.dumps(history["analyses"], indent=2)}

Best val_bpb achieved so far: {history["best_val_bpb_overall"]}

Analyze the results and return a JSON object with EXACTLY these keys:
- "observations": string — key patterns you see across all experiments
- "successful_patterns": list of strings — ideas that consistently improved val_bpb
- "failed_patterns": list of strings — ideas that made things worse or crashed
- "next_directions": list of strings — unexplored directions to try in the next batch

Return ONLY the JSON object, no markdown, no explanation."""

        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content.strip()

        # Strip ```json ... ``` fences if the model adds them
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"[MetaAgent] Warning: analysis response is not valid JSON. Using raw text.")
            return {
                "observations":        text,
                "successful_patterns": [],
                "failed_patterns":     [],
                "next_directions":     [],
            }

    # -----------------------------------------------------------------------
    # generate_program — write better instructions
    # -----------------------------------------------------------------------

    def generate_program(self, history, analysis):
        """
        Call the meta-agent LLM to generate an improved program.md.

        Args:
            history  (dict): returned by load_history()
            analysis (dict): returned by analyze_results()

        Returns:
            program_content (str): full content of the new program.md
        """
        # Collect all ideas already tried to avoid repetition
        ideas_tried = []
        for batch in history.get("results", []):
            for exp in batch.get("experiments", []):
                ideas_tried.append(exp.get("description", ""))
        ideas_tried = list(set(filter(None, ideas_tried)))

        prompt = f"""You are generating research instructions for an autonomous LLM training agent.

Goal: {self.goal}

Analysis of all previous experiments:
{json.dumps(analysis, indent=2)}

Ideas already tried — DO NOT repeat these:
{json.dumps(ideas_tried, indent=2)}

Best val_bpb achieved so far: {history.get("best_val_bpb_overall", "None — no experiments yet")}

Your task: generate a complete program.md that will guide the research agent in the next batch.

Rules for the new program.md:
1. Keep the exact same section structure (Setup, Experimentation, Output format, Logging results, The experiment loop)
2. Focus on unexplored directions from the analysis
3. Be specific about which hyperparameters or architectures to prioritize
4. Keep all constraints (no modifying prepare.py, no new packages, 5-minute budget)
5. Do not suggest ideas already in the "failed_patterns" list

Base template to adapt (keep the structure, update the research strategy):
{BASE_PROGRAM}

Return ONLY the program.md content, starting with '# autoresearch'. No other text."""

        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    # -----------------------------------------------------------------------
    # evaluate_program — score a batch of results
    # -----------------------------------------------------------------------

    def evaluate_program(self, batch_results):
        """
        Score a batch of results to compare program.md versions.

        Args:
            batch_results (dict): one element from history["results"]

        Returns:
            score (float|None): lower is better. None if all experiments crashed.

        Formula:
            score = best_val_bpb × 0.7 + crash_rate × 0.3

        Rationale:
            - best_val_bpb is the primary metric (70% of score)
            - crash_rate penalizes instructions that cause crashes (30%)
            - A stable but slightly worse program.md may be preferred over a
              high-variance one that occasionally crashes
        """
        summary = batch_results.get("summary", {})
        best_val_bpb = summary.get("best_val_bpb")
        crash_rate   = summary.get("crash_rate", 1.0)

        if best_val_bpb is None:
            return None   # All runs crashed — no score

        return round(best_val_bpb * 0.7 + crash_rate * 0.3, 6)

    # -----------------------------------------------------------------------
    # _save_best_train_py — persist the best train.py for cumulative evolution
    # -----------------------------------------------------------------------

    def _save_best_train_py(self, experiments):
        """
        After a batch, extract train.py from the best experiment's git commit
        and save it to history/best_train.py.

        This enables cumulative evolution: each batch starts from the best
        configuration found so far rather than always resetting to the original.
        Inspired by AlphaEvolve (DeepMind, 2025).
        """
        valid = [
            e for e in experiments
            if e["status"] != "crash" and e.get("commit", "unknown") != "unknown"
        ]
        if not valid:
            return

        best_exp = min(valid, key=lambda e: e["val_bpb"])
        commit   = best_exp["commit"]

        result = subprocess.run(
            ["git", "show", f"{commit}:train.py"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            BEST_TRAIN_PY.write_text(result.stdout, encoding="utf-8")
            print(f"[MetaAgent] Saved best train.py → history/best_train.py  (val_bpb={best_exp['val_bpb']:.6f}, commit={commit})")
        else:
            print(f"[MetaAgent] Warning: could not extract train.py from commit {commit}")

    # -----------------------------------------------------------------------
    # run_batch — launch N experiments with a given program.md
    # -----------------------------------------------------------------------

    def run_batch(self, program_content, n_experiments=10):
        """
        Run a batch of experiments with the given program.md.

        Workflow:
            1. Save program.md to history/programs/ and write it to disk
            2. Record the current branch to return to after the batch
            3. Create a dedicated git branch (autoresearch/batch_XXX)
            4. Run the inner agent (LLM with tools)
            5. Parse results from results.tsv
            6. Return to the original branch
            7. Save results to history/results/

        Args:
            program_content (str): program.md content to use
            n_experiments   (int): number of experiments to run

        Returns:
            (experiments, batch_id, program_version)
        """
        batch_id, _      = get_next_ids()
        program_version  = save_program(program_content)
        timestamp_start  = datetime.now().isoformat(timespec="seconds")

        # Write program.md to disk (inner agent reads it via read_file)
        Path("program.md").write_text(program_content, encoding="utf-8")

        # Initialize results.tsv (auto-log will append rows after each training run)
        Path("results.tsv").write_text(
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
            encoding="utf-8"
        )

        print(f"\n[MetaAgent] ═══ Batch {batch_id} — program v{program_version:03d} ═══")

        # Cumulative evolution: apply the best train.py found so far as starting point
        if BEST_TRAIN_PY.exists():
            shutil.copy(BEST_TRAIN_PY, "train.py")
            print(f"[MetaAgent] Applied best train.py from history — cumulative evolution active")
        else:
            print(f"[MetaAgent] No previous best train.py — starting from baseline")

        # Record the current branch to return to after the batch
        current_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        # Create a dedicated branch for this batch (mirrors autoresearch convention)
        branch_name = f"autoresearch/batch_{batch_id:03d}"
        # Delete the branch first if it exists (interrupted run recovery)
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            check=True, capture_output=True,
        )
        print(f"[MetaAgent] Created branch: {branch_name}")

        try:
            # Run the inner agent — blocks until n_experiments done or agent stops
            experiments = run_inner_agent(self.client, program_content, n_experiments)

            # Backfill timestamp_start for each experiment if missing
            for exp in experiments:
                exp.setdefault("timestamp_start", timestamp_start)

        finally:
            # Always return to the original branch, even if an error occurred
            result = subprocess.run(["git", "checkout", current_branch], capture_output=True)
            if result.returncode != 0:
                print(f"[MetaAgent] Warning: failed to return to branch '{current_branch}' — run 'git checkout {current_branch}' manually")
            else:
                print(f"[MetaAgent] Returned to branch: {current_branch}")

        # Save results to history/
        save_results(batch_id, program_version, experiments)

        # Cumulative evolution: persist the best train.py for the next batch
        self._save_best_train_py(experiments)

        score = self.evaluate_program({"summary": {
            "best_val_bpb": min((e["val_bpb"] for e in experiments if e["status"] != "crash"), default=None),
            "crash_rate":   sum(1 for e in experiments if e["status"] == "crash") / max(len(experiments), 1),
        }})

        print(f"[MetaAgent] Batch {batch_id} complete — {len(experiments)} experiments, score={score}")
        return experiments, batch_id, program_version
