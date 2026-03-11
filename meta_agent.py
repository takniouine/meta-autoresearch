"""
meta_agent.py — Le cerveau de meta-autoresearch.

La classe MetaAgent gère la boucle de recherche de haut niveau :
1. Analyser les résultats des batches précédents
2. Générer un meilleur program.md
3. Lancer un batch d'expériences (via inner_agent)
4. Sauvegarder les résultats et l'analyse
5. Répéter
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import openai

from logger import save_program, save_results, save_analysis, load_history, get_next_ids
from inner_agent import run_inner_agent

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:7b"

# Template de base : on part du program.md original d'autoresearch
# S'il n'est pas disponible localement, on utilise une chaîne vide
_autoresearch_program = Path("autoresearch/program.md")
BASE_PROGRAM = _autoresearch_program.read_text(encoding="utf-8") if _autoresearch_program.exists() else ""


# ---------------------------------------------------------------------------
# MetaAgent
# ---------------------------------------------------------------------------

class MetaAgent:
    """
    L'agent de niveau 1 : optimise les instructions de recherche (program.md).

    Il n'entraîne jamais de modèle directement — il génère les instructions
    qui guident l'inner agent qui, lui, entraîne les modèles.
    """

    def __init__(self, goal):
        """
        goal (str) : objectif en langage naturel.
                     Ex: "find the best LLM architecture for TinyStories"
        """
        self.client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.goal = goal
        print(f"[MetaAgent] Initialized. Goal: {self.goal}")

    # -----------------------------------------------------------------------
    # analyze_results — comprendre ce qui s'est passé
    # -----------------------------------------------------------------------

    def analyze_results(self, history):
        """
        Appelle le meta-agent LLM pour analyser l'historique et identifier les patterns.

        Arguments :
            history (dict) : retourné par load_history()

        Retourne un dict avec les clés :
            observations        (str)  : patterns observés
            successful_patterns (list) : idées qui ont amélioré val_bpb
            failed_patterns     (list) : idées qui ont empiré ou crashé
            next_directions     (list) : directions à explorer au prochain batch
        """
        # Cas spécial : pas encore d'historique
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

        # Retire les balises ```json ... ``` si le modèle les ajoute
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
    # generate_program — écrire de meilleures instructions
    # -----------------------------------------------------------------------

    def generate_program(self, history, analysis):
        """
        Appelle le meta-agent LLM pour générer un nouveau program.md amélioré.

        Arguments :
            history  (dict) : retourné par load_history()
            analysis (dict) : retourné par analyze_results()

        Retourne :
            program_content (str) : contenu complet du nouveau program.md
        """
        # Collecte toutes les idées déjà testées pour éviter les répétitions
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
    # evaluate_program — scorer un batch de résultats
    # -----------------------------------------------------------------------

    def evaluate_program(self, batch_results):
        """
        Score un batch de résultats pour comparer les versions de program.md.

        Arguments :
            batch_results (dict) : un élément de history["results"]

        Retourne :
            score (float|None) : plus bas = meilleur. None si tout a crashé.

        Formule :
            score = best_val_bpb × 0.7 + crash_rate × 0.3

        Pourquoi cette formule ?
            - best_val_bpb est la métrique principale (70% du score)
            - crash_rate pénalise les instructions qui causent des plantages (30%)
            - Un program.md stable mais légèrement moins bon peut être préféré
        """
        summary = batch_results.get("summary", {})
        best_val_bpb = summary.get("best_val_bpb")
        crash_rate   = summary.get("crash_rate", 1.0)

        if best_val_bpb is None:
            return None   # Tous les runs ont crashé — pas de score

        return round(best_val_bpb * 0.7 + crash_rate * 0.3, 6)

    # -----------------------------------------------------------------------
    # run_batch — lancer N expériences avec un program.md donné
    # -----------------------------------------------------------------------

    def run_batch(self, program_content, n_experiments=10):
        """
        Lance un batch d'expériences avec le program.md donné.

        Workflow :
            1. Sauvegarde program.md dans history/programs/ et sur disque
            2. Mémorise la branche courante pour y revenir après
            3. Crée une branche git dédiée (autoresearch/batch_XXX)
            4. Lance l'inner agent (Claude avec outils)
            5. Parse les résultats depuis results.tsv
            6. Revient sur la branche d'origine
            7. Sauvegarde les résultats dans history/results/

        Arguments :
            program_content (str) : contenu du program.md à utiliser
            n_experiments   (int) : nombre d'expériences à effectuer

        Retourne :
            (experiments, batch_id, program_version)
        """
        batch_id, _      = get_next_ids()
        program_version  = save_program(program_content)
        timestamp_start  = datetime.now().isoformat(timespec="seconds")

        # Écrit program.md sur disque (l'inner agent le lira via read_file)
        Path("program.md").write_text(program_content, encoding="utf-8")

        # Initialise results.tsv (l'inner agent y écrira les lignes de résultats)
        Path("results.tsv").write_text(
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
            encoding="utf-8"
        )

        print(f"\n[MetaAgent] ═══ Batch {batch_id} — program v{program_version:03d} ═══")

        # Mémorise la branche courante pour y revenir après le batch
        current_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        # Crée une branche dédiée pour ce batch (comme autoresearch le fait)
        branch_name = f"autoresearch/batch_{batch_id:03d}"
        # Supprime la branche si elle existe déjà (run précédent interrompu)
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
            # Lance l'inner agent — bloquant jusqu'à n_experiments ou end_turn
            experiments = run_inner_agent(self.client, program_content, n_experiments)

            # Ajoute le timestamp de début à chaque expérience (si manquant)
            for exp in experiments:
                exp.setdefault("timestamp_start", timestamp_start)

        finally:
            # Revient TOUJOURS sur la branche d'origine, même en cas d'erreur
            result = subprocess.run(["git", "checkout", current_branch], capture_output=True)
            if result.returncode != 0:
                print(f"[MetaAgent] Warning: failed to return to branch '{current_branch}' — run 'git checkout {current_branch}' manually")
            else:
                print(f"[MetaAgent] Returned to branch: {current_branch}")

        # Sauvegarde les résultats dans history/
        save_results(batch_id, program_version, experiments)

        score = self.evaluate_program({"summary": {
            "best_val_bpb": min((e["val_bpb"] for e in experiments if e["status"] != "crash"), default=None),
            "crash_rate":   sum(1 for e in experiments if e["status"] == "crash") / max(len(experiments), 1),
        }})

        print(f"[MetaAgent] Batch {batch_id} complete — {len(experiments)} experiments, score={score}")
        return experiments, batch_id, program_version
