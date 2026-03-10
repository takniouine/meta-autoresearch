"""
inner_agent.py — L'agent inner qui exécute les expériences d'entraînement.

Cet agent reçoit un program.md comme instructions et :
1. Modifie train.py avec des idées d'amélioration
2. Lance l'entraînement (5 minutes)
3. Mesure val_bpb
4. Garde si meilleur, discard sinon
5. Répète N fois
"""

import subprocess
from datetime import datetime
from pathlib import Path

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8096
COMMAND_TIMEOUT = 720     # 12 minutes max par commande (5 min training + large buffer)
MAX_OUTPUT_CHARS = 4000   # Limite la sortie renvoyée à Claude (limites de contexte)


# ---------------------------------------------------------------------------
# Définition des outils
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file (e.g. 'train.py', 'run.log', 'results.tsv')"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating it if needed or overwriting it completely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file"
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return its output (stdout + stderr). "
            "For training runs, redirect output to run.log: "
            "'uv run train.py > run.log 2>&1'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    }
]


# ---------------------------------------------------------------------------
# Exécution des outils
# ---------------------------------------------------------------------------

def execute_tool(name, input_data):
    """
    Exécute un outil demandé par Claude et retourne le résultat comme string.
    Capture toutes les erreurs — ne laisse jamais planter la boucle principale.
    """
    try:
        if name == "read_file":
            path = Path(input_data["path"])
            if not path.exists():
                return f"Error: file '{path}' does not exist"
            content = path.read_text(encoding="utf-8")
            if len(content) > MAX_OUTPUT_CHARS:
                return content[:MAX_OUTPUT_CHARS] + f"\n\n... (truncated — {len(content)} total chars)"
            return content

        elif name == "write_file":
            path = Path(input_data["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(input_data["content"], encoding="utf-8")
            return f"OK: wrote {len(input_data['content'])} chars to '{path}'"

        elif name == "run_command":
            command = input_data["command"]
            print(f"    [cmd] $ {command}")
            result = subprocess.run(
                command,
                shell=True,       # Nécessaire pour les redirections (> run.log 2>&1)
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
            )
            output = (result.stdout + result.stderr).strip()
            if not output:
                output = f"(command finished with exit code {result.returncode})"
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
            return output

        else:
            return f"Error: unknown tool '{name}'"

    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {COMMAND_TIMEOUT}s — treated as crash"
    except Exception as e:
        return f"Error executing tool '{name}': {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Boucle tool use principale
# ---------------------------------------------------------------------------

def run_inner_agent(client, program_content, n_experiments):
    """
    Lance l'agent inner avec le program.md donné.
    Tourne jusqu'à n_experiments expériences ou jusqu'à stop_reason == "end_turn".

    Arguments :
        client          (anthropic.Anthropic) : client API partagé avec MetaAgent
        program_content (str)                 : contenu du program.md à utiliser
        n_experiments   (int)                 : nombre d'expériences à effectuer

    Retourne :
        experiments (list[dict]) : liste des expériences parsées depuis results.tsv
    """
    train_py = Path("train.py").read_text(encoding="utf-8")

    # Message initial : donne tout le contexte à l'agent
    initial_message = f"""You are an autonomous LLM research agent. Your goal is to minimize val_bpb.

IMPORTANT SETUP NOTES:
- You are already on a dedicated git branch. Do NOT run 'git checkout -b'.
- results.tsv already has its header row. Append results after each experiment.
- Run exactly {n_experiments} experiments, then stop. Do not ask for confirmation.
- Use 'uv run train.py > run.log 2>&1' to run training (5 minutes).
- Use 'grep "^val_bpb:\\|^peak_vram_mb:" run.log' to extract results.

WINDOWS ENVIRONMENT — CRITICAL:
- Flash Attention 3 (FA3 / the 'kernels' package) is NOT available on Windows. Do NOT import or use it.
- Use torch.nn.functional.scaled_dot_product_attention() for ALL attention operations.
- If train.py imports 'kernels', you MUST replace that attention with F.scaled_dot_product_attention before running.
- Standard PyTorch SDPA supports causal masks and works fine on Windows with CUDA.

=== YOUR RESEARCH INSTRUCTIONS (program.md) ===
{program_content}
=== END INSTRUCTIONS ===

Current train.py for your reference:
```python
{train_py}
```

Begin now. Your first experiment should establish the baseline (run train.py as-is)."""

    messages = [{"role": "user", "content": initial_message}]

    print(f"\n  [inner_agent] Starting — {n_experiments} experiments requested")

    iteration = 0
    while True:
        iteration += 1
        print(f"  [inner_agent] API call #{iteration}...", end=" ", flush=True)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        print(f"stop={response.stop_reason}")

        # Ajoute la réponse complète de Claude à l'historique des messages
        messages.append({"role": "assistant", "content": response.content})

        # Affiche un extrait du texte de réflexion de Claude
        for block in response.content:
            if hasattr(block, "text") and block.text:
                preview = block.text.strip()[:150].replace("\n", " ")
                print(f"    [claude] {preview}...")

        if response.stop_reason == "end_turn":
            print("  [inner_agent] Agent signaled end_turn — done.")
            break

        if response.stop_reason == "max_tokens":
            # La réponse a été tronquée. Si elle contient des tool_use incomplets,
            # on doit fournir des tool_result vides pour garder la conversation valide.
            print("  [inner_agent] Warning: max_tokens hit — recovering...")
            incomplete = [b for b in response.content if b.type == "tool_use"]
            if incomplete:
                tool_results = [
                    {
                        "type":        "tool_result",
                        "tool_use_id": b.id,
                        "content":     "Response was truncated. Please continue from where you left off.",
                    }
                    for b in incomplete
                ]
                messages.append({"role": "user", "content": tool_results})
            else:
                # Pas de tool_use : on relance juste avec une invite de continuation
                messages.append({"role": "user", "content": "Continue."})
            continue

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"    [tool] {block.name}({list(block.input.keys())})")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
            # Renvoie les résultats à Claude pour qu'il continue son raisonnement
            messages.append({"role": "user", "content": tool_results})

    return _parse_results_tsv()


# ---------------------------------------------------------------------------
# Parsing des résultats
# ---------------------------------------------------------------------------

def _parse_results_tsv():
    """
    Parse results.tsv écrit par l'agent inner pendant les expériences.

    Format attendu (tab-separated) :
        commit  val_bpb  memory_gb  status  description

    Retourne une liste de dicts d'expériences compatibles avec logger.save_results().
    """
    tsv_path = Path("results.tsv")
    if not tsv_path.exists():
        print("  [inner_agent] Warning: results.tsv not found after agent run")
        return []

    experiments = []
    lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")

    for i, line in enumerate(lines[1:], 1):  # lines[0] est le header
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        commit, val_bpb_str, memory_gb_str, status, description = parts[:5]
        try:
            experiments.append({
                "experiment_id":    i,
                "commit":           commit.strip(),
                "val_bpb":          float(val_bpb_str.strip()),
                "memory_gb":        float(memory_gb_str.strip()),
                "training_seconds": 300.0,   # Durée fixe (TIME_BUDGET dans prepare.py)
                "status":           status.strip(),
                "description":      description.strip(),
                "timestamp_start":  datetime.now().isoformat(timespec="seconds"),
            })
        except ValueError as e:
            print(f"  [inner_agent] Warning: could not parse TSV line {i}: '{line}' — {e}")

    print(f"  [inner_agent] Parsed {len(experiments)} experiments from results.tsv")
    return experiments
