"""
inner_agent.py — L'agent inner qui exécute les expériences d'entraînement.

Cet agent reçoit un program.md comme instructions et :
1. Modifie train.py avec des idées d'amélioration
2. Lance l'entraînement (5 minutes)
3. Mesure val_bpb
4. Garde si meilleur, discard sinon
5. Répète N fois
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

MODEL = "qwen2.5:7b"
MAX_TOKENS = 8096
COMMAND_TIMEOUT = 720     # 12 minutes max par commande (5 min training + large buffer)
MAX_OUTPUT_CHARS = 4000   # Limite la sortie renvoyée à l'agent (limites de contexte)


# ---------------------------------------------------------------------------
# Définition des outils
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file (e.g. 'train.py', 'run.log', 'results.tsv')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if needed or overwriting it completely.",
            "parameters": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command and return its output (stdout + stderr). "
                "For training runs, redirect output to run.log: "
                "'uv run train.py > run.log 2>&1'"
            ),
            "parameters": {
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
    Tourne jusqu'à n_experiments expériences ou jusqu'à finish_reason == "stop".

    Arguments :
        client          (openai.OpenAI) : client Ollama partagé avec MetaAgent
        program_content (str)           : contenu du program.md à utiliser
        n_experiments   (int)           : nombre d'expériences à effectuer

    Retourne :
        experiments (list[dict]) : liste des expériences parsées depuis results.tsv
    """
    train_py = Path("train.py").read_text(encoding="utf-8")

    initial_message = f"""You are an autonomous LLM research agent. Your goal is to minimize val_bpb.

IMPORTANT SETUP NOTES:
- You are already on a dedicated git branch. Do NOT run 'git checkout -b'.
- Run exactly {n_experiments} experiments, then stop. Do not ask for confirmation.
- Use 'uv run train.py > run.log 2>&1' to run training (5 minutes).
- Use 'grep "^val_bpb:\\|^peak_vram_mb:" run.log' to extract results.

RESULTS LOGGING — THIS IS MANDATORY:
After each experiment, append exactly ONE tab-separated line to 'results.tsv' (NOT results.txt, NOT any other file).
The file already exists with this header: commit<TAB>val_bpb<TAB>memory_gb<TAB>status<TAB>description
Your line must follow this EXACT format (use actual tab characters, not spaces):
  <7-char-git-hash><TAB><val_bpb float><TAB><memory_gb float><TAB><keep|discard|crash><TAB><short description>
Example: abc1234\t1.3648\t2.1\tkeep\tbaseline default architecture
Use run_command with: echo "abc1234\t1.3648\t2.1\tkeep\tbaseline" >> results.tsv
To get the git hash: git rev-parse --short HEAD
To get val_bpb: grep "^val_bpb:" run.log
To get memory_gb: grep "^peak_vram_mb:" run.log  (divide by 1024)

WINDOWS ENVIRONMENT — train.py is already patched for Windows:
- FA3/kernels replaced with F.scaled_dot_product_attention (already done, do NOT re-patch).
- torch.compile disabled (Triton not available on Windows) — do NOT re-enable it.
- DEVICE_BATCH_SIZE=4, TOTAL_BATCH_SIZE=2**13 (tuned for 6GB VRAM) — adjust only if you get OOM.

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

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        print(f"stop={finish_reason}")

        # Affiche un extrait du texte de réflexion de l'agent
        if msg.content:
            preview = msg.content.strip()[:150].replace("\n", " ")
            print(f"    [agent] {preview}...")

        # Sérialise le message assistant pour l'historique
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if finish_reason == "stop":
            print("  [inner_agent] Agent signaled stop — done.")
            break

        if finish_reason == "length":
            # Réponse tronquée — continuer
            print("  [inner_agent] Warning: max_tokens hit — recovering...")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Response was truncated. Please continue from where you left off.",
                    })
            else:
                messages.append({"role": "user", "content": "Continue."})
            continue

        if finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    input_data = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"    [tool] Warning: malformed JSON arguments for {tc.function.name}: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: could not parse tool arguments as JSON: {e}. Please retry with valid JSON.",
                    })
                    continue
                print(f"    [tool] {tc.function.name}({list(input_data.keys())})")
                result = execute_tool(tc.function.name, input_data)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

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
