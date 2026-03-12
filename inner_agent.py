"""
inner_agent.py — The inner agent that runs training experiments.

This agent receives program.md as instructions and:
1. Modifies train.py with improvement ideas
2. Runs training (5 minutes)
3. Measures val_bpb
4. Keeps if better, discards otherwise
5. Repeats N times
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import openai

MAX_TOKENS = 8096
COMMAND_TIMEOUT = 720     # 12 minutes max per command (5 min training + large buffer)
MAX_OUTPUT_CHARS = 4000   # Limit output returned to the agent (context window constraints)


# ---------------------------------------------------------------------------
# Tool definitions
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
# Auto-logging after each training run
# ---------------------------------------------------------------------------

def _auto_log_from_run_log():
    """
    Extract val_bpb and peak_vram_mb from run.log and append a row to results.tsv.
    Called automatically after any command containing 'train.py'.
    Does nothing if run.log does not exist or contains no val_bpb (crash).
    """
    try:
        log_path = Path("run.log")
        if not log_path.exists():
            return

        log_text = log_path.read_text(encoding="utf-8", errors="replace")

        # Extract val_bpb
        val_bpb = None
        for line in log_text.splitlines():
            if line.startswith("val_bpb:"):
                try:
                    val_bpb = float(line.split(":")[1].strip())
                except ValueError:
                    pass

        if val_bpb is None:
            # Training crashed — log as crash
            _append_tsv_row("unknown", 0.0, 0.0, "crash", "training crashed — no val_bpb in log")
            print("  [auto-log] Training crashed — logged as crash")
            return

        # Extract peak_vram_mb
        memory_gb = 0.0
        for line in log_text.splitlines():
            if line.startswith("peak_vram_mb:"):
                try:
                    memory_gb = round(float(line.split(":")[1].strip()) / 1024, 3)
                except ValueError:
                    pass

        # Get the short git hash
        git_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, encoding="utf-8",
        )
        commit = git_result.stdout.strip() if git_result.returncode == 0 else "unknown"

        # Read already-logged results to determine keep/discard
        tsv_path = Path("results.tsv")
        prev_best = None
        if tsv_path.exists():
            for line in tsv_path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        v = float(parts[1])
                        if prev_best is None or v < prev_best:
                            prev_best = v
                    except ValueError:
                        pass

        status = "keep" if (prev_best is None or val_bpb < prev_best) else "discard"
        _append_tsv_row(commit, val_bpb, memory_gb, status, "auto-logged")
        print(f"  [auto-log] val_bpb={val_bpb} memory_gb={memory_gb} status={status} commit={commit}")

    except Exception as e:
        print(f"  [auto-log] Warning: could not auto-log result: {e}")


def _append_tsv_row(commit, val_bpb, memory_gb, status, description):
    """Append a row to results.tsv (creates the header if the file is empty)."""
    tsv_path = Path("results.tsv")
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        tsv_path.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")
    with open(tsv_path, "a", encoding="utf-8") as f:
        f.write(f"{commit}\t{val_bpb}\t{memory_gb}\t{status}\t{description}\n")


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(name, input_data):
    """
    Execute a tool requested by the inner agent and return the result as a string.
    Catches all errors — never lets exceptions crash the main loop.
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
                shell=True,       # Required for redirections (> run.log 2>&1)
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=COMMAND_TIMEOUT,
            )
            output = (result.stdout + result.stderr).strip()
            if not output:
                output = f"(command finished with exit code {result.returncode})"
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"

            # Auto-log: if this was a training run, extract and log results
            # immediately — without relying on the model to call any logging tool.
            if "train.py" in command and "grep" not in command:
                _auto_log_from_run_log()

            return output

        else:
            return f"Error: unknown tool '{name}'"

    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {COMMAND_TIMEOUT}s — treated as crash"
    except Exception as e:
        return f"Error executing tool '{name}': {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Main tool-use loop
# ---------------------------------------------------------------------------

def run_inner_agent(client, program_content, n_experiments, model="qwen2.5:7b"):
    """
    Run the inner agent with the given program.md.
    Loops until n_experiments are done or finish_reason == "stop".

    Args:
        client          (openai.OpenAI) : Ollama client shared with MetaAgent
        program_content (str)           : program.md content to use
        n_experiments   (int)           : number of experiments to run
        model           (str)           : Ollama model name (from config.yaml)

    Returns:
        experiments (list[dict]): list of experiments parsed from results.tsv
    """
    train_py = Path("train.py").read_text(encoding="utf-8")

    is_evolved = Path("history/best_train.py").exists()
    evolution_note = (
        "IMPORTANT: train.py already contains the BEST configuration found across all previous batches. "
        "Build on top of it — do not reset it to defaults. Your first experiment should run it as-is to confirm the baseline, then improve from there."
        if is_evolved else
        "This is the first batch — train.py is the original baseline."
    )

    initial_message = f"""You are an autonomous LLM research agent. Your goal is to minimize val_bpb.

IMPORTANT SETUP NOTES:
- You are already on a dedicated git branch. Do NOT run 'git checkout -b'.
- Run exactly {n_experiments} experiments, then stop. Do not ask for confirmation.
- Use 'uv run train.py > run.log 2>&1' to run training (5 minutes).
- Results are logged automatically after each training run — do NOT try to write to results.tsv yourself.
- {evolution_note}

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

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                tools=TOOLS,
                messages=messages,
            )
            msg           = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            print(f"stop={finish_reason}")

        except openai.BadRequestError as e:
            # Llama on Groq sometimes emits tool calls in its native format
            # <function=name {"arg": "val"}> instead of the OpenAI format.
            # Groq rejects it with 400 tool_use_failed but includes the
            # failed_generation so we can parse and execute it ourselves.
            body = e.response.json() if hasattr(e, "response") else {}
            failed = body.get("error", {}).get("failed_generation", "")
            match  = re.search(r"<function=(\w+)[(\s]+(\{.*?\})\)?\s*</function>", failed, re.DOTALL)
            if not match:
                raise  # unknown error — re-raise
            tool_name = match.group(1)
            try:
                input_data = json.loads(match.group(2))
            except json.JSONDecodeError:
                raise
            fake_id = f"fallback_{iteration}"
            print(f"stop=tool_calls (fallback from native Llama format)")
            print(f"    [tool] {tool_name}({list(input_data.keys())}) [fallback]")
            result = execute_tool(tool_name, input_data)
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": fake_id, "type": "function",
                 "function": {"name": tool_name, "arguments": json.dumps(input_data)}}
            ]})
            messages.append({"role": "tool", "tool_call_id": fake_id, "content": result})
            continue

        # Print a preview of the agent's reasoning
        if msg.content:
            preview = msg.content.strip()[:150].replace("\n", " ")
            print(f"    [agent] {preview}...")

        # Serialize assistant message for the conversation history
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
            # Response was truncated — recover and continue
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
# Results parsing
# ---------------------------------------------------------------------------

def _parse_results_tsv():
    """
    Parse results.tsv written by auto-log after each training run.

    Expected format (tab-separated):
        commit  val_bpb  memory_gb  status  description

    Returns a list of experiment dicts compatible with logger.save_results().
    """
    tsv_path = Path("results.tsv")
    if not tsv_path.exists():
        print("  [inner_agent] Warning: results.tsv not found after agent run")
        return []

    experiments = []
    lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")

    for i, line in enumerate(lines[1:], 1):  # lines[0] is the header
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
                "training_seconds": 300.0,   # Fixed duration (TIME_BUDGET in prepare.py)
                "status":           status.strip(),
                "description":      description.strip(),
                "timestamp_start":  datetime.now().isoformat(timespec="seconds"),
            })
        except ValueError as e:
            print(f"  [inner_agent] Warning: could not parse TSV line {i}: '{line}' — {e}")

    print(f"  [inner_agent] Parsed {len(experiments)} experiments from results.tsv")
    return experiments
