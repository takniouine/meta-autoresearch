# meta-autoresearch

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20local-orange.svg)](https://ollama.com)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)

> **meta-autoresearch** builds on top of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by adding a second outer agent that autonomously optimizes the research instructions themselves.

---

## What is this?

### karpathy/autoresearch (the foundation)

[autoresearch](https://github.com/karpathy/autoresearch) is a single-loop autonomous research system:
- An LLM agent reads a `program.md` file (research instructions written by a human)
- It modifies `train.py`, runs 5-minute training experiments, measures `val_bpb`
- It keeps the best modifications and repeats

The research strategy (what to try, in what order) is fixed by the human-written `program.md`.

### meta-autoresearch (this repo)

**meta-autoresearch** adds a second loop on top — a **meta-agent** that reads all past results and rewrites `program.md` itself, making the research strategy smarter over time:

```
Level 0: human writes a one-sentence goal
Level 1: meta-agent reads history → rewrites program.md (research strategy)
Level 2: inner agent reads program.md → modifies train.py → runs experiments
```

The key insight: instead of optimizing *model hyperparameters*, the meta-agent optimizes *the instructions given to the researcher*. This is a form of automatic prompt optimization applied to the research loop itself.

---

## Key Features

- **Two-level autonomy** — meta-agent rewrites research instructions, not just code
- **Full history awareness** — every decision is informed by all past experiments across all batches
- **Crash-resilient** — each experiment is isolated; crashes are logged and skipped gracefully
- **Resume anywhere** — checkpoints every batch; restart from exactly where you stopped
- **Real-time dashboard** — live Flask + Chart.js web UI, auto-refreshes every 15 seconds
- **Git-native** — each batch runs on its own branch; full experiment history preserved
- **Fully local** — runs on a single consumer GPU with no external API dependency

---

## Architecture

| Component | File | Role |
|---|---|---|
| Orchestrator | `run_meta.py` | Parses CLI args, manages the outer loop |
| Meta Agent | `meta_agent.py` | Analyzes results, generates `program.md`, runs batches |
| Inner Agent | `inner_agent.py` | Modifies `train.py`, runs experiments, parses results |
| Logger | `logger.py` | Saves/loads structured history (programs, results, analyses) |
| Dashboard | `dashboard/api.py` | Flask backend serving live stats |
| Config | `config.yaml` | All hyperparameters in one place |

---

## Local Model Choice

This project runs both agents (meta and inner) via **Ollama** using `qwen2.5:7b` locally, rather than a cloud API. This was a deliberate choice for two reasons:

1. **Hardware constraint** — the target machine has an NVIDIA RTX 4050 (6 GB VRAM), which cannot run models above ~7B parameters
2. **Zero cost** — no API key, no credits, fully reproducible offline

The codebase uses the **OpenAI-compatible API** exposed by Ollama (`http://localhost:11434/v1`), so swapping to any other provider (OpenAI, Anthropic, Groq, DeepSeek) requires changing two lines: `OLLAMA_BASE_URL` and `MODEL` in `meta_agent.py` and `inner_agent.py`.

**Trade-off**: `qwen2.5:7b` is significantly less capable than frontier models for complex reasoning and tool use. Expect more crashed experiments and less coherent research directions compared to GPT-4o or Claude Sonnet. For best results on capable hardware, use a larger model (e.g. `qwen2.5:72b`, `llama3.3:70b`, or a cloud API).

---

## Requirements

- Python 3.12+
- CUDA GPU (tested on RTX 4050, 6 GB VRAM, CUDA 12.8)
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- [Ollama](https://ollama.com) — local LLM runtime

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/takniouine/meta-autoresearch
cd meta-autoresearch

# 2. Install uv (if not already installed)
pip install uv

# 3. Install dependencies
uv sync

# 4. Install Ollama and pull the model
# Download Ollama from https://ollama.com
ollama pull qwen2.5:7b

# 5. Download dataset and prepare tokenizer (~3 GB, one-time)
uv run prepare.py
```

---

## Usage

### Run the meta-research loop

```bash
uv run run_meta.py --goal "find the best LLM architecture for TinyStories"
```

The loop will:
1. Analyze all past experiment history
2. Generate new `program.md` research instructions
3. Run a batch of N experiments (each trains for 5 minutes)
4. Save results, update state, check convergence
5. Repeat until `max_batches` or convergence

### Launch the dashboard (optional, separate terminal)

```bash
uv run dashboard/api.py
# Open http://localhost:5000
```

### Full setup (3 terminals)

```
Terminal 1: ollama serve          # local LLM runtime
Terminal 2: uv run dashboard/api.py   # live dashboard at http://localhost:5000
Terminal 3: uv run run_meta.py --goal "..."   # main loop
```

### Configuration (`config.yaml`)

```yaml
n_experiments: 5         # experiments per batch (~25 min/batch)
max_batches: 50          # hard stop
convergence_threshold: 3 # stop after N batches with no improvement
model: "qwen2.5:7b"      # Ollama model name
```

---

## How it works in detail

### Batch lifecycle

```
Batch N
  |- load_history()              <- read all past results
  |- agent.analyze_results()     <- LLM identifies what worked / what failed
  |- agent.generate_program()    <- LLM rewrites program.md
  |- agent.run_batch()
  |     |- git checkout -b autoresearch/batch_N
  |     |- run_inner_agent()     <- inner agent loop (tool use)
  |     |     |- read train.py
  |     |     |- modify train.py  (new architecture idea)
  |     |     |- uv run train.py > run.log 2>&1
  |     |     |- grep val_bpb from run.log
  |     |     +- append to results.tsv
  |     +- git checkout <original branch>
  |- save_analysis()             <- history/analysis/analysis_N.json
  +- save_state()                <- state.json  (resumable)
```

### History layout

```
history/
  programs/    <- every version of program.md (numbered)
  results/     <- one JSON per batch, with all experiments and summary
  analysis/    <- meta-agent reasoning per batch
```

### Convergence

The loop stops when `val_bpb` has not improved for `convergence_threshold` consecutive batches.
Stop at any time with `Ctrl+C` — state is saved after every batch and the next run resumes seamlessly.

### Score formula

Each `program.md` version is scored as:

```
score = best_val_bpb × 0.7 + crash_rate × 0.3
```

Lower is better. The 30% crash rate penalty discourages research directions that are theoretically interesting but practically unstable.

---

## Cost

**Free.** All inference runs locally via Ollama. No API key required.

If you switch to a cloud model, estimated costs per batch (5 experiments):

| Model | Provider | Estimated cost/batch |
|---|---|---|
| qwen2.5:7b | Ollama (local) | $0.00 |
| DeepSeek-V3 | deepseek.com | ~$0.01 |
| Llama 3.3 70B | Groq | ~$0.03 |
| Claude Sonnet 4.6 | Anthropic | ~$0.17 |

---

## Built on

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — inner agent loop, `train.py`, `prepare.py`
- [Ollama](https://ollama.com) — local LLM runtime (OpenAI-compatible API)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — 7B model by Alibaba DAMO Academy
- [Muon optimizer](https://github.com/KellerJordan/modded-nanogpt) — Newton-Schulz orthogonalization
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — efficient attention (replaces Flash Attention 3 for Windows compatibility)

---

## License

MIT — see [LICENSE](LICENSE).

Portions of this project (`prepare.py`, `train.py`) are adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy, used under the MIT License.
