# meta-autoresearch

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Anthropic](https://img.shields.io/badge/powered%20by-Claude%20Sonnet-orange.svg)](https://anthropic.com)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)

> **meta-autoresearch** builds on top of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by adding a second outer agent that autonomously optimizes the research instructions themselves.

---

## What is this?

Most AI research automation stops at one loop: an agent modifies code, runs experiments, and picks the best result. **meta-autoresearch** adds a second loop on top — a meta-agent that reads all past results and rewrites the *instructions* given to the inner agent, making the research process itself smarter over time.



---

## Key Features

- **Two-level autonomy** — the meta-agent rewrites its own research instructions, not just the code
- **Full history awareness** — every decision is informed by all past experiments across all batches
- **Crash-resilient** — each experiment is isolated; crashes are logged and skipped gracefully
- **Resume anywhere** —  checkpoints every batch; restart from exactly where you stopped
- **Real-time dashboard** — live Flask + Chart.js web UI, auto-refreshes every 15 seconds
- **Git-native** — each batch runs on its own  branch; full history preserved

---

## Architecture

| Component | File | Role |
|---|---|---|
| Orchestrator |  | Parses CLI args, manages the outer loop |
| Meta Agent |  | Analyzes results, generates , runs batches |
| Inner Agent |  | Modifies , runs experiments, parses results |
| Logger |  | Saves/loads structured history (programs, results, analyses) |
| Dashboard |  | Flask backend serving live stats |
| Config |  | All hyperparameters in one place |

---

## Requirements

- Python 3.12+
- CUDA GPU (tested on RTX 4050, 6 GB VRAM, CUDA 12.8)
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- Anthropic API key ([get one here](https://console.anthropic.com))

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

# 4. Configure your API key
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-...

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
2. Generate new  research instructions
3. Run a batch of N experiments (each trains for 5 minutes)
4. Save results, update state, check convergence
5. Repeat until  or convergence

### Launch the dashboard

```bash
uv run dashboard/api.py
# Open http://localhost:5000
```

### Configuration ()

```yaml
n_experiments: 5         # experiments per batch (~25 min/batch)
max_batches: 50          # hard stop
convergence_threshold: 3 # stop after N batches with no improvement
model: "claude-sonnet-4-6"
```

---

## How it works in detail

### Batch lifecycle

```
Batch N
  |- load_history()              <- read all past results
  |- agent.analyze_results()     <- Claude identifies what worked / what failed
  |- agent.generate_program()    <- Claude rewrites program.md
  |- agent.run_batch()
  |     |- git checkout -b autoresearch/batch_N
  |     |- run_inner_agent()     <- inner agent loop (tool use)
  |     |     |- read train.py
  |     |     |- modify train.py  (new architecture idea)
  |     |     |- uv run train.py > run.log 2>&1
  |     |     |- grep val_bpb from run.log
  |     |     +- append to results.tsv
  |     +- git checkout <original branch>
  |- save_analysis()             <- history/analysis/batch_N_*.json
  +- save_state()                <- state.json  (resumable)
```

### History layout

```
history/
  programs/    <- every version of program.md (numbered)
  results/     <- one JSON per batch, with all experiments
  analysis/    <- meta-agent reasoning per batch
```

### Convergence

The loop stops when  has not improved for  consecutive batches.
Stop at any time with  — state is saved after every batch and the next run resumes seamlessly.

---

## Cost estimate

| Setting | Batches | Experiments | Estimated cost |
|---|---|---|---|
| Quick test | 1 | 5 | ~$0.17 |
| Short run | 5 | 25 | ~$0.85 |
| Full run | 20 | 100 | ~$3.40 |

*Estimates based on Claude Sonnet 4.6 pricing (~$0.05/batch meta-agent + ~$0.12/batch inner agent).*

---

## Built on

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — inner agent loop, train.py, prepare.py
- [Anthropic Claude](https://anthropic.com) — both the meta-agent and inner agent
- [Flash Attention 3](https://github.com/Dao-AILab/flash-attention) — fast attention kernels
- [Muon optimizer](https://github.com/KellerJordan/modded-nanogpt) — Newton-Schulz orthogonalization

---

## License

MIT
