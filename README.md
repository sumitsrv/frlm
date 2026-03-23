# FRLM — Factual Retrieval Language Model

> Separates factual knowledge from linguistic competence in language models.
> Instead of encoding facts in model parameters, facts are stored in an external
> temporal knowledge graph and retrieved via learned dense embeddings.

---

## Motivation

Large language models memorize vast amounts of factual knowledge inside their
parameters during pre-training. This design has three well-known failure modes:
facts go stale as the world changes, the model "hallucinates" plausible but
incorrect statements, and updating even a single fact requires expensive
re-training or fragile post-hoc patching. FRLM starts from a different premise —
**linguistic competence and factual knowledge are separable concerns and should
live in separate systems.** The model keeps its language abilities (syntax,
reasoning, discourse) in learned parameters, while every factual claim is
grounded in an external, append-only temporal knowledge graph that can be
inspected, corrected, and versioned independently of the model weights. A learned
router decides, at each generation step, whether the next token should come from
the model's own vocabulary head or from a retrieval operation against the
knowledge graph. The result is a system whose facts can be updated in real time
without retraining, whose outputs are traceable to explicit source records, and
whose factual accuracy can be measured and audited separately from its language
fluency.

---

> **🚧 Project Status (March 2026):**  
> **Phase 1 — Data Pipeline: Building Knowledge Base**  
> Steps 1–3 complete (corpus downloaded, entities extracted, relations extracted).  
> Currently working on **Step 4 — Populating the Neo4j knowledge graph**.  
> Steps 5–11 have not been started yet.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Domain](#domain)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [The Pipeline — End to End](#the-pipeline--end-to-end)
  - [Data Flow & Step Dependencies](#data-flow--step-dependencies)
  - [Phase 1 — Data Pipeline (Steps 1–6)](#phase-1--data-pipeline-steps-16)
  - [Phase 2 — Training Pipeline (Steps 7–9)](#phase-2--training-pipeline-steps-79)
  - [Phase 3 — Evaluation & Inference (Steps 10–11)](#phase-3--evaluation--inference-steps-1011)
- [Running the Pipeline](#running-the-pipeline)
  - [Full Automated Run](#full-automated-run)
  - [Step-by-Step Execution](#step-by-step-execution)
  - [Resuming from a Step](#resuming-from-a-step)
  - [Dry Run & Status](#dry-run--status)
  - [Cost Estimation](#cost-estimation)
- [Training Phases (Deep Dive)](#training-phases-deep-dive)
- [How This Project Was Built (Master Prompt Phases)](#how-this-project-was-built-master-prompt-phases)
  - [Execution Sequence](#execution-sequence)
  - [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
  - [How to Reproduce or Modify](#how-to-reproduce-or-modify)
  - [Validation Checklist (Section K)](#validation-checklist-section-k)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Docker](#docker)
- [Makefile Reference](#makefile-reference)
- [License](#license)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         FRLM Model                               │
│                                                                  │
│  ┌──────────────────────┐                                        │
│  │   BioMedLM 2.7B      │  Decoder-only transformer backbone    │
│  │   (backbone)          │                                       │
│  └─────────┬────────────┘                                        │
│            │ hidden_states                                        │
│    ┌───────┼───────────────────────┐                             │
│    │       │                       │                             │
│    ▼       ▼                       ▼                             │
│ ┌────────┐ ┌──────────────┐  ┌──────────────┐                   │
│ │ Router │ │  Retrieval   │  │  Generation  │                   │
│ │  Head  │ │    Head      │  │    Head      │                   │
│ │        │ │              │  │              │                   │
│ │sigmoid │ │ ┌──────────┐ │  │ next-token   │                   │
│ │ > 0.5? │ │ │ semantic │ │  │ prediction   │                   │
│ │        │ │ │ granular.│ │  │ (cross-ent.) │                   │
│ │ ret/gen│ │ │ temporal │ │  │              │                   │
│ └───┬────┘ └──────┬─────┘  └──────────────┘                   │
│     │             │                                              │
│     │   retrieval │                                              │
│     │             ▼                                              │
│     │     ┌───────────────┐     ┌───────────────────┐           │
│     │     │  FAISS Index  │◄───►│  Neo4j Temporal KG │           │
│     │     │  (IVF-PQ GPU) │     │  (append-only)     │           │
│     │     └───────────────┘     └───────────────────┘           │
│     │             │                                              │
│     │             ▼                                              │
│     │     Retrieved facts injected into generation context       │
└─────┴────────────────────────────────────────────────────────────┘
         ▲
         │  Frozen SapBERT encoder embeds KG facts
         │  (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
```

| Component          | Details |
|--------------------|---------|
| **Backbone**       | BioMedLM 2.7B (`stanford-crfm/BioMedLM`) — decoder-only transformer |
| **Router Head**    | `Linear(2560→256) → ReLU → Linear(256→1) → Sigmoid` — classifies each step as retrieval or generation |
| **Retrieval Head** | Three sub-heads: **Semantic** (projects into 768-d SapBERT space), **Granularity** (4 levels), **Temporal** (3 modes) |
| **Generation Head**| Standard next-token prediction over vocabulary |
| **Knowledge Graph**| Neo4j with temporal fact schema — facts have `(valid_from, valid_to)` envelopes, append-only, version chains |
| **Vector Index**   | FAISS IVF-PQ on GPU over SapBERT fact embeddings, with 4-level hierarchical indexing |
| **Frozen Encoder** | SapBERT — never trained, only used to embed KG facts for the FAISS index |

---

## Domain

**Biomedical oncology pharmacology.** Corpus: PubMed Central Open Access subset.

---

## Quick Start

```bash
# 1. Install Python dependencies
make setup          # CPU
make setup-gpu      # GPU (CUDA 11.8)

# 2. Install & start Neo4j (requires Docker — see "Neo4j Setup" below)
make setup-neo4j

# 3. Set credentials in config/secrets.properties
cp config/secrets.properties.example config/secrets.properties
# Then edit config/secrets.properties with your actual values:
#   neo4j.password=frlm_dev_password
#   anthropic.api_key=sk-ant-...
#   wandb.api_key=your-key

# 4. Estimate costs before running (Claude API + GPU hours + storage)
make estimate-costs

# 5. Run the full 11-step pipeline
make full-pipeline

# 6. Or run individual stages
make data-pipeline  # Steps 1–6
make train-all      # Steps 7–9
make evaluate       # Step 10
make serve          # Step 11 (FastAPI server)
```

---

## Neo4j Setup

Neo4j is the temporal knowledge graph backend (Steps 4, 5, 10, 11 depend on it).
A setup script is provided with two installation methods:

### Option A: Docker (recommended)

The simplest approach — no Java installation, isolated environment, persistent data.

```bash
# Start Neo4j in Docker (pulls neo4j:5.26.0, creates frlm-neo4j container)
make setup-neo4j

# Or directly:
./scripts/setup_neo4j.sh docker
```

This will:
- Pull the `neo4j:5.26.0` Docker image
- Start a container named `frlm-neo4j` with ports **7474** (HTTP browser) and **7687** (Bolt protocol)
- Persist data to `./neo4j_data/` so your KG survives container restarts
- Set the password from `neo4j.password` in `config/secrets.properties` (default: `frlm_dev_password`)
- Install the APOC plugin (used for batch operations)

After setup:
- **Browser UI:** http://localhost:7474
- **Bolt URI:** `bolt://localhost:7687` (this is what the FRLM code connects to)
- **Username:** `neo4j`
- **Password:** whatever you set in `config/secrets.properties`

### Option B: Native install (apt)

If you prefer a system-level install (requires Java 17+):

```bash
make setup-neo4j-native

# Or directly:
./scripts/setup_neo4j.sh native
```

This adds the official Neo4j apt repository, installs Neo4j Community Edition,
sets the password, and starts the `neo4j` systemd service.

### Managing Neo4j

```bash
make neo4j-status           # Check if Neo4j is running and reachable
make neo4j-stop             # Stop the Neo4j instance

# Docker-specific commands
docker logs frlm-neo4j      # View Neo4j logs
docker start frlm-neo4j     # Restart after a reboot
docker stop frlm-neo4j      # Stop
docker rm frlm-neo4j        # Remove container (data persists in neo4j_data/)
```

### Configuration

The Neo4j connection settings live in `config/default.yaml`, with sensitive
values (URI, username, password) overridden from `config/secrets.properties`:

```yaml
# config/default.yaml — non-sensitive defaults
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "CHANGE_ME"          # overridden by config/secrets.properties
  database: "neo4j"              # Community Edition only supports "neo4j"
```

```properties
# config/secrets.properties — your actual credentials (not committed to git)
neo4j.uri=bolt://localhost:7687
neo4j.username=neo4j
neo4j.password=frlm_dev_password
```

> **Note:** Neo4j Community Edition only supports the default `neo4j` database.
> If you need a separate `frlm` database, you'll need Neo4j Enterprise Edition.

---

## Project Structure

```
frlm/
├── config/
│   ├── default.yaml              # All hyperparameters — the single source of truth
│   ├── config.py                 # Pydantic loader with secrets.properties overrides
│   ├── secrets.properties        # API keys, passwords, Neo4j credentials (git-ignored)
│   ├── secrets.properties.example # Template for secrets.properties
│   └── deepspeed_config.json     # ZeRO Stage 2 config for multi-GPU training
├── data/
│   ├── corpus/                   # Raw PMC papers (XML)
│   ├── processed/                # Parsed text with entity annotations
│   ├── kg/                       # KG export files
│   └── labels/                   # Router training labels (token-level)
├── src/
│   ├── kg/                       # Neo4j client, Pydantic schema, temporal logic
│   ├── embeddings/               # SapBERT encoder, FAISS index, hierarchical indexing
│   ├── extraction/               # SciSpacy NER, Claude relation extraction, PMC corpus loader
│   ├── model/                    # BioMedLM backbone, router/retrieval/generation heads, losses
│   ├── training/                 # Three-phase training pipeline + utilities
│   ├── labeling/                 # Claude API span labeling + quality validation
│   ├── evaluation/               # Retrieval, generation, router, and end-to-end metrics
│   └── inference/                # Full inference pipeline + FastAPI server
├── scripts/                      # Numbered pipeline scripts (01–11) + orchestration
│   ├── 01_download_corpus.py     ─┐
│   ├── 02_extract_entities.py     │  Data Pipeline
│   ├── 03_extract_relations.py    │  (Steps 1–6)
│   ├── 04_populate_kg.py          │
│   ├── 05_build_faiss_index.py    │
│   ├── 06_generate_router_labels.py ─┘
│   ├── 07_train_router.py        ─┐
│   ├── 08_train_retrieval.py      │  Training Pipeline
│   ├── 09_train_joint.py         ─┘  (Steps 7–9)
│   ├── 10_evaluate.py            ── Evaluation (Step 10)
│   ├── 11_run_inference.py       ── Inference  (Step 11)
│   ├── run_full_pipeline.py      # Master orchestrator (runs all 11 steps)
│   └── estimate_costs.py         # Cost estimator (Claude API, GPU, storage)
├── tests/                        # Unit tests for all modules
├── notebooks/                    # Jupyter notebooks for exploration & analysis
├── requirements.txt
├── setup.py
├── Makefile
├── Dockerfile
└── frlm-master-prompt.md         # Master prompt used to generate this project
```

---

## The Pipeline — End to End

The full FRLM pipeline has **11 numbered steps**, grouped into three logical phases.
Each step is a standalone script in `scripts/` and can be run independently or
orchestrated via `scripts/run_full_pipeline.py` (or `make full-pipeline`).

```
 PHASE 1: DATA PIPELINE             PHASE 2: TRAINING         PHASE 3: EVAL & SERVE
 ─────────────────────────           ─────────────────          ──────────────────────
 ┌──────────────────────┐            ┌──────────────┐          ┌──────────────────┐
 │ Step 1               │            │ Step 7       │          │ Step 10          │
 │ Download Corpus      │            │ Train Router │          │ Evaluate         │
 │ (PMC Open Access)    │            │ (freeze BB)  │          │ (all metrics)    │
 └─────────┬────────────┘            └──────┬───────┘          └────────┬─────────┘
           │                                │                           │
           ▼                                ▼                           ▼
 ┌──────────────────────┐            ┌──────────────┐          ┌──────────────────┐
 │ Step 2               │            │ Step 8       │          │ Step 11          │
 │ Extract Entities     │            │ Train Retr.  │          │ Inference /      │
 │ (SciSpacy + UMLS)    │            │ (InfoNCE)    │          │ FastAPI Server   │
 └─────────┬────────────┘            └──────┬───────┘          └──────────────────┘
           │                                │
           ▼                                ▼
 ┌──────────────────────┐            ┌──────────────┐
 │ Step 3               │            │ Step 9       │
 │ Extract Relations    │            │ Joint Train  │
 │ (Claude API) [$$$]   │            │ (combined    │
 └─────────┬────────────┘            │  loss)       │
           │                         └──────────────┘
           ▼
 ┌──────────────────────┐
 │ Step 4               │
 │ Populate Neo4j KG    │
 └─────────┬────────────┘
           │
           ▼
 ┌──────────────────────┐
 │ Step 5               │
 │ Build FAISS Index    │
 │ (SapBERT embeddings) │
 └─────────┬────────────┘
           │
           ▼
 ┌──────────────────────┐
 │ Step 6               │
 │ Generate Router      │
 │ Labels (Claude) [$$$]│
 └──────────────────────┘
```

> **`[$$$]`** = Steps 3 and 6 use the Claude API and incur costs. Run
> `make estimate-costs` first to see projected spend.

### Data Flow & Step Dependencies

Each step reads outputs from previous steps. The orchestrator
(`run_full_pipeline.py`) checks for these outputs and skips steps that are
already complete. The dependency graph:

```
Step 1  ──► data/corpus/*.xml
                │
Step 2  ──► data/processed/{texts, entities}  ◄── reads corpus/
                │
Step 3  ──► data/processed/relations.json     ◄── reads processed/ [$$$]
                │
Step 4  ──► Neo4j populated                   ◄── reads processed/{entities, relations}
                │
Step 5  ──► data/faiss_indices/               ◄── reads from Neo4j
                │
Step 6  ──► data/labels/                      ◄── reads processed/  [$$$]
                │
       ┌────────┘       ┌───────────────────────────┐
       │                │                             │
Step 7  ──► checkpoints/router.pt     ◄── reads labels/
                │
Step 8  ──► checkpoints/retrieval.pt  ◄── reads faiss_indices/ + router checkpoint
                │
Step 9  ──► checkpoints/joint.pt      ◄── reads router + retrieval checkpoints
                │
Step 10 ──► evaluation results        ◄── reads joint checkpoint + faiss + Neo4j
                │
Step 11 ──► FastAPI server / batch    ◄── reads joint checkpoint + faiss + Neo4j
```

> **Note:** Steps 1–6 can be run independently of Steps 7–9. Steps 7–9 are
> sequential — each training phase loads checkpoints from the previous one.
> Steps 10–11 only require the final joint checkpoint plus the KG and FAISS
> index from Steps 4–5.

### Phase 1 — Data Pipeline (Steps 1–6)

These steps build the knowledge infrastructure that the model learns against.

| Step | Script | What it does | Inputs | Outputs |
|------|--------|-------------|--------|---------|
| **1** | `01_download_corpus.py` | Downloads PMC Open Access papers via NCBI E-utilities API | Config (query terms, max papers) | `data/corpus/` (raw XML files) |
| **2** | `02_extract_entities.py` | Runs SciSpacy NER + UMLS linking to identify biomedical entities (drugs, genes, diseases, proteins, etc.) | `data/corpus/` | `data/processed/` (annotated texts + entity lists) |
| **3** | `03_extract_relations.py` | Uses Claude API (`claude-sonnet-4-20250514`) to extract structured biomedical relations from text + entities. Outputs JSON facts with confidence scores. **Costs money.** | `data/processed/` | `data/processed/` (extracted relations) |
| **4** | `04_populate_kg.py` | Loads extracted entities and relations into Neo4j as a temporal knowledge graph. Handles deduplication and version chains. | `data/processed/` | Neo4j database populated |
| **5** | `05_build_faiss_index.py` | Encodes all KG facts with frozen SapBERT encoder, builds 4-level hierarchical FAISS IVF-PQ index. | Neo4j facts | `data/faiss_indices/` |
| **6** | `06_generate_router_labels.py` | Uses Claude API to annotate corpus text at the token level — each token labeled as `factual` or `linguistic`. These become training labels for the router head. **Costs money.** | `data/processed/` | `data/labels/` |

Run the entire data pipeline:
```bash
make data-pipeline
```

### Phase 2 — Training Pipeline (Steps 7–9)

Training happens in **three sequential phases**, each building on the previous checkpoint.

| Step | Script | Training Phase | What trains | What's frozen | Loss | Key metrics |
|------|--------|---------------|-------------|---------------|------|-------------|
| **7** | `07_train_router.py` | **Phase 1: Router Pre-training** | Router head only | Backbone (frozen) | Binary cross-entropy | Accuracy, Precision, Recall, F1 |
| **8** | `08_train_retrieval.py` | **Phase 2: Retrieval Head Training** | Retrieval head (+ optionally backbone) | Router (loaded from Step 7 checkpoint) | InfoNCE contrastive loss (τ=0.07) with hard negatives | P@1, P@5, P@10, MRR |
| **9** | `09_train_joint.py` | **Phase 3: Joint Fine-tuning** | Everything except frozen SapBERT | — | Combined: `L = 1.0·L_router + 2.0·L_retrieval + 1.0·L_generation` | All metrics jointly |

Run all training phases:
```bash
make train-all
```

Or run them individually:
```bash
make train-router      # Phase 1 only
make train-retrieval   # Phase 2 only (requires Phase 1 checkpoint)
make train-joint       # Phase 3 only (requires Phase 1 + 2 checkpoints)
```

### Phase 3 — Evaluation & Inference (Steps 10–11)

| Step | Script | What it does |
|------|--------|-------------|
| **10** | `10_evaluate.py` | Runs the full evaluation suite: retrieval metrics (P@1/5/10, MRR, temporal accuracy), generation metrics (perplexity on non-factual spans), router metrics (accuracy, F1, confusion matrix), and end-to-end comparison against vanilla BioMedLM. |
| **11** | `11_run_inference.py` | Two modes: **batch** inference or **serve** (starts a FastAPI server at `localhost:8000` with `/generate`, `/fact/{id}`, `/entity/{id}/facts`, and `/health` endpoints). |

```bash
make evaluate          # Run evaluation
make serve             # Start FastAPI server
make inference         # Batch inference mode
```

---

## Running the Pipeline

### Full Automated Run

The master orchestrator (`scripts/run_full_pipeline.py`) runs all 11 steps
sequentially. It skips steps whose outputs already exist, prompts for
confirmation before Claude API steps, and logs timing per step.

```bash
# Interactive (prompts before costly steps)
make full-pipeline

# Non-interactive (auto-confirm everything)
python scripts/run_full_pipeline.py --config config/default.yaml --yes
```

### Step-by-Step Execution

Run any individual step directly:

```bash
python scripts/01_download_corpus.py        --config config/default.yaml
python scripts/02_extract_entities.py       --config config/default.yaml
python scripts/03_extract_relations.py      --config config/default.yaml
python scripts/04_populate_kg.py            --config config/default.yaml
python scripts/05_build_faiss_index.py      --config config/default.yaml
python scripts/06_generate_router_labels.py --config config/default.yaml
python scripts/07_train_router.py           --config config/default.yaml
python scripts/08_train_retrieval.py        --config config/default.yaml
python scripts/09_train_joint.py            --config config/default.yaml
python scripts/10_evaluate.py               --config config/default.yaml
python scripts/11_run_inference.py          --config config/default.yaml --mode serve
```

### Resuming from a Step

If the pipeline fails or you stopped it, resume from any step:

```bash
make resume STEP=7          # Resume from step 7 (train router)

# Or directly:
python scripts/run_full_pipeline.py --config config/default.yaml --start-from 7
```

### Dry Run & Status

```bash
make dry-run    # Print what each step would do, without executing
make status     # Show which steps have completed (checks for output files)
```

### Cost Estimation

Before running the pipeline, estimate Claude API, GPU, and storage costs:

```bash
make estimate-costs

# Or with custom document limit:
python scripts/estimate_costs.py --config config/default.yaml --max-documents 5000

# Export as JSON:
python scripts/estimate_costs.py --config config/default.yaml --json costs.json
```

---

## Training Phases (Deep Dive)

### Phase 1 — Router Pre-training (`07_train_router.py`)

**Goal:** Teach the router head to distinguish factual tokens from linguistic tokens.

- **Frozen:** Backbone (BioMedLM)
- **Trainable:** Router head only (`Linear(2560→256) → ReLU → Dropout → Linear(256→1)`)
- **Dataset:** `RouterDataset` — token-level binary labels from Step 6 (Claude annotations)
- **Loss:** Binary cross-entropy (optional class weighting)
- **Early stopping:** On validation F1
- **Output:** Router checkpoint saved to `checkpoints/`

### Phase 2 — Retrieval Head Training (`08_train_retrieval.py`)

**Goal:** Train the retrieval head to project hidden states into the SapBERT embedding space so that the model can "point" at the correct fact in the FAISS index.

- **Frozen:** Router head (loaded from Phase 1 checkpoint)
- **Trainable:** Retrieval head (semantic + granularity + temporal sub-heads), optionally backbone
- **Dataset:** `RetrievalDataset` — factual spans paired with correct fact embeddings + hard negatives mined from FAISS
- **Loss:** InfoNCE contrastive loss (temperature τ=0.07)
- **Curriculum:** Starts with easy (random) negatives, transitions to hard negatives
- **Hard negative refresh:** Re-mines from FAISS every N steps
- **Output:** Retrieval checkpoint saved to `checkpoints/`

### Phase 3 — Joint Fine-tuning (`09_train_joint.py`)

**Goal:** Fine-tune all components together so that router decisions, retrieval accuracy, and generation quality improve jointly.

- **Frozen:** SapBERT encoder only (never trained)
- **Trainable:** Backbone + router head + retrieval head + generation head
- **Dataset:** `JointDataset` — combines router labels, fact embeddings, hard negatives, and token-level generation labels
- **Loss:** `L_total = 1.0 × L_router + 2.0 × L_retrieval + 1.0 × L_generation`
- **Optimizer:** Lower LR (5e-6), linear warmup, gradient clipping (max_norm=1.0)
- **Multi-GPU:** DeepSpeed ZeRO Stage 2 (see `config/deepspeed_config.json`)
- **Output:** Final model checkpoint

---

## How This Project Was Built (Master Prompt Phases)

The entire codebase was generated using `frlm-master-prompt.md` — a modular
master prompt designed to be fed to Claude Opus **one phase at a time** across
multiple conversations. This section explains those code-generation phases and
how they map to the runtime pipeline above.

> **Key distinction:** The master prompt has **9 code-generation phases**
> (Sections B–J) that produce the source code. The resulting code then has
> **11 runtime steps** (scripts `01`–`11`) grouped into the 3 operational
> phases described [above](#the-pipeline--end-to-end). Don't confuse the two.

### Execution Sequence

Every conversation with Claude must start by pasting **Section A (System
Context)** — it gives Claude the full architectural understanding. Then paste
the relevant Phase section. Within each phase, numbered sub-tasks can be sent
as individual prompts if a single phase is too large for one response.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  ALWAYS START EVERY CONVERSATION WITH:                                  │
 │   Section A  — System Context (architecture, libraries, code standards) │
 └────────────────────────────────┬────────────────────────────────────────┘
                                  │
          Then paste ONE of the phase sections below
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      │                           │                           │
      ▼                           ▼                           ▼
 ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
 │Phase B  │→│Phase C  │→│Phase D  │→│Phase E  │→│Phase F  │→ ...
 │Scaffold │ │KG Schema│ │Extract. │ │Embeddings│ │Labels  │
 └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
                               ↓                       ↓
                          ┌─────────┐ ┌─────────┐ ┌─────────┐
                      ... │Phase G  │→│Phase H  │→│Phase I  │→ Phase J
                          │Model    │ │Training │ │Eval/Inf │  Orchestr.
                          └─────────┘ └─────────┘ └─────────┘
```

### Phase-by-Phase Breakdown

| Master Prompt Section | Phase Name | What It Generates | Key Files Produced |
|-----------------------|-----------|-------------------|-------------------|
| **Section A** | System Context | *(No code — architectural context for Claude)* | — |
| **Section B** | Project Scaffolding & Configuration | Directory structure, YAML config, Pydantic loader, `requirements.txt`, Makefile, Dockerfile | `config/default.yaml`, `config/config.py`, `Makefile`, `Dockerfile`, `setup.py`, all `__init__.py` files |
| **Section C** | KG Schema & Neo4j Client | Pydantic data models, Neo4j CRUD, temporal resolution, Cypher queries | `src/kg/schema.py`, `src/kg/neo4j_client.py`, `src/kg/temporal.py`, `tests/test_schema.py`, `tests/test_kg.py` |
| **Section D** | Entity & Relation Extraction Pipeline | PMC corpus loader, SciSpacy NER, Claude-powered relation extraction, KG populator | `src/extraction/corpus_loader.py`, `src/extraction/entity_extractor.py`, `src/extraction/relation_extractor.py`, `src/kg/populator.py`, `scripts/01–04_*.py` |
| **Section E** | Embedding Pipeline & FAISS Index | SapBERT encoder wrapper, FAISS index (build/search/hard negatives), 4-level hierarchical indexing | `src/embeddings/sapbert.py`, `src/embeddings/faiss_index.py`, `src/embeddings/hierarchical.py`, `scripts/05_build_faiss_index.py` |
| **Section F** | Router Label Generation | Claude-powered token-level labeling (`factual`/`linguistic`), label validation & statistics | `src/labeling/llm_labeler.py`, `src/labeling/label_validator.py`, `scripts/06_generate_router_labels.py` |
| **Section G** | Model Architecture | BioMedLM backbone, router/retrieval/generation heads, InfoNCE loss, combined loss, full FRLM model | `src/model/backbone.py`, `src/model/router_head.py`, `src/model/retrieval_head.py`, `src/model/generation_head.py`, `src/model/losses.py`, `src/model/frlm.py`, `tests/test_model.py` |
| **Section H** | Training Pipeline | Three dataset classes, three phase-specific trainers, checkpointing/logging utilities, DeepSpeed config | `src/training/dataset.py`, `src/training/router_trainer.py`, `src/training/retrieval_trainer.py`, `src/training/joint_trainer.py`, `src/training/utils.py`, `scripts/07–09_*.py`, `config/deepspeed_config.json` |
| **Section I** | Evaluation & Inference | Retrieval/generation/router/E2E metrics, inference pipeline, FastAPI server | `src/evaluation/*.py`, `src/inference/pipeline.py`, `src/inference/server.py`, `scripts/10_evaluate.py`, `scripts/11_run_inference.py` |
| **Section J** | Automation & Orchestration | Master orchestrator, cost estimator, Makefile updates, `--start-from` / `--dry-run` / `--status` support | `scripts/run_full_pipeline.py`, `scripts/estimate_costs.py`, Makefile (final version) |

### How to Reproduce or Modify

1. **Regenerate a single module** — paste Section A + the relevant Phase section
   into a new Claude conversation. Claude will produce all files for that phase.
2. **Fix a bug** — paste Section A + the broken file + the error message.
   Claude will fix it with full architectural context.
3. **Add a feature** — paste Section A + describe the feature. Reference the
   relevant Phase section if the feature belongs to an existing module.
4. **Estimated effort** — 9–12 Claude conversations, ~15–30K output tokens each,
   ~200K total output tokens.

### Validation Checklist (Section K)

After all phases are generated, the master prompt includes a
validation checklist with 10 questions:

- Can `make setup` install all dependencies?
- Can `make full-pipeline` run end-to-end with only a config file and API keys?
- Are all hyperparameters in `default.yaml` with no hardcoded values?
- Is every file importable with no circular dependencies?
- Do all scripts use argparse and read from config?
- Is there error handling and logging in every file?
- Are all tensor shapes documented in comments?
- Can training resume from any checkpoint after interruption?
- Are API calls rate-limited and retried on failure?
- Is there a test for every critical function?

---

## Configuration

All hyperparameters live in **`config/default.yaml`** — the single source of truth.
No magic numbers in source code. The Pydantic config loader (`config/config.py`)
validates the YAML and loads sensitive values from **`config/secrets.properties`**:

| Property Key         | Config Field                  | Description |
|---------------------|-------------------------------|-------------|
| `neo4j.uri`          | `neo4j.uri`                   | Neo4j Bolt URI |
| `neo4j.username`     | `neo4j.username`              | Neo4j username |
| `neo4j.password`     | `neo4j.password`              | Neo4j database password |
| `anthropic.api_key`  | `extraction.relation.api_key` | Claude API key (for Steps 3 & 6) |
| `wandb.api_key`      | `wandb.api_key`               | Weights & Biases tracking |
| `ncbi.api_key`       | *(corpus loader)*             | NCBI E-utilities API key |

To get started, copy the example file and fill in your values:

```bash
cp config/secrets.properties.example config/secrets.properties
```

---

## Requirements

| Requirement | Minimum |
|-------------|---------|
| Python      | 3.10+   |
| CUDA        | 11.8+   |
| Neo4j       | 5.0+    |
| GPU         | 1× A100 80 GB (recommended) or 2× A6000 48 GB |

Key libraries: PyTorch 2.1+, HuggingFace Transformers ≥ 4.36, faiss-gpu ≥ 1.7.4,
SciSpacy ≥ 0.5.3, DeepSpeed ≥ 0.12, pytorch-metric-learning ≥ 2.3, Anthropic SDK,
W&B. Full list in `requirements.txt`.

---

## Docker

```bash
make docker-build       # Build image (CUDA base, all deps)
make docker-run         # Run training in container (mounts data/, checkpoints/, logs/)
make docker-serve       # Run FastAPI inference server on port 8000
```

Requires `--gpus all` and an `.env` file with your API keys.

---

## Makefile Reference

```bash
make help               # Show all available targets
make setup              # Install dependencies (CPU)
make setup-gpu          # Install dependencies (GPU/CUDA 11.8)
make setup-neo4j        # Setup Neo4j via Docker (recommended)
make setup-neo4j-native # Setup Neo4j via native apt
make neo4j-status       # Check Neo4j status
make neo4j-stop         # Stop Neo4j
make data-pipeline      # Run Steps 1–6 (data pipeline)
make train-all          # Run Steps 7–9 (all training phases)
make full-pipeline      # Run all 11 steps end-to-end
make resume STEP=N      # Resume pipeline from step N
make dry-run            # Print pipeline plan without executing
make status             # Show which steps have completed
make estimate-costs     # Estimate Claude API + GPU + storage costs
make evaluate           # Run evaluation
make serve              # Start FastAPI server
make test               # Run unit tests with coverage
make lint               # Run ruff + mypy
make format             # Auto-format with black + isort
make clean              # Remove build artifacts
make clean-data         # Remove processed data (keeps raw corpus)
make clean-all          # Remove everything
```

---

## License

© 2026 Sumit Srivastava. All rights reserved.

This project is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))**, with the additional terms below:

### You MAY:

- ✅ Use this software for **personal, academic, and non-commercial research** purposes.
- ✅ Modify, adapt, or build upon this software.
- ✅ Share your modifications — **only under this same license (CC BY-NC-SA 4.0)**.

### You MAY NOT:

- 🚫 Use this software, or any derivative of it, for **any commercial purpose** — including but not limited to: selling, licensing, offering as a service (SaaS), incorporating into commercial products, or using it within a for-profit organization's operations — **without explicit written permission from the original author**.
- 🚫 Change the license of your derivative work. All modifications and derivative works **must be distributed under this exact same license**.
- 🚫 Apply additional legal or technological restrictions that prevent others from exercising the rights granted by this license.

### Commercial Licensing

If you wish to use FRLM or any derivative for commercial purposes, you **must** obtain a separate commercial license from the original author.

**Contact:** Sumit Srivastava — [sumit-srivastava on GitHub](https://github.com/sumitsrv)

### Attribution

You must give appropriate credit, provide a link to this repository and to the
license, and indicate if changes were made.

