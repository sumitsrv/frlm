# FRLM - Factual Retrieval Language Model

Separates factual knowledge from linguistic competence in language models. Instead of encoding facts in model parameters, facts are stored in an external temporal knowledge graph and retrieved via learned dense embeddings.

## Architecture

- **Backbone**: BioMedLM 2.7B (decoder-only transformer)
- **Router Head**: Classifies each generation step as retrieval or generation
- **Retrieval Head**: Projects into SapBERT embedding space with granularity and temporal sub-heads
- **Generation Head**: Standard next-token prediction
- **Knowledge Graph**: Neo4j with temporal fact schema (append-only, version chains)
- **Vector Index**: FAISS IVF-PQ on GPU over SapBERT fact embeddings

## Domain

Biomedical oncology pharmacology. Corpus: PubMed Central Open Access subset.

## Quick Start

```bash
# Setup
make setup

# Full data pipeline
make data-pipeline

# Three-phase training
make train-all

# Evaluate
make evaluate

# Serve
make serve
```

## Project Structure

```
frlm/
├── config/           # YAML configs and Pydantic loader
├── data/             # Corpus, processed data, KG exports, labels
├── src/
│   ├── kg/           # Neo4j client, schema, temporal logic
│   ├── embeddings/   # SapBERT encoder, FAISS index, hierarchical indexing
│   ├── extraction/   # SciSpacy NER, Claude relation extraction, corpus loading
│   ├── model/        # BioMedLM backbone, router/retrieval/generation heads
│   ├── training/     # Three-phase training pipeline
│   ├── labeling/     # Claude API span labeling
│   ├── evaluation/   # Retrieval, generation, router, and E2E metrics
│   └── inference/    # Pipeline and FastAPI server
├── scripts/          # Numbered pipeline scripts (01-11)
├── tests/            # Unit tests
└── notebooks/        # Exploration and analysis notebooks
```

## Training Phases

1. **Router Pre-training**: Binary classification of retrieval vs. generation tokens
2. **Retrieval Head Training**: InfoNCE contrastive loss against FAISS index
3. **Joint Fine-tuning**: Combined loss (L_total = 1.0*L_router + 2.0*L_retrieval + 1.0*L_generation)

## Configuration

All hyperparameters live in `config/default.yaml`. Sensitive values (API keys, passwords) are overridden via environment variables:

| Environment Variable    | Config Field                  |
|------------------------|-------------------------------|
| `FRLM_NEO4J_PASSWORD`  | `neo4j.password`              |
| `ANTHROPIC_API_KEY`     | `extraction.relation.api_key` |
| `WANDB_API_KEY`         | `wandb.api_key`               |

## Requirements

- Python 3.10+
- CUDA 11.8+
- Neo4j 5.0+
- 1x A100 80GB (recommended) or 2x A6000 48GB

## License

Proprietary. All rights reserved.