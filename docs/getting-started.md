# Getting Started

## Installation

```bash
pip install codebase-indexer-py
```

Or install from source:

```bash
git clone https://github.com/julianstephens/codebase-indexer-py.git
cd codebase-indexer-py
pip install .
```

## Index a repository

```bash
indexer index /path/to/my-repo
```

This produces:

```
/path/to/my-repo/.codebase-index/
    graph.db.zst       compressed knowledge graph
    artifact.json      metadata (node counts, compression ratio, etc.)
    .gitattributes     marks graph.db.zst as binary + merge=ours
```

The working database is cached at:

```
~/.cache/codebase-indexer/<project>.db
```

## Print the skeleton

```bash
indexer skeleton my-app
```

Example output:

```txt
# my-app -- 42 files, 312 nodes  [skeleton]
# schema: Class=18 File=12 Function=156 Method=98 Interface=8 Type=20

### src/payments/service.py
# imports: stripe, src.payments.models, src.auth.models
def charge(user: User, amount_cents: int, currency: str) -> Payment:  # src.payments.service.charge
def refund(payment: Payment) -> bool:  # src.payments.service.refund

### src/payments/models.py
class Payment(BaseModel):  # src.payments.models.Payment
    def save(self) -> None:  # src.payments.models.Payment.save
    def to_dict(self) -> dict:  # src.payments.models.Payment.to_dict
```

## Query from the CLI

```bash
# Fetch a node's full source
indexer get-source "src.payments.service.charge" --project my-app

# Full-text search
indexer search "sql injection" --project my-app
```

## Incremental re-index

Only changed files are re-extracted:

```bash
indexer index /path/to/my-repo --incremental
```

## Index from Python

```python
from indexer.pipeline import run, PipelineConfig

result = run("/path/to/my-repo", PipelineConfig(
    project="my-app",
    incremental=True,
    export_artifact=True,
    artifact_compression_level=9,
))

print(f"Indexed {result.nodes_total} nodes in {result.elapsed_seconds:.1f}s")
print(f"Artifact: {result.artifact_path}")
```
