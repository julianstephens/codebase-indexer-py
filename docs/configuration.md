# Configuration

## PipelineConfig

All indexing behaviour is controlled through `PipelineConfig`:

```python
from indexer.pipeline import PipelineConfig
from indexer.walker import WalkConfig

config = PipelineConfig(
    project="my-app",               # default: derived from repo dir name
    cache_dir="~/.cache/codebase-indexer",
    artifact_dir="/repo/.codebase-index",
    max_workers=8,                  # parallel read/extract/resolve threads
    walk_config=WalkConfig(
        max_file_bytes=2 * 1024 * 1024,
        include_unknown_extensions=True,
        extra_ignore_patterns=["*.generated.ts", "migrations/"],
        follow_gitignore=True,
    ),
    min_confidence=0.0,             # minimum edge confidence to store
    incremental=True,               # skip unchanged files
    export_artifact=True,
    artifact_compression_level=9,   # zstd level 1-22
    verbose=False,
)
```

### PipelineConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `project` | `str` | dir name | Project name used for cache file naming |
| `cache_dir` | `str` | `~/.cache/codebase-indexer` | Directory for the working `.db` file |
| `artifact_dir` | `str` | `<repo>/.codebase-index` | Directory for the exported artifact |
| `max_workers` | `int` | `4` | Number of parallel worker threads |
| `min_confidence` | `float` | `0.0` | Minimum call-edge confidence score |
| `incremental` | `bool` | `False` | Skip files unchanged since last index |
| `export_artifact` | `bool` | `True` | Write compressed `.db.zst` on completion |
| `artifact_compression_level` | `int` | `9` | zstd compression level (1 = fast, 22 = best) |
| `verbose` | `bool` | `False` | Log per-file progress |
| `walk_config` | `WalkConfig` | see below | File discovery settings |

### WalkConfig fields

| Field | Type | Default | Description |
|---|---|---|---|
| `max_file_bytes` | `int` | `1_048_576` | Skip files larger than this |
| `include_unknown_extensions` | `bool` | `False` | Index files with unrecognised extensions as `File` nodes |
| `extra_ignore_patterns` | `list[str]` | `[]` | Additional gitignore-style patterns |
| `follow_gitignore` | `bool` | `True` | Respect `.gitignore` at the repo root |

## .cbmignore

Place a `.cbmignore` file at the repo root to add project-specific ignore rules on top of `.gitignore`. It uses the same gitignore syntax:

```gitignore
# .cbmignore
migrations/
*.pb.go
*_generated.py
fixtures/
```

## Token budget and rendering modes

`build_context()` automatically selects a rendering mode based on the estimated token count of the full skeleton versus your budget:

| Mode | When used | Approximate token cost |
|---|---|---|
| `skeleton` | Full skeleton fits in budget | 100% |
| `compact` | Up to 2x over budget | ~60% of skeleton |
| `summary` | Up to 10x over budget | ~10% of skeleton |
| `deps` | Over 10x budget | Minimal |

Override the mode explicitly if needed:

```python
from indexer.context import build_context

ctx = build_context(db_path, "my-app", mode="compact")
```

## Artifact compression

Typical compression ratios for real repositories:

| Repo type | Uncompressed | Compressed | Ratio |
|---|---|---|---|
| Small Python app (50 files) | 2 MB | 200 KB | 10x |
| Medium TypeScript monorepo (400 files) | 18 MB | 1.4 MB | 13x |
| Large Go service (1200 files) | 65 MB | 5 MB | 13x |

The artifact is safe to commit alongside the repository. The `.gitattributes` file written by `export()` marks it `binary merge=ours` so git never produces merge conflicts on it.
