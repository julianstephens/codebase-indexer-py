# codebase-indexer-py

**A simplified, non-mcp version of [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp)**

A Python tool that walks a source repository, extracts every function,
class, method, interface, and type definition using tree-sitter, stores
them in a compressed SQLite knowledge graph, and exposes the result to
an AI agent as a lightweight skeleton + three on-demand retrieval tools.

The core problem it solves: AI agents that reason about large codebases
need to read real code without loading an entire repository into context
on every session. The agent receives a ~2–4k token skeleton at session
start, then fetches full source on demand as it works.

---

## How it works

```txt
repo on disk
     │
     ▼
walker.py          discovers files, applies .gitignore + .cbmignore rules
     │
     ▼
extractor.py       routes each file to tree-sitter or fallback extractor
     │
     ▼
treesitter.py      parses AST → NodeRecord (name, signature, source,
fallback.py        start_line, end_line, label, parent, properties)
     │
     ▼
fqn.py             assigns qualified names: src.payments.service.charge
     │
     ▼
registry.py        builds symbol index, resolves call sites → edges
     │             (same_module → import_map → fuzzy → unresolved)
     ▼
store.py           bulk-inserts nodes + edges into SQLite (WAL, FTS5)
     │
     ▼
artifact.py        VACUUM INTO + zstd → .codebase-index/graph.db.zst
     │
     ▼
context.py         renders skeleton string for the agent (4 modes)
tools.py           get_source() · search() · trace_callers()
```

### What the agent sees

At session start the agent receives a skeleton — every file header,
import list, and signature in the repo, grouped by file:

```txt
# my-app — 42 files, 312 nodes  [skeleton]
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

The agent then calls three tools to read code on demand:

```python
get_source("src.payments.service.charge")   # full source + callers + callees
search("sql injection")                      # FTS5 across name, signature, source
trace_callers("src.payments.service.charge") # blast radius BFS up the call graph
```

---

## Stack

| Layer | Library |
|---|---|
| AST extraction | `tree-sitter` + language-specific parser packages |
| Database | `sqlite3` stdlib (WAL + FTS5) |
| Compression | `zstandard` |
| File discovery | `pathspec` (gitignore syntax) |
| Python ≥ | 3.13 |

---

## Installation

```bash
pip install codebase-indexer-py
```

Or install as an isolated CLI with pipx:

```bash
pipx install codebase-indexer-py --python python3.13
```

---

## Quickstart

### Index a repository

```bash
indexer index /path/to/my-repo
```

Produces:

```
/path/to/my-repo/.codebase-index/
    graph.db.zst       compressed knowledge graph
    artifact.json      metadata (node counts, compression ratio, etc.)
    .gitattributes     marks graph.db.zst as binary + merge=ours
```

The working database is cached at:

```txt
~/.cache/codebase-indexer/<project>.db
```

### Print the skeleton

```bash
indexer skeleton my-app
```

### Query from the CLI

```bash
# Fetch a node's full source
indexer get-source "src.payments.service.charge" --project my-app

# Full-text search
indexer search "sql injection" --project my-app
```

### Re-index (incremental)

```bash
# Only changed files are re-extracted
indexer index /path/to/my-repo --incremental
```

### Use from Python

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

### Attach to an agent session

```python
from indexer.context import build_context
from indexer.tools import get_source, search, trace_callers

db_path = "~/.cache/codebase-indexer/my-app.db"

# Build the skeleton (auto-selects rendering mode by token budget)
skeleton = build_context(db_path, project="my-app", token_budget=8_000)

messages = [
    {
        "role": "system",
        "content": (
            "You are a software engineer working on this codebase.\n"
            "Use get_source(qn) to read function bodies, search(query) to find\n"
            "relevant code, and trace_callers(qn) to understand what depends on\n"
            "a given function before making changes.\n\n"
            f"Repository skeleton:\n\n{skeleton}"
        ),
    }
]

# Register the three tools with your agent framework and pass `messages`.
# Each tool call receives db_path automatically via a closure or partial.
```

---

## Supported languages

| Language | Extensions |
|---|---|
| Python | `.py` `.pyi` |
| TypeScript | `.ts` `.tsx` |
| JavaScript | `.js` `.jsx` `.mjs` `.cjs` |
| Go | `.go` |
| Rust | `.rs` |
| Java | `.java` |
| C | `.c` `.h` |
| C++ | `.cpp` `.cc` `.cxx` `.hpp` `.hxx` |
| C# | `.cs` |
| Ruby | `.rb` |
| PHP | `.php` |
| Kotlin | `.kt` `.kts` |
| Swift | `.swift` |
| Scala | `.scala` |
| Lua | `.lua` |
| Elixir | `.ex` `.exs` |
| Bash | `.sh` `.bash` |

Unrecognised file types (YAML, TOML, Dockerfile, SQL, Markdown, etc.)
are stored as single `File` nodes so `get_source()` still works on them.

---

## Configuration

### PipelineConfig

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
    artifact_compression_level=9,   # zstd level 1–22
    verbose=False,
)
```

### .cbmignore

Place a `.cbmignore` file at the repo root to add project-specific
ignore rules on top of `.gitignore`. Uses the same gitignore syntax:

```gitignore
# .cbmignore
migrations/
*.pb.go
*_generated.py
fixtures/
```

---

## Token budget and rendering modes

`build_context()` automatically chooses a rendering mode based on the
estimated token count of the full skeleton versus your budget:

| Mode | When used | Token cost |
|---|---|---|
| `skeleton` | Full skeleton fits in budget | Full |
| `compact` | Up to 2× over budget | ~60% of skeleton |
| `summary` | Up to 10× over budget | ~10% of skeleton |
| `deps` | Over 10× budget | Minimal |

Override the mode explicitly if needed:

```python
from indexer.context import build_context

# Always use compact mode regardless of size
ctx = build_context(db_path, "my-app", mode="compact")
```

---

## Artifact compression

Typical compression ratios for real repositories:

| Repo type | Uncompressed | Compressed | Ratio |
|---|---|---|---|
| Small Python app (50 files) | 2 MB | 200 KB | 10× |
| Medium TypeScript monorepo (400 files) | 18 MB | 1.4 MB | 13× |
| Large Go service (1200 files) | 65 MB | 5 MB | 13× |

The artifact is safe to commit alongside the repository. The
`.gitattributes` file written by `export()` marks it `binary merge=ours`
so git never produces merge conflicts on it.

---

## Call resolution

The registry resolves raw call sites to qualified names using a
three-strategy chain:

Call and import inputs are read from `NodeRecord.properties` in pass 6
(`calls` / `call_sites` and `imports` payloads).

```
1. same_module   confidence=0.95
   callee matches a node in the caller's own module or parent package

2. import_map    confidence=0.85
   callee's root name is in the file's import list, and the resolved
   module+name exists in the registry

3. fuzzy         confidence=0.40
   bare callee name matches exactly one node across all modules
   (suppressed if > 5 nodes share the name)

4. unresolved    confidence=0.0
   no edge emitted; counted in PipelineResult.calls_unresolved
```

Set `min_confidence` in `PipelineConfig` to filter out low-confidence
edges if fuzzy matches produce too much noise for your use case.

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

Individual test files:

```bash
pytest tests/test_treesitter.py -v   # AST extraction for all 17 languages
pytest tests/test_registry.py   -v   # call resolution strategies
pytest tests/test_pipeline.py   -v   # pass-6 v2 call/import parsing and stats
pytest tests/test_store.py      -v   # SQLite graph store
pytest tests/test_tools.py      -v   # agent tool functions
```

Benchmark matrix (opt-in):

```bash
# runs language x repo-size benchmark scenarios
make benchmark

# equivalent direct command
pytest -n=0 --run-benchmarks -s tests/benchmarks/ -v
```

The benchmark suite reports, for each language/size combo:

- indexing runtime (seconds)
- average tokens per file (raw repository text)
- average tokens per file (agent context from `build_context`)
- token savings (absolute + percent)

By default benchmark tests are skipped in normal `pytest` runs.

| Language   | Size   | Avg Index (s) | Raw Tokens | Context Tokens | Saved Tokens | Saved % |
|------------|--------|---------------|------------|----------------|--------------|---------|
| go         | large  | 0.4270        | 77,487     | 2,903          | 74,584       | 96.25%  |
| go         | medium | 0.0874        | 20,402     | 5,801          | 14,601       | 71.57%  |
| go         | small  | 0.0270        | 3,734      | 1,799          | 1,935        | 51.82%  |
| python     | large  | 0.4583        | 71,759     | 2,904          | 68,855       | 95.95%  |
| python     | medium | 0.0981        | 18,969     | 6,400          | 12,569       | 66.26%  |
| python     | small  | 0.0316        | 3,494      | 2,023          | 1,471        | 42.10%  |
| typescript | large  | 0.4848        | 91,079     | 2,908          | 88,171       | 96.81%  |
| typescript | medium | 0.0973        | 24,049     | 8,914          | 15,135       | 62.93%  |
| typescript | small  | 0.0266        | 4,424      | 2,491          | 1,933        | 43.69%  |

---

## Incremental indexing

On re-index, the pipeline compares each file's `sha256` and `mtime_ns`
against the stored values. Only changed files are re-extracted. Stale
nodes and edges for changed files are deleted before new ones are
inserted, so the graph is always consistent.

```
First run:   247 nodes, 12.3s
Second run:  3 files changed → 6 nodes updated, 0.8s
```

---

## Limitations

- **Call resolution is best-effort.** Dynamic dispatch, higher-order
  functions, and metaprogramming produce unresolved or low-confidence
  edges. The graph is a static approximation.

- **No cross-file type inference.** The registry resolves names by
  structural matching, not type flow. A call `user.charge()` where
  `user` could be one of several classes will not be resolved unless
  the name is unambiguous.

- **Monorepos with many small packages** may produce noisy fuzzy edges.
  Set `min_confidence=0.85` to restrict edges to same-module and
  import-map resolutions only.

- **Generated files** (protobuf, GraphQL codegen, migration files) are
  excluded by default via `_ALWAYS_SKIP_FILENAMES` and
  `_ALWAYS_SKIP_EXTENSIONS`. Add project-specific patterns to
  `.cbmignore`.
