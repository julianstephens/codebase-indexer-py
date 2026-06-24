# CLI Reference

## indexer index

Index a repository and write the knowledge graph to disk.

```
indexer index [REPO_PATH] [OPTIONS]
```

| Argument / Option | Default | Description |
|---|---|---|
| `REPO_PATH` | `.` | Path to the repository root |
| `--project`, `-p` | dir name | Project name (used for cache file naming) |
| `--cache-dir` | `~/.cache/codebase-indexer` | Directory for the working `.db` file |
| `--workers`, `-w` | `0` (auto) | Number of parallel workers |
| `--incremental` / `--no-incremental` | `--incremental` | Skip files unchanged since last index |
| `--export` / `--no-export` | `--export` | Write compressed `.zst` artifact on completion |
| `--verbose`, `-v` | off | Enable debug logging |

**Example:**

```bash
indexer index /path/to/my-repo --project my-app --workers 8
```

The command summary includes relationship diagnostics:

- discovered/resolved/unresolved/unsupported call counts
- malformed extraction payload count (if nonzero)
- languages where relationship extraction is unavailable

---

## indexer skeleton

Print a skeleton of the indexed codebase (file headers, imports, and signatures).

```
indexer skeleton PROJECT [OPTIONS]
```

| Argument / Option | Default | Description |
|---|---|---|
| `PROJECT` | required | Project name |
| `--mode`, `-m` | auto | Rendering mode: `skeleton`, `compact`, `summary`, `deps` |
| `--db` | derived | Path to the `.db` file (overrides cache lookup) |
| `--cache-dir` | `~/.cache/codebase-indexer` | Directory for `.db` files |

**Example:**

```bash
indexer skeleton my-app --mode compact
```

---

## indexer get-source

Fetch the full source of a symbol by its fully-qualified name.

```
indexer get-source QUALIFIED_NAME [OPTIONS]
```

| Argument / Option | Default | Description |
|---|---|---|
| `QUALIFIED_NAME` | required | FQN of the symbol, e.g. `my_app.src.service.charge` |
| `--project`, `-p` | none | Project name (required when `--db` is not given) |
| `--db` | derived | Path to the `.db` file |
| `--cache-dir` | `~/.cache/codebase-indexer` | Directory for `.db` files |

**Example:**

```bash
indexer get-source "src.payments.service.charge" --project my-app
```

---

## indexer search

Full-text search across node names, signatures, and source.

```
indexer search QUERY [OPTIONS]
```

| Argument / Option | Default | Description |
|---|---|---|
| `QUERY` | required | Full-text search query |
| `--project`, `-p` | none | Filter results to a specific project |
| `--label`, `-l` | none | Filter by node type: `Function`, `Class`, `Method`, `Interface`, `Type` |
| `--file`, `-f` | none | SQL `LIKE` pattern for file path, e.g. `src/payments/%` |
| `--limit`, `-n` | `20` | Maximum number of results to return |
| `--db` | derived | Path to the `.db` file |
| `--cache-dir` | `~/.cache/codebase-indexer` | Directory for `.db` files |

**Example:**

```bash
indexer search "authentication" --project my-app --label Function --limit 10
```
