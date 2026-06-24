# codebase-indexer-py

This project is a reverse-engineered, simplified reimplementation of [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) by DeusData. It drops the MCP protocol and reduces the implementation to a standalone Python library and CLI, retaining the core ideas around indexing approach, graph schema, and agent tooling design.

It walks a source repository, extracts every function, class, method, interface, and type definition using tree-sitter, stores them in a compressed SQLite knowledge graph, and exposes the result to an AI agent as a lightweight skeleton and three on-demand retrieval tools.

## The problem it solves

AI agents that reason about large codebases need to read real code without loading an entire repository into context on every session. The agent receives a ~2-4k token skeleton at session start, then fetches full source on demand as it works.

## How it works

```txt
repo on disk
     |
     v
walker.py          discovers files, applies .gitignore + .cbmignore rules
     |
     v
extractor.py       routes each file to tree-sitter or fallback extractor
     |
     v
treesitter.py      parses AST -> NodeRecord (name, signature, source,
fallback.py        start_line, end_line, label, parent, properties)
     |
     v
fqn.py             assigns qualified names: src.payments.service.charge
     |
     v
registry.py        builds symbol index, resolves call sites -> edges
     |             (same_module -> import_map -> fuzzy -> unresolved)
     v
store.py           bulk-inserts nodes + edges into SQLite (WAL, FTS5)
     |
     v
artifact.py        VACUUM INTO + zstd -> .codebase-index/graph.db.zst
     |
     v
context.py         renders skeleton string for the agent (4 modes)
tools.py           get_source() / search() / trace_callers()
```

## Stack

| Layer | Library |
|---|---|
| AST extraction | `tree-sitter` |
| Database | `sqlite3` stdlib (WAL + FTS5) |
| Compression | `zstandard` |
| File discovery | `pathspec` (gitignore syntax) |
| Python | 3.13+ |
