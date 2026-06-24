"""
pipeline.py — Orchestrates the full indexing pipeline.

Wires together walker, extractor, registry, store, and artifact into a
single end-to-end run that takes a repo path and produces a compressed
SQLite artifact.

Pipeline passes (in order):
    Pass 1 — Discover        walker.walk() → list[FileInfo]
    Pass 2 — Read            read file content for each FileInfo
    Pass 3 — Extract         extractor.extract_files() → ExtractionResult per file
    Pass 4 — Assign QNs      fqn.compute() / fqn.module() on every NodeRecord
    Pass 5 — Build registry  registry.build() → Registry
    Pass 6 — Resolve calls   registry.resolve_all() → edges per file (parallel)
    Pass 7 — Store nodes     store.insert_nodes() for all records
    Pass 8 — Store edges     store.insert_edges() for all resolved calls
    Pass 9 — Store files     store.insert_files() + insert_file_hashes()
    Pass 10 — Dump           store.dump_to_file() → .db on disk
    Pass 11 — Export         artifact.export() → graph.db.zst

Incremental mode:
    When a previous artifact exists for the project, the pipeline loads
    the stored file hashes (sha256 + mtime_ns) and skips files whose
    content has not changed. Changed files are re-extracted; their old
    nodes and edges are deleted before new ones are inserted.

Parallelism:
    Pass 2 (read) and Pass 3 (extract) run in a ThreadPoolExecutor.
    Pass 6 (resolve) also runs in a ThreadPoolExecutor.
    Passes that write to SQLite are serial — SQLite's WAL mode supports
    concurrent readers but only one writer at a time.

Public API:
    run(repo_path, config)          — full pipeline, returns PipelineResult
    run_incremental(repo_path, config) — incremental re-index
    PipelineConfig                  — all tunable parameters
    PipelineResult                  — statistics and output paths
"""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .artifact import export
from .errors import FileNotFoundError as StoreFileNotFoundError
from .extractor import (
    ExtractionResult,
    FileInfo,
    extract_file_detailed,
)
from .fqn import compute, from_path, module
from .registry import (
    CallSite,
    Import,
    Registry,
    ResolutionContext,
    build,
)
from .relations import REL_STATE_DEFS_AND_RELS, get_relationship_capability
from .store import (
    DEFAULT_CACHE_DIR,
    Store,
    default_db_path,
    open_memory,
    open_path,
    open_path_readonly,
)
from .treesitter import NodeRecord
from .walker import WalkConfig, walk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    All tunable parameters for a pipeline run.

    Attributes:
        project:
            Project name stored in the database and used to namespace
            all nodes, edges, and files. If empty, derived from the
            repo directory name via fqn.from_path().

        cache_dir:
            Directory for the working .db file.
            Defaults to store.DEFAULT_CACHE_DIR.

        artifact_dir:
            Directory where the compressed .zst artifact is written.
            Defaults to "<repo_path>/.codebase-index/".

        max_workers:
            Number of threads for parallel read/extract/resolve passes.
            Defaults to min(8, os.cpu_count() or 4).

        walk_config:
            WalkConfig passed to walker.walk(). Controls ignore patterns,
            max file size, and extension filtering.

        min_confidence:
            Minimum confidence threshold for emitting a CALLS edge.
            Resolutions below this threshold are counted but no edge is
            stored. Defaults to 0.0 (emit all resolved calls).

        incremental:
            If True, load previous file hashes and skip unchanged files.
            Defaults to True.

        export_artifact:
            If True, compress the database to a .zst artifact after
            indexing. Defaults to True.

        artifact_compression_level:
            zstd compression level (1-22). Higher = smaller file,
            slower compression. Defaults to 9.

        on_progress:
            Optional callback invoked after each pass completes.
            Signature: (pass_name: str, current: int, total: int) -> None.
            Useful for CLI progress bars.

        verbose:
            If True, emit DEBUG-level log messages including per-file
            extraction results. Defaults to False.
    """

    project: str = ""
    cache_dir: str = DEFAULT_CACHE_DIR
    artifact_dir: str = ""
    max_workers: int = 0  # 0 = auto
    walk_config: WalkConfig = field(default_factory=WalkConfig)
    min_confidence: float = 0.0
    incremental: bool = True
    export_artifact: bool = True
    artifact_compression_level: int = 9
    on_progress: Callable | None = None
    verbose: bool = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """
    Statistics and output paths from a completed pipeline run.

    Attributes:
        project:            project name used for this run
        db_path:            absolute path to the working .db file
        artifact_path:      absolute path to the .zst artifact, or ""
                            if export_artifact=False
        files_discovered:   total files found by walker
        files_skipped:      files excluded (lock files, binaries, etc.)
        files_unchanged:    files skipped in incremental mode
        files_added:        newly discovered files in incremental mode
        files_changed:      modified files in incremental mode
        files_removed:      files removed since the previous index
        files_extracted:    files that were actually parsed
        nodes_total:        total nodes inserted
        nodes_by_label:     dict of label → count
        edges_total:        total edges inserted
        edges_by_type:      dict of type → count
        calls_discovered:   call sites discovered from extraction payloads
        calls_resolved:     call sites resolved to a known node
        calls_unresolved:   call sites that could not be resolved
        calls_unsupported:  call expressions omitted as unsupported
        malformed_payloads: malformed call/import payload items ignored
        relationship_unavailable_languages:
                    languages where relationship extraction is
                    unavailable for this run
        elapsed_seconds:    wall-clock time for the full run
        errors:             list of (file_path, error_message) for any
                            file that failed to read or extract
    """

    project: str
    db_path: str = ""
    artifact_path: str = ""
    files_discovered: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    files_added: int = 0
    files_changed: int = 0
    files_removed: int = 0
    files_extracted: int = 0
    nodes_total: int = 0
    nodes_by_label: dict[str, int] = field(default_factory=dict)
    edges_total: int = 0
    edges_by_type: dict[str, int] = field(default_factory=dict)
    calls_discovered: int = 0
    calls_resolved: int = 0
    calls_unresolved: int = 0
    calls_unsupported: int = 0
    malformed_payloads: int = 0
    relationship_unavailable_languages: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    repo_path: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Run the full indexing pipeline on a repository.

    Executes all 11 passes in order. If config.incremental is True and
    a previous database exists for this project, unchanged files are
    skipped automatically.

    The working database is written to:
        <config.cache_dir>/<project>.db

    The compressed artifact (if config.export_artifact is True) is
    written to:
        <config.artifact_dir>/graph.db.zst
    defaulting to <repo_path>/.codebase-index/graph.db.zst.

    Args:
        repo_path: absolute or relative path to the repository root.
                   Must be an existing directory.
        config:    PipelineConfig controlling all tunable parameters.
                   If None, uses PipelineConfig() defaults.

    Returns:
        PipelineResult with statistics about the completed run.

    Raises:
        NotADirectoryError: if repo_path does not exist or is not a dir.

    Examples:
        >>> result = run("/path/to/my-repo")
        >>> result.nodes_total
        247
        >>> result.artifact_path
        '/path/to/my-repo/.codebase-index/graph.db.zst'
    """
    repo_path = str(Path(repo_path).resolve())
    if not Path(repo_path).is_dir():
        raise NotADirectoryError

    if config is None:
        config = PipelineConfig()
    config = _resolve_config(repo_path, config)

    t_start = time.monotonic()
    result = PipelineResult(project=config.project)

    db_path = default_db_path(config.project, config.cache_dir)
    use_existing_db = config.incremental and Path(db_path).exists()
    db = open_path(db_path) if use_existing_db else open_memory()
    try:
        # Pass 1: Discover
        all_file_infos = _pass_discover(repo_path, config)
        result.files_discovered = len(all_file_infos)

        # Incremental: classify file states
        if config.incremental:
            stored_hashes = (
                db.get_file_hashes(config.project) if use_existing_db else {}
            )
            file_infos, unchanged, added, removed = _pass_filter_unchanged(
                all_file_infos,
                stored_hashes,
            )
            result.files_unchanged = len(unchanged)
            result.files_changed = len(file_infos) - len(added)
            result.files_added = len(added)
            result.files_removed = len(removed)
        else:
            file_infos = all_file_infos
            removed = []

        # Pass 3: Extract
        extraction_results = _pass_extract(file_infos, config)
        result.files_skipped = sum(
            1 for r in extraction_results.values() if r.extractor == "skip"
        )
        result.files_extracted = len(file_infos) - result.files_skipped
        result.errors = [
            (r.path, r.error) for r in extraction_results.values() if r.error
        ]

        # Pass 4: Assign QNs
        records = _pass_assign_qns(extraction_results, config.project)
        result.relationship_unavailable_languages = sorted(
            {
                r.language
                for r in extraction_results.values()
                if r.language
                and get_relationship_capability(r.language).state
                != REL_STATE_DEFS_AND_RELS
            }
        )

        if use_existing_db:
            (
                all_records,
                edges,
                edges_inserted,
                calls_discovered,
                calls_resolved,
                calls_unresolved,
                calls_unsupported,
                malformed_payloads,
            ) = _pass_store_incremental(
                db=db,
                project=config.project,
                repo_path=repo_path,
                changed_records=records,
                extraction_results=extraction_results,
                changed_file_infos=file_infos,
                removed_paths=[fi.path for fi in removed],
                config=config,
            )

            result.calls_discovered = calls_discovered
            result.calls_resolved = calls_resolved
            result.calls_unresolved = calls_unresolved
            result.calls_unsupported = calls_unsupported
            result.malformed_payloads = malformed_payloads
            result.nodes_total = len(all_records)
            result.nodes_by_label = _count_by_label(all_records)
            result.edges_total = edges_inserted
            result.edges_by_type = _count_by_edge_type(edges)
        else:
            # Pass 5: Build registry
            reg = _pass_build_registry(records)

            # Pass 6: Resolve calls
            (
                edges,
                calls_discovered,
                calls_resolved,
                calls_unresolved,
                calls_unsupported,
                malformed_payloads,
            ) = _pass_resolve_calls(
                extraction_results,
                reg,
                config.project,
                config,
            )
            result.calls_discovered = calls_discovered
            result.calls_resolved = calls_resolved
            result.calls_unresolved = calls_unresolved
            result.calls_unsupported = calls_unsupported
            result.malformed_payloads = malformed_payloads

            # Collect file contents + hashes
            file_contents = _collect_file_contents(file_infos, extraction_results)
            file_hashes = _collect_file_hashes(file_infos, file_contents)

            # Passes 7-9: Store
            file_languages = {fi.path: fi.language for fi in file_infos}
            _, edges_inserted = _pass_store(
                db,
                config.project,
                repo_path,
                records,
                edges,
                file_contents,
                file_hashes,
                file_languages,
            )

            result.nodes_total = len(records)
            result.nodes_by_label = _count_by_label(records)
            result.edges_total = edges_inserted
            result.edges_by_type = _count_by_edge_type(edges)

        # Pass 10: Dump
        _pass_dump(db, db_path)
        result.db_path = db_path
    finally:
        db.close()

    # Pass 11: Export artifact
    if config.export_artifact:
        try:
            result.artifact_path = _pass_export(
                result.db_path, config.artifact_dir, config.artifact_compression_level
            )
        except NotImplementedError:
            logger.debug("Artifact export skipped: module not yet available")

    result.elapsed_seconds = time.monotonic() - t_start
    _log_result(result)
    return result


def run_incremental(
    repo_path: str,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Convenience wrapper for run() with incremental=True forced.

    Identical to run() except config.incremental is always set to True
    regardless of what was passed. Useful for callers that want to be
    explicit about incremental mode.

    Args:
        repo_path: absolute or relative path to the repository root
        config:    PipelineConfig. incremental will be overridden to True.

    Returns:
        PipelineResult from the underlying run() call.
    """
    if config is None:
        config = PipelineConfig()
    config.incremental = True
    return run(repo_path, config)


# ---------------------------------------------------------------------------
# Internal passes
# ---------------------------------------------------------------------------


def _resolve_config(
    repo_path: str,
    config: PipelineConfig,
) -> PipelineConfig:
    """
    Fill in derived / default fields on the config before the run.

    Mutates and returns the same config object:
      - Sets config.project from fqn.from_path() if empty.
      - Sets config.artifact_dir to "<repo_path>/.codebase-index" if empty.
      - Sets config.max_workers to min(8, cpu_count) if 0.

    Args:
        repo_path: absolute repo root path (already resolved)
        config:    PipelineConfig, possibly with empty fields

    Returns:
        The same config object with all fields populated.
    """
    if not config.project:
        config.project = from_path(repo_path)
    if not config.artifact_dir:
        config.artifact_dir = f"{repo_path}/.codebase-index"
    if config.max_workers <= 0:
        import os

        config.max_workers = min(8, os.cpu_count() or 4)
    return config


def _pass_discover(
    repo_path: str,
    config: PipelineConfig,
) -> list[FileInfo]:
    """
    Pass 1: Discover all indexable files under repo_path.

    Delegates to walker.walk(). Emits a progress callback after
    completion if config.on_progress is set.

    Args:
        repo_path: absolute repo root path
        config:    resolved PipelineConfig

    Returns:
        Sorted list of FileInfo objects for all indexable files.
    """
    discovered_files = walk(repo_path, config.walk_config)
    if config.on_progress:
        _progress(config, "discover", len(discovered_files), len(discovered_files))
    return discovered_files


def _pass_filter_unchanged(
    file_infos: list[FileInfo],
    stored_hashes: dict[str, tuple[str, int, int]],
) -> tuple[list[FileInfo], list[FileInfo], list[FileInfo], list[FileInfo]]:
    """
    Split FileInfo list into (changed, unchanged, added, removed).

    A file is considered unchanged when:
      1. Its path is in stored_hashes, AND
      2. Its mtime_ns matches the stored mtime_ns, AND
      3. Its sha256 (computed here) matches the stored sha256.

    The mtime check is a cheap fast-path: if mtime matches, skip the
    sha256 computation entirely.

    For files where mtime differs but sha256 is the same (e.g. a file
    was touched without content change), the file is also considered
    unchanged.

    Args:
        file_infos:     all discovered FileInfo objects
        stored_hashes:  dict from store.get_file_hashes(), mapping
                        rel_path → (sha256_hex, mtime_ns, size_bytes)

    Returns:
        (changed, unchanged, added, removed) tuple of FileInfo lists.
        changed + unchanged == file_infos (every discovered file is in one list).
        added is a subset of changed.
        removed are files present in stored_hashes but missing from discovery.
    """
    changed = []
    unchanged = []
    added = []
    current_paths = {fi.path for fi in file_infos}
    removed = [
        FileInfo(path=path, abs_path="", language=None)
        for path in sorted(stored_hashes.keys() - current_paths)
    ]
    for fi in file_infos:
        stored = stored_hashes.get(fi.path)
        if stored is None:
            changed.append(fi)
            added.append(fi)
            continue
        stored_sha256, stored_mtime_ns, _ = stored
        if fi.mtime_ns != stored_mtime_ns:
            changed.append(fi)
            continue

        content, err = _read_file(fi.abs_path)
        if err:
            # If we can't read the file, treat it as changed so it will be
            # re-extracted (and fail gracefully in _pass_extract()).
            changed.append(fi)
            continue
        sha256 = _compute_sha256(content)
        if sha256 == stored_sha256:
            unchanged.append(fi)
        else:
            changed.append(fi)
    return changed, unchanged, added, removed


def _group_records_by_file(records: list[NodeRecord]) -> dict[str, list[NodeRecord]]:
    """Group NodeRecords by file_path."""
    grouped: dict[str, list[NodeRecord]] = {}
    for record in records:
        grouped.setdefault(record.file_path, []).append(record)
    return grouped


def _results_from_records(records: list[NodeRecord]) -> dict[str, ExtractionResult]:
    """Build ExtractionResult mapping from records already loaded from storage."""
    grouped = _group_records_by_file(records)
    return {
        path: ExtractionResult(
            path=path,
            records=file_records,
            language="unknown",
            extractor="stored",
        )
        for path, file_records in grouped.items()
    }


def _pass_extract(
    file_infos: list[FileInfo],
    config: PipelineConfig,
) -> dict[str, ExtractionResult]:
    """
    Pass 3: Extract NodeRecords from all changed files in parallel.

    Uses a ThreadPoolExecutor with config.max_workers threads. Reads
    each file via _read_file() and calls extractor.extract_file_detailed().

    Files that fail to read are recorded in the result with
    extractor="skip" and reason="read_error" — they do not raise.

    Args:
        file_infos: list of FileInfo objects for files to extract
        config:     resolved PipelineConfig

    Returns:
        Dict mapping repo-relative path → ExtractionResult.
        Every path in file_infos has an entry.
    """
    results: dict[str, ExtractionResult] = {}
    total = len(file_infos)

    def _extract_one(fi: FileInfo) -> tuple[str, ExtractionResult]:
        content, err = _read_file(fi.abs_path)
        if err:
            return fi.path, ExtractionResult(
                path=fi.path,
                records=[],
                language=fi.language or "unknown",
                extractor="skip",
                reason="read_error",
                error=err,
            )
        return fi.path, extract_file_detailed(fi.path, content)

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {pool.submit(_extract_one, fi): fi for fi in file_infos}
        for done, future in enumerate(as_completed(futures), 1):
            path, result = future.result()
            results[path] = result
            if config.verbose and result.extractor == "skip":
                logger.debug(
                    "skip %s reason=%s error=%s", path, result.reason, result.error
                )
            _progress(config, "extract", done, total)

    return results


def _pass_assign_qns(
    results: dict[str, ExtractionResult],
    project: str,  # noqa: ARG001
) -> list[NodeRecord]:
    """
    Pass 4: Set qualified_name on every NodeRecord.

    Iterates all ExtractionResults. For each NodeRecord with an empty
    qualified_name, calls fqn.compute() (for symbol nodes) or
    fqn.module() (for File nodes). Sets record.qualified_name in place.

    Records whose qualified_name is still empty after assignment are
    logged as warnings and excluded from the returned list.

    Args:
        results: dict of ExtractionResults from _pass_extract()
        project: project name (unused here but available for future
                 cross-project QN prefixing)

    Returns:
        Flat list of all NodeRecord objects with qualified_name set.
    """
    all_records: list[NodeRecord] = []
    for result in results.values():
        for record in result.records:
            if not record.qualified_name:
                try:
                    if record.label == "File":
                        record.qualified_name = module(record.file_path)
                    else:
                        record.qualified_name = compute(
                            record.file_path, record.name, record.parent or None
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not assign QN to %r in %s: %s",
                        record.name,
                        record.file_path,
                        exc,
                    )
            if record.qualified_name:
                all_records.append(record)
            else:
                logger.warning(
                    "Dropping record with empty QN: name=%r file=%s",
                    record.name,
                    record.file_path,
                )
    return all_records


def _pass_build_registry(
    records: list[NodeRecord],
) -> Registry:
    """
    Pass 5: Build the symbol registry from all extracted NodeRecords.

    Delegates to registry.build(). This is a serial pass — the registry
    must be fully built before call resolution begins.

    Args:
        records: flat list of all NodeRecords with QNs assigned

    Returns:
        Fully populated Registry.
    """
    return build(records)


def _pass_resolve_calls(
    results: dict[str, ExtractionResult],
    reg: Registry,
    project: str,
    config: PipelineConfig,
) -> tuple[list[tuple[str, str, str, dict]], int, int, int, int, int]:
    """
    Pass 6: Resolve call sites to edges in parallel.

     For each file in results:
        1. Builds a ResolutionContext with module_qn and imports parsed
            from NodeRecord.properties["imports"].
        2. Builds a list of CallSite objects from
            NodeRecord.properties["calls"].
      3. Calls reg.resolve_all(calls, ctx).
      4. Filters resolutions below config.min_confidence.
      5. Yields (source_qn, target_qn, "CALLS", properties) tuples.

    Runs in a ThreadPoolExecutor — resolution is CPU-bound (pure Python
    dict lookups) but benefits from parallelism on large repos due to
    per-file context setup overhead.

    Args:
        results: dict of ExtractionResults (includes parsed imports)
        reg:     fully built Registry from _pass_build_registry()
        project: project name
        config:  resolved PipelineConfig (for min_confidence, max_workers)

    Returns:
                (edges, calls_discovered, calls_resolved, calls_unresolved,
                 calls_unsupported, malformed_payloads) where:
          - edges is a list of (source_qn, target_qn, edge_type, properties)
            tuples ready for store.insert_edges().
                    - calls_discovered is the count of call sites discovered.
          - calls_resolved is the count of call sites with a resolved target.
          - calls_unresolved is the count of call sites that could not be
            resolved.
          - calls_unsupported is the count of unsupported call expressions.
          - malformed_payloads is the count of malformed payload items ignored.
    """
    all_edges: list[tuple[str, str, str, dict]] = []
    calls_discovered = 0
    calls_resolved = 0
    calls_unresolved = 0
    calls_unsupported = 0
    malformed_payloads = 0
    items = list(results.items())
    total = len(items)

    def _resolve_file(
        path: str, result: ExtractionResult
    ) -> tuple[list[tuple[str, str, str, dict]], int, int, int, int, int]:
        imports, import_malformed = _collect_imports_from_records(result.records)
        calls, call_malformed, unsupported_count = _collect_calls_from_records(
            result.records
        )
        ctx = ResolutionContext(
            file_path=path,
            module_qn=module(path),
            imports=imports,
            project=project,
        )
        edges: list[tuple[str, str, str, dict]] = []
        discovered_count = len(calls)
        resolved_count = 0
        unresolved_count = 0
        for res in reg.resolve_all(calls, ctx):
            if res.target_qn:
                resolved_count += 1
            else:
                unresolved_count += 1
            if not res.target_qn or res.confidence < config.min_confidence:
                continue
            edges.append(
                (
                    res.source_qn,
                    res.target_qn,
                    "CALLS",
                    {
                        "confidence": res.confidence,
                        "strategy": res.strategy,
                        "line": res.call_site.line,
                    },
                )
            )
        return (
            edges,
            discovered_count,
            resolved_count,
            unresolved_count,
            unsupported_count,
            import_malformed + call_malformed,
        )

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(_resolve_file, path, result): path for path, result in items
        }
        for done, future in enumerate(as_completed(futures), 1):
            (
                edges,
                discovered_count,
                resolved_count,
                unresolved_count,
                unsupported_count,
                malformed_count,
            ) = future.result()
            all_edges.extend(edges)
            calls_discovered += discovered_count
            calls_resolved += resolved_count
            calls_unresolved += unresolved_count
            calls_unsupported += unsupported_count
            malformed_payloads += malformed_count
            _progress(config, "resolve", done, total)

    return (
        all_edges,
        calls_discovered,
        calls_resolved,
        calls_unresolved,
        calls_unsupported,
        malformed_payloads,
    )


def _collect_imports_from_records(
    records: list[NodeRecord],
) -> tuple[list[Import], int]:
    """
    Collect Import objects from NodeRecord.properties["imports"].

    Accepts multiple v2-compatible shapes:
      - "imports": "src.payments.service"
      - "imports": ["src.payments.service", {...}]
      - "imports": {"module_path": "src.payments", "names": ["service"]}

    Invalid entries are ignored.
    """
    collected: list[Import] = []
    malformed = 0
    seen: set[tuple[str, tuple[str, ...], str, int, str]] = set()

    for record in records:
        raw_imports = record.properties.get("imports")
        if raw_imports is None:
            continue

        parsed_items, malformed_items = _parse_import_items(
            raw_imports,
            default_in_function=record.qualified_name,
        )
        malformed += malformed_items
        for imp in parsed_items:
            key = (
                imp.module_path,
                tuple(imp.names),
                imp.alias,
                imp.line,
                imp.in_function,
            )
            if key in seen:
                continue
            seen.add(key)
            collected.append(imp)

    return collected, malformed


def _parse_import_items(
    raw_imports: object,
    default_in_function: str,
) -> tuple[list[Import], int]:
    """Parse raw import payload(s) into Import objects."""
    if isinstance(raw_imports, (str, dict)):
        raw_items = [raw_imports]
    elif isinstance(raw_imports, list):
        raw_items = raw_imports
    else:
        return [], 1

    imports: list[Import] = []
    malformed = 0
    for item in raw_items:
        if isinstance(item, str):
            module_path = item.strip()
            if module_path:
                imports.append(Import(module_path=module_path, in_function=""))
            else:
                malformed += 1
            continue

        if not isinstance(item, dict):
            malformed += 1
            continue

        module_path_obj = (
            item.get("module_path")
            or item.get("module")
            or item.get("path")
            or item.get("import")
        )
        if not isinstance(module_path_obj, str):
            malformed += 1
            continue
        module_path = module_path_obj.strip()
        if not module_path:
            malformed += 1
            continue

        names_raw = item.get("names")
        names: list[str] = []
        if isinstance(names_raw, list):
            names = [name.strip() for name in names_raw if isinstance(name, str)]
        elif isinstance(names_raw, str):
            names = [name.strip() for name in names_raw.split(",") if name.strip()]

        alias_raw = item.get("alias") or item.get("as")
        alias = alias_raw.strip() if isinstance(alias_raw, str) else ""

        scope_raw = item.get("scope")
        scope = scope_raw.strip() if isinstance(scope_raw, str) else ""

        in_function_raw = item.get("in_function") or item.get("source_qn")
        if isinstance(in_function_raw, str) and in_function_raw.strip():
            in_function = in_function_raw.strip()
        elif scope == "local":
            in_function = default_in_function
        else:
            in_function = ""

        line_raw = item.get("line", 0)
        line = line_raw if isinstance(line_raw, int) else 0

        imports.append(
            Import(
                module_path=module_path,
                names=names,
                alias=alias,
                line=line,
                in_function=in_function,
            )
        )

    return imports, malformed


def _collect_calls_from_records(
    records: list[NodeRecord],
) -> tuple[list[CallSite], int, int]:
    """
    Collect CallSite objects from NodeRecord.properties call payloads.

    v2 payloads may use "calls" or "call_sites" and can be strings,
    dicts, or lists of either.
    """
    collected: list[CallSite] = []
    malformed = 0
    unsupported = 0
    seen: set[tuple[str, int, str, str]] = set()

    for record in records:
        unsupported_raw = record.properties.get("unsupported_calls", 0)
        if isinstance(unsupported_raw, int) and unsupported_raw > 0:
            unsupported += unsupported_raw

        raw_calls = record.properties.get("calls")
        if raw_calls is None:
            raw_calls = record.properties.get("call_sites")
        if raw_calls is None:
            continue

        default_in_function = record.qualified_name
        parsed_items, malformed_items = _parse_call_items(
            raw_calls, default_in_function
        )
        malformed += malformed_items
        for call in parsed_items:
            key = (call.callee, call.line, call.qualifier, call.in_function)
            if key in seen:
                continue
            seen.add(key)
            collected.append(call)

    return collected, malformed, unsupported


def _parse_call_items(
    raw_calls: object,
    default_in_function: str,
) -> tuple[list[CallSite], int]:
    """Parse raw call payload(s) into CallSite objects."""
    if isinstance(raw_calls, (str, dict)):
        raw_items = [raw_calls]
    elif isinstance(raw_calls, list):
        raw_items = raw_calls
    else:
        return [], 1

    calls: list[CallSite] = []
    malformed = 0
    for item in raw_items:
        if isinstance(item, str):
            callee = item.strip()
            if callee:
                calls.append(
                    CallSite(
                        callee=callee,
                        line=0,
                        in_function=default_in_function,
                    )
                )
            else:
                malformed += 1
            continue

        if not isinstance(item, dict):
            malformed += 1
            continue

        callee_raw = item.get("callee") or item.get("name") or item.get("target")
        if not isinstance(callee_raw, str):
            malformed += 1
            continue
        callee = callee_raw.strip()
        if not callee:
            malformed += 1
            continue

        qualifier_raw = item.get("qualifier") or item.get("qual")
        qualifier = qualifier_raw.strip() if isinstance(qualifier_raw, str) else ""

        in_function_raw = item.get("in_function") or item.get("source_qn")
        if isinstance(in_function_raw, str) and in_function_raw.strip():
            in_function = in_function_raw.strip()
        else:
            in_function = default_in_function

        line_raw = item.get("line", 0)
        line = line_raw if isinstance(line_raw, int) else 0

        calls.append(
            CallSite(
                callee=callee,
                line=line,
                qualifier=qualifier,
                in_function=in_function,
            )
        )

    return calls, malformed


def _pass_store(
    db: Store,
    project: str,
    repo_path: str,
    records: list[NodeRecord],
    edges: list[tuple[str, str, str, dict]],
    file_contents: dict[str, str],
    file_hashes: list[tuple[str, str, int, int]],
    file_languages: dict[str, str | None] | None = None,
) -> tuple[dict[str, int], int]:
    """
    Passes 7-9: Write all data to the store in a single bulk transaction.

    Sequence:
        db.begin_bulk()
        db.drop_indexes()
        db.begin()
          db.upsert_project()
          db.insert_nodes()      → qn_to_id
          db.insert_edges()
          db.insert_files()
          db.insert_file_hashes()
        db.commit()
        db.create_indexes()
        db.end_bulk()
        db.checkpoint()

    Args:
        db:            open Store (in-memory or file-backed)
        project:       project name
        repo_path:     absolute repo root (stored on the project record)
        records:       all NodeRecords with QNs assigned
        edges:         (source_qn, target_qn, type, props) tuples
        file_contents: dict of rel_path → source text
        file_hashes:   list of (rel_path, sha256, mtime_ns, size_bytes)

    Returns:
        (qn_to_id, edges_inserted) tuple where qn_to_id maps
        qualified_name → SQLite row ID and edges_inserted is the count
        of edges that were actually written.
    """
    db.begin_bulk()
    db.drop_indexes()
    db.begin()
    db.upsert_project(project, repo_path)
    qn_to_id = db.insert_nodes(records, project)
    edges_inserted = db.insert_edges(edges, qn_to_id, project)
    db.insert_files(file_contents, project, file_languages)
    db.insert_file_hashes(file_hashes, project)
    db.commit()
    db.create_indexes()
    db.end_bulk()
    db.checkpoint()
    return qn_to_id, edges_inserted


def _pass_store_incremental(
    db: Store,
    project: str,
    repo_path: str,
    changed_records: list[NodeRecord],
    extraction_results: dict[str, ExtractionResult],
    changed_file_infos: list[FileInfo],
    removed_paths: list[str],
    config: PipelineConfig,
) -> tuple[
    list[NodeRecord],
    list[tuple[str, str, str, dict]],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """
    Apply an incremental update while preserving unchanged graph state.

    Steps:
      1. Upsert changed/new nodes with ID-preserving conflict updates.
      2. Delete disappeared symbols from changed files and all symbols
         from removed files.
      3. Update file contents/hashes only for changed/new files and
         remove rows for deleted files.
      4. Rebuild all edges from stored node properties.
    """
    changed_records_by_file = _group_records_by_file(changed_records)
    changed_file_languages = {fi.path: fi.language for fi in changed_file_infos}
    changed_file_contents = _collect_file_contents(
        changed_file_infos,
        extraction_results,
    )
    changed_file_hashes = _collect_file_hashes(
        changed_file_infos,
        changed_file_contents,
    )

    db.begin_bulk()
    db.drop_indexes()
    db.begin()
    db.upsert_project(project, repo_path)

    # Remove all symbols for files that disappeared from the repo.
    for removed_path in removed_paths:
        db.delete_nodes_for_file(project, removed_path)

    # For each changed file, remove symbols that no longer exist and
    # upsert symbols that remain/newly appear.
    for fi in changed_file_infos:
        path = fi.path
        existing_qns = db.get_qns_for_file(project, path)
        next_records = changed_records_by_file.get(path, [])
        next_qns = {record.qualified_name for record in next_records}
        stale_qns = sorted(existing_qns - next_qns)
        if stale_qns:
            db.delete_nodes_by_qns(project, stale_qns)
        if next_records:
            db.insert_nodes(next_records, project)

    if removed_paths:
        db.delete_files(project, removed_paths)
        db.delete_file_hashes(project, removed_paths)

    if changed_file_contents:
        db.insert_files(changed_file_contents, project, changed_file_languages)
    if changed_file_hashes:
        db.insert_file_hashes(changed_file_hashes, project)

    # Rebuild all relationships from stored call/import properties.
    all_records = db.get_node_records(project)
    reg = _pass_build_registry(all_records)
    stored_results = _results_from_records(all_records)
    (
        all_edges,
        calls_discovered,
        calls_resolved,
        calls_unresolved,
        calls_unsupported,
        malformed_payloads,
    ) = _pass_resolve_calls(
        stored_results,
        reg,
        project,
        config,
    )
    db.delete_edges_for_project(project)
    qn_to_id = db.get_qn_to_id(project)
    edges_inserted = db.insert_edges(all_edges, qn_to_id, project)

    db.commit()
    db.create_indexes()
    db.end_bulk()
    db.checkpoint()

    return (
        all_records,
        all_edges,
        edges_inserted,
        calls_discovered,
        calls_resolved,
        calls_unresolved,
        calls_unsupported,
        malformed_payloads,
    )


def _pass_dump(
    db: Store,
    db_path: str,
) -> None:
    """
    Pass 10: Persist the in-memory database to a file.

    Calls db.dump_to_file(db_path). Creates parent directories if
    needed.

    Args:
        db:      open in-memory Store
        db_path: absolute path to write the .db file
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db.dump_to_file(db_path)


def _pass_export(
    db_path: str,
    artifact_dir: str,
    compression_level: int,
) -> str:
    """
    Pass 11: Compress the database to a .zst artifact.

    Calls artifact.export(db_path, artifact_dir, compression_level).
    Returns the artifact path.

    Args:
        db_path:           absolute path to the .db file
        artifact_dir:      directory to write graph.db.zst into
        compression_level: zstd compression level

    Returns:
        Absolute path to the written .zst file.
    """

    stats = export(db_path, artifact_dir, compression_level)
    return stats.artifact_path


# ---------------------------------------------------------------------------
# Incremental helpers
# ---------------------------------------------------------------------------


def _load_stored_hashes(
    project: str,
    config: PipelineConfig,
) -> dict[str, tuple[str, int, int]]:
    """
    Load file hashes from the existing working database for a project.

    Opens the database at default_db_path(project, config.cache_dir) in
    read-only mode and calls store.get_file_hashes(). If the database
    does not exist, returns an empty dict (treat all files as changed).

    Args:
        project: project name
        config:  resolved PipelineConfig

    Returns:
        Dict mapping rel_path → (sha256_hex, mtime_ns, size_bytes).
        Empty dict if no previous database exists.
    """
    db_path = default_db_path(project, config.cache_dir)
    db: Store | None = None
    try:
        db = open_path_readonly(db_path)
        return db.get_file_hashes(project)
    except StoreFileNotFoundError:
        return {}
    finally:
        if db is not None:
            db.close()


def _delete_stale_nodes(
    db: Store,
    project: str,
    changed_paths: list[str],
) -> int:
    """
    Delete nodes and edges for files that are being re-indexed.

    Calls db.delete_nodes_for_file() for each path in changed_paths.
    Should be called inside the bulk transaction before inserting new
    nodes, so the database never has both old and new nodes for the
    same file simultaneously.

    Args:
        db:            open Store
        project:       project name
        changed_paths: list of repo-relative file paths

    Returns:
        Total number of nodes deleted across all files.
    """
    total_deleted = 0
    db.begin_bulk()
    for path in changed_paths:
        total_deleted += db.delete_nodes_for_file(project, path)
    db.end_bulk()
    return total_deleted


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _read_file(abs_path: str) -> tuple[str, str]:
    """
    Read a file from disk, returning (content, error_message).

    Uses UTF-8 decoding with 'replace' error handling so non-UTF-8
    bytes produce replacement characters rather than raising.

    Args:
        abs_path: absolute filesystem path

    Returns:
        (content, "") on success.
        ("", error_message) on any OSError or exception.

    Examples:
        >>> content, err = _read_file("/repo/src/service.py")
        >>> err
        ''
        >>> content, err = _read_file("/nonexistent/file.py")
        >>> content
        ''
        >>> len(err) > 0
        True
    """
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), ""
    except Exception as e:
        return "", str(e)


def _compute_sha256(content: str) -> str:
    """
    Compute the SHA-256 hex digest of a string's UTF-8 encoding.

    Used by _pass_filter_unchanged() to detect content changes when
    mtime alone is not reliable (e.g. after git checkout).

    Args:
        content: file content string

    Returns:
        Lowercase hex string, e.g.
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    Examples:
        >>> _compute_sha256("")
        'e3b0c44298fc1c149afbf4c8996fb924...'  # sha256 of empty string
        >>> len(_compute_sha256("hello"))
        64
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _collect_file_contents(
    file_infos: list[FileInfo],
    extraction_results: dict[str, ExtractionResult],
) -> dict[str, str]:
    """
    Build a dict of rel_path → source content for all successfully read files.

    Reads file content from each FileInfo.abs_path. Files that failed
    during extraction (extractor="skip", reason="read_error") are
    excluded.

    Used by _pass_store() to populate the files table so get_file_source()
    works for any file in the repo, not just those with extracted symbols.

    Args:
        file_infos:         list of FileInfo objects
        extraction_results: ExtractionResult dict from _pass_extract()

    Returns:
        Dict mapping repo-relative path → source text. Files that could
        not be read are absent from the dict.
    """
    contents: dict[str, str] = {}
    for fi in file_infos:
        result = extraction_results.get(fi.path)
        if result and result.extractor == "skip" and result.reason == "read_error":
            continue
        content, err = _read_file(fi.abs_path)
        if not err:
            contents[fi.path] = content
    return contents


def _collect_file_hashes(
    file_infos: list[FileInfo],
    file_contents: dict[str, str],
) -> list[tuple[str, str, int, int]]:
    """
    Build file hash records for all files with available content.

    Computes sha256 for each file in file_contents and pairs it with
    mtime_ns and size_bytes from the corresponding FileInfo.

    Args:
        file_infos:    list of FileInfo objects (for mtime_ns, size_bytes)
        file_contents: dict of rel_path → source text

    Returns:
        List of (rel_path, sha256_hex, mtime_ns, size_bytes) tuples.
    """
    return [
        (
            fi.path,
            _compute_sha256(file_contents[fi.path]),
            fi.mtime_ns,
            fi.size_bytes,
        )
        for fi in file_infos
        if fi.path in file_contents
    ]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _count_by_label(records: list[NodeRecord]) -> dict[str, int]:
    """
    Count NodeRecords grouped by label.

    Args:
        records: list of NodeRecord objects

    Returns:
        Dict mapping label → count, e.g.
        {"Function": 120, "Class": 30, "Method": 85, "File": 12}
    """
    return {record.label: 0 for record in records} | {
        record.label: sum(1 for r in records if r.label == record.label)
        for record in records
    }


def _count_by_edge_type(
    edges: list[tuple[str, str, str, dict]],
) -> dict[str, int]:
    """
    Count edges grouped by edge type.

    Args:
        edges: list of (source_qn, target_qn, type, properties) tuples

    Returns:
        Dict mapping edge type → count, e.g.
        {"CALLS": 200, "IMPORTS": 50}
    """
    return {edge_type: 0 for _, _, edge_type, _ in edges} | {
        edge_type: sum(1 for _, _, et, _ in edges if et == edge_type)
        for _, _, edge_type, _ in edges
    }


def _progress(
    config: PipelineConfig,
    pass_name: str,
    current: int,
    total: int,
) -> None:
    """
    Emit a progress callback if one is configured.

    No-op when config.on_progress is None.

    Args:
        config:     PipelineConfig with optional on_progress callback
        pass_name:  short name of the current pass, e.g. "extract"
        current:    number of items completed so far
        total:      total number of items in this pass
    """
    if config.on_progress:
        try:
            config.on_progress(pass_name, current, total)
        except Exception as e:
            logger.warning("Progress callback raised an exception: %s", str(e))


def _log_result(result: PipelineResult) -> None:
    """
    Emit a structured INFO-level log summary of a completed pipeline run.

    Logs: project, elapsed time, files (discovered/extracted/skipped),
    nodes (total, by label), edges (total, by type), resolution stats,
    artifact path.

    Args:
        result: completed PipelineResult
    """
    logger.info(
        (
            "Pipeline completed: project=%s elapsed=%.2fs files=%d/%d/%d "
            "nodes=%d edges=%d calls=%d/%d artifact=%s"
        ),
        result.project,
        result.elapsed_seconds,
        result.files_extracted,
        result.files_discovered,
        result.files_skipped,
        result.nodes_total,
        result.edges_total,
        result.calls_resolved,
        result.calls_unresolved,
        result.artifact_path or "(not exported)",
    )
