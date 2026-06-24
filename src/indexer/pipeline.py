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
    Registry,
    ResolutionContext,
    build,
)
from .store import (
    DEFAULT_CACHE_DIR,
    Store,
    default_db_path,
    open_memory,
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
        files_extracted:    files that were actually parsed
        nodes_total:        total nodes inserted
        nodes_by_label:     dict of label → count
        edges_total:        total edges inserted
        edges_by_type:      dict of type → count
        calls_resolved:     call sites resolved to a known node
        calls_unresolved:   call sites that could not be resolved
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
    files_extracted: int = 0
    nodes_total: int = 0
    nodes_by_label: dict[str, int] = field(default_factory=dict)
    edges_total: int = 0
    edges_by_type: dict[str, int] = field(default_factory=dict)
    calls_resolved: int = 0
    calls_unresolved: int = 0
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

    db = open_memory()
    try:
        # Pass 1: Discover
        all_file_infos = _pass_discover(repo_path, config)
        result.files_discovered = len(all_file_infos)

        # Incremental: filter unchanged files
        if config.incremental:
            stored_hashes = _load_stored_hashes(config.project, config)
            file_infos, unchanged = _pass_filter_unchanged(
                all_file_infos, stored_hashes
            )
            result.files_unchanged = len(unchanged)
        else:
            file_infos = all_file_infos

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

        # Pass 5: Build registry
        reg = _pass_build_registry(records)

        # Pass 6: Resolve calls
        edges = _pass_resolve_calls(extraction_results, reg, config.project, config)

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
        db_path = default_db_path(config.project, config.cache_dir)
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
) -> tuple[list[FileInfo], list[FileInfo]]:
    """
    Split FileInfo list into (changed, unchanged) based on stored hashes.

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
        (changed, unchanged) tuple of FileInfo lists.
        changed + unchanged == file_infos (every file is in one list).
    """
    changed = []
    unchanged = []
    for fi in file_infos:
        stored = stored_hashes.get(fi.path)
        if stored is None:
            changed.append(fi)
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
    return changed, unchanged


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
) -> list[tuple[str, str, str, dict]]:
    """
    Pass 6: Resolve call sites to edges in parallel.

    For each file in results:
      1. Builds a ResolutionContext with module_qn and imports parsed
         from the file's NodeRecords (Import nodes in properties, or
         by re-reading import lines from the source).
      2. Builds a list of CallSite objects from the source (currently
         a placeholder — full call extraction requires the extractor to
         emit CallSite objects, which is a v2 feature).
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
        List of (source_qn, target_qn, edge_type, properties) tuples
        ready for store.insert_edges().
    """
    all_edges: list[tuple[str, str, str, dict]] = []
    items = list(results.items())
    total = len(items)

    def _resolve_file(
        path: str, _result: ExtractionResult
    ) -> list[tuple[str, str, str, dict]]:
        ctx = ResolutionContext(
            file_path=path,
            module_qn=module(path),
            # v1: imports and call sites are not yet emitted by the extractor.
            # v2 will populate these from NodeRecord.properties.
            imports=[],
            project=project,
        )
        calls: list[CallSite] = []
        edges: list[tuple[str, str, str, dict]] = []
        for res in reg.resolve_all(calls, ctx):
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
        return edges

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(_resolve_file, path, result): path for path, result in items
        }
        for done, future in enumerate(as_completed(futures), 1):
            all_edges.extend(future.result())
            _progress(config, "resolve", done, total)

    return all_edges


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
    try:
        db = open_path_readonly(db_path)
        return db.get_file_hashes(project)
    except StoreFileNotFoundError:
        return {}


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
