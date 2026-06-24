"""
extractor.py — Unified file extraction entry point.

Routes each file to the appropriate extractor and returns a list of
NodeRecord objects. This is the only module the pipeline needs to call
— it never imports treesitter.py or fallback.py directly.

Routing logic:
    1. Check should_skip() — return [] immediately for lock files,
       binaries, minified bundles, and files exceeding MAX_FILE_BYTES.
    2. Detect language via detect_language().
    3. If language is recognised and has a LANG_CONFIG entry, call
       treesitter.extract().
    4. If the tree-sitter result is non-empty, return it.
    5. If the tree-sitter result is empty (file has no extractable
       definitions, e.g. a file of only imports or constants), fall back
       to extract_fallback() with reason="no_definitions".
    6. If language is None (unrecognised extension), call
       extract_fallback() with reason="no_language".
    7. On any exception from tree-sitter, log a warning and fall back
       to extract_fallback() with reason="parse_error".

The qualified_name field on every returned NodeRecord is always "".
The pipeline (pipeline.py) sets it via fqn.compute() or fqn.module()
once the project name is known.

Public API:
    extract_file(path, source)  →  list[NodeRecord]
        Main entry point. Always returns a list, never raises.

    extract_files(file_infos, read_file)  →  dict[str, list[NodeRecord]]
        Batch extraction over a list of FileInfo objects. Reads file
        content via the provided callable (injectable for testing).
        Returns a dict mapping repo-relative path → list[NodeRecord].

Supporting types:
    FileInfo    dataclass produced by walker.py
    ExtractionResult  dataclass wrapping NodeRecord list + metadata
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

from .fallback import MAX_FILE_BYTES, extract_fallback, should_skip
from .languages import EXTENSION_TO_LANG, LANG_CONFIG
from .relations import extract_file_imports
from .treesitter import NodeRecord, _get_parser
from .treesitter import extract as ts_extract

logger = logging.getLogger(__name__)


def _detect_lang(path: str) -> str | None:
    """Detect language from path, returning None for unrecognised extensions."""

    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return EXTENSION_TO_LANG.get(ext)


# ---------------------------------------------------------------------------
# FileInfo
# ---------------------------------------------------------------------------


@dataclass
class FileInfo:
    """
    Metadata for a single file produced by walker.py.

    Attributes:
        path:       repo-relative file path, e.g. "src/payments/service.py"
        abs_path:   absolute path on disk, e.g. "/home/user/repo/src/..."
        language:   detected language name or None for unrecognised files.
                    Set by walker.py via detect_language(); the extractor
                    uses this rather than re-detecting.
        size_bytes: file size in bytes at discovery time.
        mtime_ns:   modification time in nanoseconds since epoch.
                    Used by the incremental pipeline to detect changes
                    without reading file content.
    """

    path: str
    abs_path: str
    language: str | None
    size_bytes: int = 0
    mtime_ns: int = 0


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """
    Output of extract_file() for one file.

    Wraps the list of NodeRecords with metadata useful for the pipeline
    and for debugging / progress reporting.

    Attributes:
        path:       repo-relative file path (matches FileInfo.path).
        records:    list of NodeRecord objects. Empty list means no
                    symbols were extracted (fallback may have run).
        language:   detected language or "unknown".
        extractor:  which extractor ran: "treesitter" | "fallback" | "skip".
        reason:     why fallback ran, or "" for tree-sitter success.
                    Possible values: "no_language" | "no_definitions" |
                    "parse_error" | "skipped" | "".
        error:      exception message if tree-sitter raised, else "".
        node_count: len(records) — convenience accessor.
    """

    path: str
    records: list[NodeRecord] = field(default_factory=list)
    language: str = "unknown"
    extractor: str = "treesitter"  # treesitter | fallback | skip
    reason: str = ""
    error: str = ""

    @property
    def node_count(self) -> int:
        """Number of NodeRecords extracted."""
        return len(self.records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_file(path: str, source: str) -> list[NodeRecord]:
    """
    Extract NodeRecords from a single file.

    This is the lowest-level public entry point. The pipeline calls this
    directly when it already has file content in memory. walker.py reads
    content from disk; the pipeline then calls extract_file().

    Routing order:
        1. should_skip()          → return []
        2. detect_language()
           a. recognised lang     → ts_extract()
              - non-empty result  → return records
              - empty result      → fallback(reason="no_definitions")
           b. unrecognised        → fallback(reason="no_language")
        3. ts_extract() raises    → fallback(reason="parse_error")

    The qualified_name on every NodeRecord is "". The pipeline sets it.

    Args:
        path:   repo-relative file path. Used for language detection,
                stored on each NodeRecord as file_path, and included in
                log messages.
        source: full UTF-8 file content. Must be a str, not bytes.

    Returns:
        List of NodeRecord objects, possibly empty. Never raises.

    Examples:
        >>> records = extract_file("src/payments/service.py", source)
        >>> records[0].label
        'Function'
        >>> records[0].qualified_name
        ''

        >>> extract_file("package-lock.json", "{}")
        []

        >>> records = extract_file("Dockerfile", dockerfile_source)
        >>> records[0].label
        'File'
        >>> records[0].properties["fallback"]
        True
    """
    if should_skip(path, source):
        return []
    result = _route(path, source, _detect_lang(path))
    return result.records


def extract_file_detailed(path: str, source: str) -> ExtractionResult:
    """
    Extract NodeRecords and return full extraction metadata.

    Like extract_file() but wraps the result in an ExtractionResult
    that records which extractor ran, why fallback was used (if at all),
    and any error message. Used by the pipeline for progress reporting
    and by tests to assert routing behaviour.

    Args:
        path:   repo-relative file path
        source: full UTF-8 file content

    Returns:
        ExtractionResult with records, extractor, reason, and error set.
        result.extractor is one of: "treesitter" | "fallback" | "skip".
        result.records is [] when extractor="skip".

    Examples:
        >>> result = extract_file_detailed("src/auth.py", source)
        >>> result.extractor
        'treesitter'
        >>> result.reason
        ''

        >>> result = extract_file_detailed("package-lock.json", "{}")
        >>> result.extractor
        'skip'
        >>> result.records
        []

        >>> result = extract_file_detailed("Dockerfile", source)
        >>> result.extractor
        'fallback'
        >>> result.reason
        'no_language'
    """
    if should_skip(path, source):
        return ExtractionResult(
            path=path,
            records=[],
            language=_detect_lang(path) or "unknown",
            extractor="skip",
            reason="skipped",
        )
    return _route(path, source, _detect_lang(path))


def extract_files(
    file_infos: list[FileInfo],
    read_file: Callable[[str], str],
) -> dict[str, ExtractionResult]:
    """
    Batch-extract NodeRecords for a list of files.

    Calls extract_file_detailed() for each FileInfo. File content is
    obtained by calling read_file(abs_path) — this indirection makes
    the function testable without touching disk.

    Skips files where FileInfo.size_bytes exceeds fallback.MAX_FILE_BYTES
    without calling read_file() — avoids reading large files into memory.

    If read_file() raises (e.g. PermissionError, UnicodeDecodeError),
    the file is logged as a warning and an ExtractionResult with
    extractor="skip", reason="read_error" is stored for that path.

    Args:
        file_infos: list of FileInfo objects from walker.py
        read_file:  callable that takes an absolute path string and
                    returns the file content as a UTF-8 string.
                    Typically: lambda p: Path(p).read_text(errors="replace")

    Returns:
        Dict mapping repo-relative path → ExtractionResult.
        Every path in file_infos has an entry — no paths are silently
        dropped.

    Examples:
        >>> from pathlib import Path
        >>> results = extract_files(
        ...     file_infos,
        ...     read_file=lambda p: Path(p).read_text(errors="replace"),
        ... )
        >>> results["src/payments/service.py"].extractor
        'treesitter'
        >>> results["Dockerfile"].extractor
        'fallback'
        >>> results["package-lock.json"].extractor
        'skip'
    """
    results: dict[str, ExtractionResult] = {}
    for fi in file_infos:
        if fi.size_bytes > MAX_FILE_BYTES:
            results[fi.path] = ExtractionResult(
                path=fi.path,
                records=[],
                language=fi.language or "unknown",
                extractor="skip",
                reason="skipped",
            )
            continue
        try:
            content = read_file(fi.abs_path)
        except Exception as err:
            logger.warning("Could not read %s: %s", fi.path, err)
            results[fi.path] = ExtractionResult(
                path=fi.path,
                records=[],
                language=fi.language or "unknown",
                extractor="skip",
                reason="read_error",
                error=str(err),
            )
            continue
        results[fi.path] = extract_file_detailed(fi.path, content)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _route(
    path: str,
    source: str,
    language: str | None,
) -> ExtractionResult:
    """
    Core routing logic shared by extract_file() and extract_file_detailed().

    Separated from the public API so it can be tested independently of
    the should_skip() pre-check, which is applied before _route() is
    called.

    Args:
        path:     repo-relative file path
        source:   full UTF-8 file content (should_skip already passed)
        language: detected language from detect_language(), or None

    Returns:
        ExtractionResult. extractor is "treesitter" or "fallback".
        Never "skip" — the caller handles skip before calling _route().
    """
    if language is None or language not in LANG_CONFIG:
        return _run_fallback(path, source, language or "unknown", "no_language")
    return _run_treesitter(path, source, language)


def _run_treesitter(path: str, source: str, language: str) -> ExtractionResult:
    """
    Run tree-sitter extraction and return an ExtractionResult.

    Catches all exceptions from ts_extract() and converts them to a
    fallback result with reason="parse_error". This ensures a broken
    grammar or unexpected AST structure never propagates to the pipeline.

    Args:
        path:     repo-relative file path
        source:   full UTF-8 file content
        language: canonical language name (used for the result only;
                  ts_extract() re-detects from path)

    Returns:
        ExtractionResult with extractor="treesitter" on success,
        or extractor="fallback", reason="parse_error" on exception.
    """
    try:
        records = ts_extract(path, source)
    except Exception as exc:
        logger.warning("tree-sitter failed on %s: %s", path, exc)
        return _run_fallback(path, source, language, "parse_error", error=str(exc))

    if records:
        return ExtractionResult(
            path=path,
            records=records,
            language=language,
            extractor="treesitter",
        )

    fallback = _run_fallback(path, source, language, "no_definitions")
    _attach_file_imports_to_fallback(path, source, language, fallback.records)
    return fallback


def _attach_file_imports_to_fallback(
    path: str,  # noqa: ARG001
    source: str,
    language: str,
    records: list[NodeRecord],
) -> None:
    """Attach extracted file-scope imports to fallback records when available."""
    if not records or not source.strip() or language not in LANG_CONFIG:
        return

    try:
        parser_name = str(LANG_CONFIG[language]["parser"])
        parser = _get_parser(parser_name)
        tree = parser.parse(bytes(source, "utf8"))
        imports = extract_file_imports(tree.root_node, language, source.splitlines())
    except Exception:
        return

    if not imports:
        return

    for record in records:
        merged_imports = list(record.properties.get("imports", []))  # type: ignore
        merged_imports.extend(imports)
        record.properties["imports"] = merged_imports


def _run_fallback(
    path: str,
    source: str,
    language: str,
    reason: str,
    error: str = "",
) -> ExtractionResult:
    """
    Run fallback extraction and return an ExtractionResult.

    Wraps extract_fallback() and packages its output into an
    ExtractionResult with the given reason and error string.

    Args:
        path:     repo-relative file path
        source:   full UTF-8 file content
        language: detected language or "unknown"
        reason:   why fallback is running: "no_language" |
                  "no_definitions" | "parse_error"
        error:    exception message if reason="parse_error", else ""

    Returns:
        ExtractionResult with extractor="fallback".
    """
    records = extract_fallback(path, source, reason=reason)
    return ExtractionResult(
        path=path,
        records=records,
        language=language,
        extractor="fallback",
        reason=reason,
        error=error,
    )


def _read_file_safe(abs_path: str) -> tuple[str, str]:
    """
    Read a file from disk, returning (content, error_message).

    Attempts UTF-8 decoding with 'replace' error handling to tolerate
    files with non-UTF-8 bytes (common in legacy codebases).

    Args:
        abs_path: absolute filesystem path to read

    Returns:
        (content, "") on success.
        ("", error_message) on any exception (PermissionError,
        FileNotFoundError, OSError, etc.).

    Examples:
        >>> content, err = _read_file_safe("/path/to/file.py")
        >>> err
        ''
        >>> content, err = _read_file_safe("/nonexistent/path")
        >>> content
        ''
        >>> len(err) > 0
        True
    """
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            return fh.read(), ""
    except Exception as exc:
        return "", str(exc)
