"""
artifact.py — Compressed artifact export and import.

Packages the working SQLite database into a portable, compressed
artifact suitable for committing alongside a repository, sharing with
teammates, or passing between CI runs.

Artifact layout on disk:
    <artifact_dir>/
        graph.db.zst        — zstd-compressed SQLite database
        artifact.json       — metadata: project, schema_ver, timestamps,
                              node/edge counts, compression stats
        .gitattributes      — marks graph.db.zst as binary + merge=ours

The compressed database is a straight zstd stream over the output of
SQLite's VACUUM INTO — a clean, page-aligned copy of the database with
no WAL frames and no free-list pages. This gives the best compression
ratio (typically 8-15x for source code repositories).

Public API:
    export(db_path, artifact_dir, compression_level) → str
        Compress db_path to artifact_dir/graph.db.zst.
        Returns the absolute path of the written artifact.

    import_artifact(artifact_dir, dest_db_path) → ArtifactMeta
        Decompress artifact_dir/graph.db.zst to dest_db_path.
        Returns the parsed ArtifactMeta.

    read_meta(artifact_dir) → ArtifactMeta | None
        Parse artifact.json without decompressing the database.

    artifact_exists(artifact_dir) → bool
        Return True if graph.db.zst is present in artifact_dir.

Supporting types:
    ArtifactMeta    — parsed contents of artifact.json
    ExportStats     — compression statistics returned by export()
"""

import json
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .errors import (
    ArtifactNotFoundError,
    InvalidArtifactError,
    InvalidStoreFileError,
    MetadataNotFoundError,
    StoreFileNotFoundError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACT_DB_NAME: str = "graph.db.zst"
ARTIFACT_META_NAME: str = "artifact.json"
GITATTRIBUTES_NAME: str = ".gitattributes"

# Default zstd compression level. Level 9 gives a good ratio/speed
# tradeoff for CI usage. Level 19+ gives maximum compression but is
# significantly slower.
DEFAULT_COMPRESSION_LEVEL: int = 9

# Minimum valid SQLite file starts with this magic header.
SQLITE_MAGIC: bytes = b"SQLite format 3\x00"

# Current artifact format version. Increment when the artifact.json
# schema changes in a backward-incompatible way.
ARTIFACT_FORMAT_VERSION: int = 1


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ArtifactMeta:
    """
    Parsed contents of artifact.json.

    Written by export() and read back by read_meta() and
    import_artifact(). All fields are JSON-serialisable.

    Attributes:
        format_version:     artifact format version (ARTIFACT_FORMAT_VERSION)
        schema_version:     SQLite schema version from the projects table
        project:            project name, e.g. "my-app"
        exported_at:        ISO-8601 UTC timestamp of the export
        compression_level:  zstd level used during compression
        uncompressed_bytes: size of the raw .db file before compression
        compressed_bytes:   size of the .zst file after compression
        compression_ratio:  uncompressed / compressed (e.g. 10.3)
        node_count:         total nodes in the database
        edge_count:         total edges in the database
        file_count:         total files in the files table
        language_counts:    dict of language → file count
        indexer_version:    version string of the indexer that wrote
                            this artifact. "" if not set.
    """

    format_version: int = ARTIFACT_FORMAT_VERSION
    schema_version: int = 1
    project: str = ""
    exported_at: str = ""
    compression_level: int = DEFAULT_COMPRESSION_LEVEL
    uncompressed_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    file_count: int = 0
    language_counts: dict[str, int] = field(default_factory=dict)
    indexer_version: str = ""


@dataclass
class ExportStats:
    """
    Compression statistics returned by export().

    Attributes:
        artifact_path:      absolute path to the written .zst file
        meta_path:          absolute path to the written artifact.json
        uncompressed_bytes: size of the raw database before compression
        compressed_bytes:   size of the artifact after compression
        compression_ratio:  uncompressed / compressed
        elapsed_seconds:    wall-clock time for the export operation
    """

    artifact_path: str
    meta_path: str
    uncompressed_bytes: int
    compressed_bytes: int
    compression_ratio: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export(
    db_path: str,
    artifact_dir: str,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
) -> ExportStats:
    """
    Compress a SQLite database to a .zst artifact.

    Steps:
        1. VACUUM INTO a temporary clean copy of db_path.
           This removes WAL frames, free-list pages, and unused space,
           ensuring the best possible compression ratio.
        2. zstd-compress the clean copy to
           <artifact_dir>/graph.db.zst.
        3. Collect metadata (node/edge/file counts, language breakdown)
           from the clean copy.
        4. Write <artifact_dir>/artifact.json.
        5. Write <artifact_dir>/.gitattributes if not already present.
        6. Delete the temporary clean copy.

    The original db_path is never modified.

    Args:
        db_path:           absolute path to the working .db file.
                           Must be a valid SQLite database.
        artifact_dir:      directory to write the artifact into.
                           Created (including parents) if it does not
                           exist.
        compression_level: zstd compression level (1-22).
                           Defaults to DEFAULT_COMPRESSION_LEVEL (9).

    Returns:
        ExportStats with paths, byte counts, ratio, and elapsed time.

    Raises:
        FileNotFoundError: if db_path does not exist.
        ValueError:        if db_path is not a valid SQLite database.
        OSError:           if artifact_dir cannot be created or the
                           artifact file cannot be written.

    Examples:
        >>> stats = export("/home/user/.cache/codebase-indexer/my-app.db",
        ...                "/path/to/repo/.codebase-index")
        >>> stats.compression_ratio
        10.3
        >>> Path(stats.artifact_path).exists()
        True
    """
    if not Path(db_path).exists():
        raise StoreFileNotFoundError(db_path)
    if not _validate_sqlite(db_path):
        raise InvalidStoreFileError(db_path)

    t_start = time.monotonic()
    _ensure_dir(artifact_dir)

    artifact_path = str(Path(artifact_dir) / ARTIFACT_DB_NAME)
    meta_path = str(Path(artifact_dir) / ARTIFACT_META_NAME)

    # Step 1: VACUUM INTO a clean temp copy
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        clean_path = tmp.name
    try:
        Path(clean_path).unlink(
            missing_ok=True
        )  # VACUUM INTO requires non-existing dest
        _vacuum_into(db_path, clean_path)

        # Step 2: Compress
        uncompressed_bytes, compressed_bytes = _compress_file(
            clean_path, artifact_path, compression_level
        )

        # Step 3+4: Collect metadata and write artifact.json
        project = _guess_project(db_path)
        meta = _collect_meta(
            clean_path, project, compression_level, uncompressed_bytes, compressed_bytes
        )
        meta_path = _write_meta(meta, artifact_dir)

        # Step 5: .gitattributes
        _write_gitattributes(artifact_dir)
    finally:
        # Step 6: Remove temp clean copy
        Path(clean_path).unlink(missing_ok=True)

    elapsed = time.monotonic() - t_start
    return ExportStats(
        artifact_path=artifact_path,
        meta_path=meta_path,
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=_compression_ratio(uncompressed_bytes, compressed_bytes),
        elapsed_seconds=elapsed,
    )


def import_artifact(
    artifact_dir: str,
    dest_db_path: str,
) -> ArtifactMeta:
    """
    Decompress a .zst artifact to a working SQLite database file.

    Steps:
        1. Read and validate artifact.json (format_version check).
        2. Decompress <artifact_dir>/graph.db.zst to a temporary file.
        3. Validate the decompressed file is a valid SQLite database
           by checking the SQLITE_MAGIC header.
        4. Move the temporary file to dest_db_path (atomic on POSIX).
        5. Return the parsed ArtifactMeta.

    The destination file is only written if decompression succeeds.
    If any step fails, no file is written at dest_db_path.

    Args:
        artifact_dir:  directory containing graph.db.zst and
                       artifact.json.
        dest_db_path:  absolute path to write the decompressed .db file.
                       Parent directories are created if needed.

    Returns:
        ArtifactMeta parsed from artifact.json.

    Raises:
        FileNotFoundError: if graph.db.zst or artifact.json are missing
                           from artifact_dir.
        ValueError:        if the artifact format_version is
                           incompatible, or if the decompressed file
                           fails the SQLite magic header check.
        OSError:           if dest_db_path cannot be written.

    Examples:
        >>> meta = import_artifact(
        ...     "/path/to/repo/.codebase-index",
        ...     "/home/user/.cache/codebase-indexer/my-app.db",
        ... )
        >>> meta.project
        'my-app'
        >>> meta.node_count
        247
    """
    artifact_path = str(Path(artifact_dir) / ARTIFACT_DB_NAME)
    meta_path = str(Path(artifact_dir) / ARTIFACT_META_NAME)

    if not Path(artifact_path).exists():
        raise ArtifactNotFoundError(artifact_path)
    if not Path(meta_path).exists():
        raise MetadataNotFoundError(meta_path)

    meta = _read_meta_file(meta_path)
    if meta is None:
        raise InvalidArtifactError(
            meta_path, f"failed to parse artifact metadata: {meta_path}"
        )
    if meta.format_version > ARTIFACT_FORMAT_VERSION:
        raise InvalidArtifactError(
            meta_path,
            f"Artifact format version {meta.format_version} is not supported "
            f"(max supported: {ARTIFACT_FORMAT_VERSION})",
        )

    _ensure_dir(str(Path(dest_db_path).parent))

    with tempfile.NamedTemporaryFile(
        dir=str(Path(dest_db_path).parent), suffix=".db.tmp", delete=False
    ) as tmp:
        tmp_path = tmp.name

    def throw_err():
        raise InvalidArtifactError(
            tmp_path, "Decompressed artifact failed SQLite magic header check"
        )

    try:
        _decompress_file(artifact_path, tmp_path)
        if not _validate_sqlite(tmp_path):
            throw_err()
        shutil.move(tmp_path, dest_db_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    return meta


def read_meta(artifact_dir: str) -> ArtifactMeta | None:
    """
    Parse artifact.json without decompressing the database.

    Useful for checking what is in an artifact before deciding whether
    to import it (e.g. comparing schema_version or exported_at).

    Args:
        artifact_dir: directory containing artifact.json

    Returns:
        ArtifactMeta if artifact.json exists and parses successfully.
        None if artifact.json does not exist or cannot be parsed.

    Examples:
        >>> meta = read_meta("/path/to/repo/.codebase-index")
        >>> meta.project
        'my-app'
        >>> read_meta("/nonexistent/dir") is None
        True
    """
    return _read_meta_file(str(Path(artifact_dir) / ARTIFACT_META_NAME))


def artifact_exists(artifact_dir: str) -> bool:
    """
    Return True if a valid artifact is present in artifact_dir.

    Checks for the presence of both graph.db.zst and artifact.json.
    Does not decompress or validate the database contents.

    Args:
        artifact_dir: directory to check

    Returns:
        True if both files exist, False otherwise.

    Examples:
        >>> artifact_exists("/path/to/repo/.codebase-index")
        True
        >>> artifact_exists("/path/to/empty/dir")
        False
    """
    d = Path(artifact_dir)
    return (d / ARTIFACT_DB_NAME).exists() and (d / ARTIFACT_META_NAME).exists()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _vacuum_into(src_db_path: str, dest_path: str) -> None:
    """
    Use SQLite's VACUUM INTO to create a clean copy of the database.

    VACUUM INTO writes a defragmented, WAL-free copy of the database to
    dest_path. This is faster and produces better compression than a
    file copy because free-list pages and WAL frames are excluded.

    Args:
        src_db_path: absolute path to the source .db file
        dest_path:   absolute path for the output clean copy.
                     Must not already exist (SQLite requires this).

    Raises:
        sqlite3.Error: if VACUUM INTO fails.
    """
    conn = sqlite3.connect(src_db_path)
    try:
        conn.execute(f"VACUUM INTO '{dest_path}'")
    finally:
        conn.close()


def _compress_file(
    src_path: str,
    dest_path: str,
    level: int,
) -> tuple[int, int]:
    """
    zstd-compress src_path to dest_path.

    Reads src_path in chunks to avoid loading the entire file into
    memory. Uses zstandard.ZstdCompressor with the given level.

    Args:
        src_path:  absolute path to the file to compress
        dest_path: absolute path to write the compressed output.
                   Parent directory must already exist.
        level:     zstd compression level (1-22)

    Returns:
        (uncompressed_bytes, compressed_bytes) tuple.

    Raises:
        OSError: if src_path cannot be read or dest_path cannot be written.
    """
    import zstandard

    cctx = zstandard.ZstdCompressor(level=level)
    uncompressed = 0
    with (
        open(src_path, "rb") as src,
        open(dest_path, "wb") as dst,
        cctx.stream_writer(dst, closefd=False) as writer,
    ):
        while True:
            chunk = src.read(131072)  # 128 KB
            if not chunk:
                break
            writer.write(chunk)
            uncompressed += len(chunk)
    compressed = Path(dest_path).stat().st_size
    return uncompressed, compressed


def _decompress_file(
    src_path: str,
    dest_path: str,
) -> int:
    """
    zstd-decompress src_path to dest_path.

    Reads the .zst stream in chunks and writes the decompressed output
    to dest_path.

    Args:
        src_path:  absolute path to the .zst file
        dest_path: absolute path to write the decompressed output.
                   Parent directory must already exist.

    Returns:
        Number of decompressed bytes written.

    Raises:
        OSError:            if src_path cannot be read.
        zstandard.ZstdError: if the input is not a valid zstd stream.
    """
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    written = 0
    with (
        open(src_path, "rb") as src,
        open(dest_path, "wb") as dst,
        dctx.stream_reader(src) as reader,
    ):
        while True:
            chunk = reader.read(131072)
            if not chunk:
                break
            dst.write(chunk)
            written += len(chunk)
    return written


def _validate_sqlite(path: str) -> bool:
    """
    Return True if path begins with the SQLite file magic header.

    Reads only the first 16 bytes — does not open a connection.
    Used after decompression to guard against corrupted artifacts.

    Args:
        path: absolute path to the file to check

    Returns:
        True if the file starts with SQLITE_MAGIC, False otherwise
        (including if the file does not exist or cannot be read).

    Examples:
        >>> _validate_sqlite("/path/to/valid.db")
        True
        >>> _validate_sqlite("/path/to/random.bin")
        False
    """
    try:
        with open(path, "rb") as f:
            return f.read(16) == SQLITE_MAGIC
    except OSError:
        return False


def _collect_meta(
    clean_db_path: str,
    project: str,
    compression_level: int,
    uncompressed_bytes: int,
    compressed_bytes: int,
) -> ArtifactMeta:
    """
    Build an ArtifactMeta by querying the clean database copy.

    Opens the database read-only and queries:
      - node count (total and by label — labels stored in node_count)
      - edge count
      - file count
      - language breakdown from the files table

    The schema_version is read from the MAX(schema_ver) of the projects
    table (or defaults to 1 if the projects table is empty).

    Args:
        clean_db_path:      absolute path to the VACUUM INTO copy
        project:            project name (may be "" for multi-project dbs)
        compression_level:  zstd level used for this export
        uncompressed_bytes: size of the clean db before compression
        compressed_bytes:   size of the artifact after compression

    Returns:
        Fully populated ArtifactMeta with exported_at set to current
        UTC time in ISO-8601 format.
    """
    conn = sqlite3.connect(f"file:{clean_db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        lang_rows = conn.execute(
            "SELECT language, COUNT(*) AS cnt FROM files"
            " WHERE language IS NOT NULL GROUP BY language"
        ).fetchall()
        language_counts = {r["language"]: r["cnt"] for r in lang_rows}
        schema_ver_row = conn.execute("SELECT MAX(schema_ver) FROM projects").fetchone()
        schema_version = schema_ver_row[0] or 1
    finally:
        conn.close()

    ratio = _compression_ratio(uncompressed_bytes, compressed_bytes)
    return ArtifactMeta(
        format_version=ARTIFACT_FORMAT_VERSION,
        schema_version=schema_version,
        project=project,
        exported_at=_utc_now_iso(),
        compression_level=compression_level,
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        node_count=node_count,
        edge_count=edge_count,
        file_count=file_count,
        language_counts=language_counts,
    )


def _write_meta(meta: ArtifactMeta, artifact_dir: str) -> str:
    """
    Serialise ArtifactMeta to artifact.json in artifact_dir.

    Uses json.dumps with indent=2 for human readability. The file is
    written atomically: first to a .tmp file, then renamed.

    Args:
        meta:         ArtifactMeta to serialise
        artifact_dir: directory to write into

    Returns:
        Absolute path to the written artifact.json.

    Raises:
        OSError: if the file cannot be written.
    """
    artifact_dir_path = Path(artifact_dir)
    meta_path = artifact_dir_path / ARTIFACT_META_NAME
    tmp_path = artifact_dir_path / (ARTIFACT_META_NAME + ".tmp")
    data = json.dumps(vars(meta), indent=2)
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.rename(meta_path)
    return str(meta_path)


def _read_meta_file(meta_path: str) -> ArtifactMeta | None:
    """
    Read and parse a single artifact.json file.

    Converts the parsed dict to an ArtifactMeta dataclass. Unknown keys
    in the JSON are silently ignored (forward compatibility). Missing
    keys are filled with ArtifactMeta defaults.

    Args:
        meta_path: absolute path to artifact.json

    Returns:
        ArtifactMeta on success, None if the file cannot be read or
        parsed.
    """
    try:
        data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    defaults = vars(ArtifactMeta())
    defaults.update({k: v for k, v in data.items() if k in defaults})
    return ArtifactMeta(**defaults)


def _write_gitattributes(artifact_dir: str) -> None:
    """
    Write a .gitattributes file marking graph.db.zst as binary.

    Content written:
        # codebase-indexer artifact
        graph.db.zst binary merge=ours

    The "binary" attribute disables diff and merge for the file.
    "merge=ours" means git will always keep the current branch's
    version on merge conflicts, avoiding spurious conflict markers in
    binary files.

    Only writes the file if it does not already exist. Does not
    overwrite an existing .gitattributes so that project-specific
    settings are preserved.

    Args:
        artifact_dir: directory to write .gitattributes into
    """
    ga_path = Path(artifact_dir) / GITATTRIBUTES_NAME
    if not ga_path.exists():
        ga_path.write_text(
            "# codebase-indexer artifact\ngraph.db.zst binary merge=ours\n",
            encoding="utf-8",
        )


def _guess_project(db_path: str) -> str:
    """Return the first project name found in the database, or ''."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute("SELECT name FROM projects LIMIT 1").fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception:
        return ""


def _ensure_dir(path: str) -> None:
    """
    Create a directory and all parents, silently if it already exists.

    Args:
        path: absolute directory path to create

    Raises:
        OSError: if the directory cannot be created (e.g. permissions)
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO-8601 string.

    Format: "2024-01-15T10:30:00+00:00"

    Returns:
        ISO-8601 UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _compression_ratio(uncompressed: int, compressed: int) -> float:
    """
    Compute the compression ratio as uncompressed / compressed.

    Returns 0.0 if compressed is zero (avoids division by zero).

    Args:
        uncompressed: size before compression in bytes
        compressed:   size after compression in bytes

    Returns:
        Float ratio, e.g. 10.3 meaning the compressed file is 10.3x
        smaller than the original.

    Examples:
        >>> _compression_ratio(10_000_000, 1_000_000)
        10.0
        >>> _compression_ratio(0, 0)
        0.0
    """
    if compressed == 0:
        return 0.0
    return round(uncompressed / compressed, 2)
