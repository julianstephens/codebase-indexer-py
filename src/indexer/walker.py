"""
walker.py — Repository file discovery with gitignore support.

Walks a repository directory tree and returns a list of FileInfo objects
— one per file that should be indexed. Applies ignore rules at every
directory level so that large excluded subtrees (node_modules, .git,
build outputs) are never descended into.

Ignore rule layering (applied in order, first match wins):
    1. Hardcoded always-skip directories (_ALWAYS_SKIP_DIRS): .git,
       node_modules, __pycache__, .venv, build, dist, etc.
    2. Hardcoded always-skip file extensions (_ALWAYS_SKIP_EXTENSIONS):
       .pyc, .class, .o, .so, compiled binaries, media files, etc.
       These duplicate fallback._SKIP_EXTENSIONS for early rejection
       before file content is read.
    3. Root .gitignore — loaded once from the repo root if present.
    4. Root .cbmignore — project-specific overrides, gitignore syntax,
       loaded from the repo root if present. Takes precedence over
       .gitignore for conflicting patterns.
    5. Per-directory .gitignore files — loaded on descent, scoped to
       their directory subtree.

Symlinks are always skipped — following symlinks can cause infinite
loops and is not needed for code analysis.

The walker does not read file content. Content is read later by
extractor.extract_files() via its read_file callable. This keeps
discovery fast and separates I/O concerns.

Public API:
    walk(repo_path, config)  →  list[FileInfo]
        Main entry point. Returns all indexable files under repo_path.

    load_ignore_file(path)   →  pathspec.PathSpec | None
        Load a single .gitignore or .cbmignore file into a PathSpec.
        Returns None if the file does not exist.

    is_ignored(rel_path, specs)  →  bool
        Check a repo-relative path against a list of PathSpec objects.

Supporting types:
    WalkConfig   — configuration dataclass controlling walk behaviour
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pathspec

from .extractor import FileInfo
from .languages import EXTENSION_TO_LANG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Always-skip sets
# ---------------------------------------------------------------------------
#
# Applied before any ignore file is consulted. Entries here are never
# indexed regardless of .gitignore or .cbmignore contents.
#
# Directory names are matched against the directory's basename only
# (not its full path), so "build" matches any directory named "build"
# at any depth.

_ALWAYS_SKIP_DIRS: frozenset[str] = frozenset(
    {
        # Version control
        ".git",
        ".svn",
        ".hg",
        # Dependency trees
        "node_modules",
        "vendor",  # Go, PHP
        "bower_components",
        # Python
        "__pycache__",
        ".venv",
        "venv",
        ".env",  # directory (not the file)
        "env",
        "site-packages",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        # Build outputs
        "build",
        "dist",
        "out",
        "target",  # Rust, Java/Maven
        "bin",
        "obj",  # .NET
        ".next",  # Next.js
        ".nuxt",  # Nuxt.js
        ".output",
        ".turbo",
        "coverage",
        ".coverage",
        "htmlcov",
        # IDE / editor
        ".idea",
        ".vscode",
        ".vs",
        # Misc
        ".cache",
        "tmp",
        "temp",
        ".tmp",
        ".terraform",
        ".serverless",
    }
)

_ALWAYS_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Compiled Python
        ".pyc",
        ".pyo",
        ".pyd",
        # Compiled JVM
        ".class",
        ".jar",
        ".war",
        ".ear",
        # Native
        ".o",
        ".a",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".lib",
        ".out",
        # WebAssembly
        ".wasm",
        # Media
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".webm",
        ".pdf",
        # Fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".zst",
        ".7z",
        ".rar",
        # Database
        ".db",
        ".sqlite",
        ".sqlite3",
        # Data / ML
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".parquet",
        ".arrow",
        # Source maps
        ".map",
    }
)

# Filenames (basename only) that are always skipped regardless of extension.
_ALWAYS_SKIP_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "composer.lock",
        "cargo.lock",
        "gemfile.lock",
        "poetry.lock",
        "pipfile.lock",
        "flake.lock",
        ".ds_store",
        "thumbs.db",
    }
)


# ---------------------------------------------------------------------------
# WalkConfig
# ---------------------------------------------------------------------------


@dataclass
class WalkConfig:
    """
    Configuration for the walk() function.

    Attributes:
        max_file_bytes:
            Files larger than this are recorded in FileInfo with a
            size_bytes value but are not passed to the extractor.
            Defaults to 2 MB (matches fallback.MAX_FILE_BYTES).

        max_files:
            Hard cap on the total number of FileInfo objects returned.
            0 means no limit. Used to guard against accidentally
            indexing an entire filesystem.

        include_unknown_extensions:
            If True, files with extensions not in EXTENSION_TO_LANG are
            included in the result (with language=None) so fallback.py
            can produce File nodes for them.
            If False, only files with recognised extensions are returned.
            Defaults to True.

        extra_ignore_patterns:
            Additional gitignore-style patterns applied globally, as if
            they were in the root .gitignore. Useful for programmatic
            exclusions (e.g. ["*.generated.ts", "migrations/"]).

        follow_gitignore:
            If True (default), load and apply .gitignore files.
            Set to False in tests to avoid interference from the test
            repo's own .gitignore.

        cbmignore_filename:
            Name of the project-specific ignore file.
            Defaults to ".cbmignore". Override in tests.
    """

    max_file_bytes: int = 2 * 1024 * 1024  # 2 MB
    max_files: int = 0  # 0 = unlimited
    include_unknown_extensions: bool = True
    extra_ignore_patterns: list[str] = field(default_factory=list)
    follow_gitignore: bool = True
    cbmignore_filename: str = ".cbmignore"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def walk(
    repo_path: str,
    config: WalkConfig | None = None,
) -> list[FileInfo]:
    """
    Walk a repository and return a list of indexable FileInfo objects.

    Descends the directory tree rooted at repo_path, applying ignore
    rules at each level. Symlinks are never followed. Hidden directories
    other than .github are skipped unless they appear in the tree due to
    an explicit include pattern — in practice, the always-skip set
    handles the important cases.

    The returned list is sorted by FileInfo.path (repo-relative,
    forward-slash separated) for deterministic output regardless of
    filesystem ordering.

    Args:
        repo_path: absolute or relative path to the repository root.
                   Must be an existing directory.
        config:    WalkConfig controlling limits and ignore behaviour.
                   If None, uses WalkConfig() defaults.

    Returns:
        Sorted list of FileInfo objects. Each path is repo-relative and
        uses forward slashes on all platforms.

    Raises:
        NotADirectoryError: if repo_path does not exist or is not a dir.

    Examples:
        >>> files = walk("/path/to/my-repo")
        >>> files[0].path
        'src/auth/models.py'
        >>> files[0].language
        'python'
        >>> files[0].size_bytes > 0
        True
    """
    rpath = Path(repo_path).resolve()
    if not rpath.is_dir():
        raise NotADirectoryError(str(repo_path))
    if config is None:
        config = WalkConfig()
    root_specs = _build_root_specs(rpath, config)
    files: list[FileInfo] = []
    for file_info in _walk_iter(rpath, config, root_specs):
        files.append(file_info)
        if 0 < config.max_files <= len(files):
            logger.warning(
                "Reached max_files limit (%d) — stopping walk early", config.max_files
            )
            break
    files.sort(key=lambda fi: fi.path)
    return files


def load_ignore_file(path: str) -> pathspec.PathSpec | None:
    """
    Load a .gitignore or .cbmignore file and return a PathSpec.

    Parses the file using gitignore syntax (pathspec's "gitwildmatch"
    factory). Blank lines and comment lines (starting with #) are
    handled by pathspec automatically.

    Args:
        path: absolute path to the ignore file

    Returns:
        A pathspec.PathSpec instance if the file exists and is readable.
        None if the file does not exist (not an error — most directories
        don't have a .gitignore).

    Raises:
        Nothing — IOError and UnicodeDecodeError are caught and logged
        as warnings, returning None.

    Examples:
        >>> spec = load_ignore_file("/repo/.gitignore")
        >>> spec is not None
        True
        >>> spec = load_ignore_file("/repo/nonexistent/.gitignore")
        >>> spec is None
        True
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except (IOError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to load ignore file {path}: {e}")
        return None


def is_ignored(
    rel_path: str,
    specs: list[pathspec.PathSpec],
) -> bool:
    """
    Return True if rel_path matches any of the provided PathSpec objects.

    Checks the path against each spec in order. Returns True on the
    first match (any spec can exclude a path). Directories should be
    passed with a trailing slash so directory-only patterns (e.g.
    "build/") match correctly.

    Args:
        rel_path: repo-relative path using forward slashes,
                  e.g. "src/payments/service.py" or "node_modules/"
        specs:    list of PathSpec objects to check against.
                  Empty list always returns False.

    Returns:
        True if any spec matches rel_path, False otherwise.

    Examples:
        >>> spec = pathspec.PathSpec.from_lines("gitwildmatch", ["*.pyc"])
        >>> is_ignored("src/utils.pyc", [spec])
        True
        >>> is_ignored("src/utils.py", [spec])
        False
        >>> is_ignored("src/utils.py", [])
        False
    """
    if len(specs) == 0:
        return False
    return any(spec.match_file(rel_path) for spec in specs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _walk_iter(
    repo_path: Path,
    config: WalkConfig,
    root_specs: list[pathspec.PathSpec],
) -> Iterator[FileInfo]:
    """
    Yield FileInfo objects by recursively descending repo_path.

    Uses os.scandir() for efficient directory listing. At each directory
    level:
      1. Loads any .gitignore and .cbmignore present in that directory
         (if config.follow_gitignore is True) and appends them to a
         local spec list scoped to the subtree.
      2. Separates entries into dirs and files.
      3. Prunes dirs against _ALWAYS_SKIP_DIRS, symlinks, and ignore specs.
      4. Yields FileInfo for each file that passes all filters.

    Directory-local ignore files are scoped: a .gitignore in src/ only
    applies to files under src/. This mirrors git's own behaviour.

    Args:
        repo_path:  absolute Path to the repository root
        config:     WalkConfig controlling limits and ignore behaviour
        root_specs: PathSpec objects loaded from the repo root (passed
                    down unchanged to every subdirectory)

    Yields:
        FileInfo objects for each accepted file.
    """
    with os.scandir(repo_path) as entries:
        dirs = []
        for entry in entries:
            rel_path = _rel_path(entry.path, repo_path)
            if entry.is_dir(follow_symlinks=False) and not _should_skip_dir(
                entry, rel_path, root_specs
            ):
                dirs.append(entry)
            elif entry.is_file(follow_symlinks=False) and not _should_skip_file(
                entry, rel_path, config, root_specs
            ):
                yield _make_file_info(entry, repo_path)
        for dir_entry in dirs:
            dir_specs = _load_dir_specs(
                Path(dir_entry.path), repo_path, config, root_specs
            )
            yield from _walk_iter(dir_entry.path, config, dir_specs)


def _make_file_info(
    entry: os.DirEntry,
    repo_path: Path,
) -> FileInfo:
    """
    Construct a FileInfo from an os.DirEntry.

    Computes the repo-relative path (forward-slash separated), detects
    the language, and reads size_bytes and mtime_ns from the entry's
    stat result.

    os.DirEntry.stat() is called once — its result is cached by the OS
    on most platforms, so this is cheap relative to a separate os.stat()
    call.

    Args:
        entry:     os.DirEntry for the file (not a directory or symlink)
        repo_path: absolute Path to the repository root, used to compute
                   the repo-relative path

    Returns:
        FileInfo with path, abs_path, language, size_bytes, mtime_ns set.

    Examples:
        Assuming entry points to /repo/src/payments/service.py:
        >>> fi = _make_file_info(entry, Path("/repo"))
        >>> fi.path
        'src/payments/service.py'
        >>> fi.language
        'python'
        >>> fi.size_bytes > 0
        True
    """
    return FileInfo(
        path=_rel_path(entry.path, repo_path),
        abs_path=entry.path,
        language=EXTENSION_TO_LANG.get(os.path.splitext(entry.name)[1].lower()),
        size_bytes=entry.stat().st_size,
        mtime_ns=entry.stat().st_mtime_ns,
    )


def _should_skip_file(
    entry: os.DirEntry,
    rel_path: str,
    config: WalkConfig,
    specs: list[pathspec.PathSpec],
) -> bool:
    """
    Return True if a file entry should be excluded from the walk result.

    Checks in order (cheapest first):
      1. Is it a symlink?                          → skip
      2. Is its extension in _ALWAYS_SKIP_EXTENSIONS?  → skip
      3. Is its basename in _ALWAYS_SKIP_FILENAMES?    → skip
      4. Does it exceed config.max_file_bytes?         → skip
      5. Is it matched by any PathSpec in specs?       → skip
      6. Is config.include_unknown_extensions False
         and language is None?                         → skip

    Args:
        entry:    os.DirEntry for the file
        rel_path: repo-relative path (forward slashes)
        config:   WalkConfig
        specs:    combined list of root + directory-local PathSpec objects

    Returns:
        True if the file should be excluded, False if it should be kept.
    """
    if entry.is_symlink():
        return True
    _, ext = os.path.splitext(entry.name)
    if ext in _ALWAYS_SKIP_EXTENSIONS:
        return True
    if entry.name in _ALWAYS_SKIP_FILENAMES:
        return True
    if config.max_file_bytes > 0 and entry.stat().st_size > config.max_file_bytes:
        return True
    if rel_path in specs:
        return True
    if not config.include_unknown_extensions:
        language = EXTENSION_TO_LANG.get(ext.lower())
        if language is None:
            return True
    return False


def _should_skip_dir(
    entry: os.DirEntry,
    rel_path: str,
    specs: list[pathspec.PathSpec],
) -> bool:
    """
    Return True if a directory entry should be pruned (not descended).

    Checks in order:
      1. Is it a symlink?                          → skip
      2. Is its name in _ALWAYS_SKIP_DIRS?         → skip
      3. Is rel_path/ matched by any PathSpec?     → skip
         (Trailing slash added so directory-only patterns match.)

    Args:
        entry:    os.DirEntry for the directory
        rel_path: repo-relative path of the directory (forward slashes,
                  no trailing slash)
        specs:    combined list of PathSpec objects

    Returns:
        True if the directory should be pruned.
    """
    if entry.is_symlink():
        return True
    if entry.name in _ALWAYS_SKIP_DIRS:
        return True
    return rel_path + "/" in specs


def _rel_path(entry_path: str, repo_path: Path) -> str:
    """
    Compute the repo-relative path of a file, using forward slashes.

    Args:
        entry_path: absolute path string from os.DirEntry.path
        repo_path:  absolute Path to the repository root

    Returns:
        Forward-slash separated relative path string,
        e.g. "src/payments/service.py".
        On Windows, backslashes are converted to forward slashes.

    Examples:
        >>> _rel_path("/repo/src/payments/service.py", Path("/repo"))
        'src/payments/service.py'
    """
    entry_path = _to_forward_slashes(entry_path)
    return Path(entry_path).relative_to(repo_path).as_posix()


def _load_dir_specs(
    dir_path: Path,
    repo_path: Path,  # noqa: ARG001
    config: WalkConfig,
    parent_specs: list[pathspec.PathSpec],
) -> list[pathspec.PathSpec]:
    """
    Load ignore specs for a directory and return the combined spec list.

    Loads .gitignore and config.cbmignore_filename from dir_path (if
    config.follow_gitignore is True) and appends them to parent_specs.

    Returns a new list — does not mutate parent_specs — so each
    directory level gets its own scoped spec list.

    Args:
        dir_path:     absolute Path to the directory being entered
        repo_path:    absolute Path to the repo root (unused currently,
                      reserved for future relative-pattern scoping)
        config:       WalkConfig
        parent_specs: spec list inherited from the parent directory

    Returns:
        New list of PathSpec objects: parent_specs + any newly loaded
        specs for this directory. Returns parent_specs unchanged (as a
        copy) if no ignore files are found.
    """
    specs = list(parent_specs)
    if not config.follow_gitignore:
        return specs
    for filename in (".gitignore", config.cbmignore_filename):
        spec = load_ignore_file(str(dir_path / filename))
        if spec:
            specs.append(spec)
    return specs


def _build_root_specs(
    repo_path: Path,
    config: WalkConfig,
) -> list[pathspec.PathSpec]:
    """
    Build the initial PathSpec list from root-level ignore files and
    config.extra_ignore_patterns.

    Loads (in order):
      1. <repo_path>/.gitignore           (if config.follow_gitignore)
      2. <repo_path>/<cbmignore_filename> (if config.follow_gitignore)
      3. config.extra_ignore_patterns     (always)

    Args:
        repo_path: absolute Path to the repository root
        config:    WalkConfig

    Returns:
        List of PathSpec objects. May be empty if no ignore files exist
        and no extra patterns are configured.
    """
    res = []
    if config.follow_gitignore:
        gitignore_path = repo_path / ".gitignore"
        spec = load_ignore_file(str(gitignore_path))
        if spec:
            res.append(spec)

        cbmignore_path = repo_path / config.cbmignore_filename
        spec = load_ignore_file(str(cbmignore_path))
        if spec:
            res.append(spec)
    spec = pathspec.PathSpec.from_lines("gitwildmatch", config.extra_ignore_patterns)
    res.append(spec)
    return res


def _to_forward_slashes(path: str) -> str:
    """
    Convert backslashes to forward slashes for cross-platform consistency.

    Repo-relative paths are always stored and compared with forward
    slashes, even on Windows where os.path uses backslashes.

    Args:
        path: any path string

    Returns:
        Path string with all backslashes replaced by forward slashes.

    Examples:
        >>> _to_forward_slashes("src\\payments\\service.py")
        'src/payments/service.py'
        >>> _to_forward_slashes("src/payments/service.py")
        'src/payments/service.py'
    """
    return path.replace("\\", "/")
