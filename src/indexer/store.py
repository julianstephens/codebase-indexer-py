"""
store.py — SQLite graph store for the repo knowledge graph.

Manages the lifecycle of the SQLite database: opening connections,
inserting nodes and edges in bulk, querying for the agent tools, and
closing cleanly. All SQL is in this module — nothing else touches the
database directly.

Write path (pipeline):
    1. open_memory() or open_path() to get a connection
    2. begin_bulk() + drop_indexes() before batch inserts
    3. insert_project(), insert_nodes(), insert_edges(), insert_files(),
       insert_file_hashes() inside a single transaction
    4. end_bulk() + create_indexes() after the transaction commits
    5. dump_to_file() to persist an in-memory db to disk
    6. close() when done

Read path (agent tools):
    1. open_path() or open_path_readonly() on the cached db file
    2. get_node_by_qn(), search_nodes(), bfs_callers(), bfs_callees(),
       get_file_source(), etc.
    3. close() when done

Thread safety:
    A single Store instance must not be used from multiple threads
    concurrently. The pipeline creates one Store per run. Agent tools
    create a short-lived Store per request.

Public types:
    Store           — the main database handle
    NodeRow         — a node as read back from the database
    EdgeRow         — an edge as read back from the database
    SearchParams    — parameters for search_nodes()
    SearchResult    — output of search_nodes()
    BFSResult       — output of bfs_callers() / bfs_callees()

Public constants:
    DEFAULT_CACHE_DIR   — ~/.cache/codebase-indexer/
    SCHEMA_VERSION      — integer, incremented on schema changes
"""

import contextlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import schema as _schema
from .errors import FileNotFoundError, InvalidNodeRecordError, StoreOperationError
from .treesitter import NodeRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR: str = str(Path.home() / ".cache" / "codebase-indexer")
SCHEMA_VERSION: int = _schema.SCHEMA_VERSION

# Batch size for bulk node/edge inserts. Larger batches are faster but
# use more memory. 500 is a safe default for typical function sizes.
_INSERT_BATCH_SIZE: int = 500


# ---------------------------------------------------------------------------
# Row types (read path)
# ---------------------------------------------------------------------------


@dataclass
class NodeRow:
    """
    A node as read back from the nodes table.

    Fields mirror the nodes table schema exactly. properties is
    deserialized from JSON to a dict.

    Attributes:
        id:             SQLite row ID
        project:        project name
        label:          Function | Class | Method | Interface | Type | File
        name:           short symbol name
        qualified_name: globally unique dotted address
        file_path:      repo-relative file path
        start_line:     1-based start line
        end_line:       1-based end line
        signature:      single-line signature string
        source:         full source text of this node
        properties:     dict of language-specific extras
    """

    id: int
    project: str
    label: str
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    source: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass
class EdgeRow:
    """
    An edge as read back from the edges table.

    Attributes:
        id:         SQLite row ID
        project:    project name
        source_id:  node ID of the edge source
        target_id:  node ID of the edge target
        type:       CALLS | IMPORTS | DEFINES | INHERITS | IMPLEMENTS | CONTAINS
        properties: dict with confidence, strategy, line, etc.
    """

    id: int
    project: str
    source_id: int
    target_id: int
    type: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass
class SearchParams:
    """
    Parameters for search_nodes().

    All fields are optional — omitting a field means "no filter on this
    dimension". Combining multiple fields narrows the result set.

    Attributes:
        project:        filter by project name. None = all projects.
        label:          filter by node label, e.g. "Function".
        name_pattern:   SQL LIKE pattern on the name column,
                        e.g. "%charge%". None = no filter.
        file_pattern:   SQL LIKE pattern on file_path,
                        e.g. "src/payments/%". None = no filter.
        fts_query:      FTS5 query string for full-text search across
                        name, signature, source. None = no FTS filter.
                        Example: "sql injection" or "charge AND stripe".
        limit:          max rows to return. Defaults to 20.
        offset:         pagination offset. Defaults to 0.
    """

    project: str | None = None
    label: str | None = None
    name_pattern: str | None = None
    file_pattern: str | None = None
    fts_query: str | None = None
    limit: int = 20
    offset: int = 0


@dataclass
class SearchResult:
    """
    Output of search_nodes().

    Attributes:
        rows:   list of NodeRow objects matching the query
        total:  total matching rows before pagination (for UI display)
    """

    rows: list[NodeRow]
    total: int


@dataclass
class BFSResult:
    """
    Output of bfs_callers() or bfs_callees().

    Attributes:
        root:       the NodeRow that was the BFS starting point
        visited:    list of (NodeRow, hop_depth) tuples in BFS order.
                    hop_depth is 1 for direct callers/callees, 2 for
                    indirect, etc. The root node is not included.
        edges:      list of EdgeRow objects traversed during BFS.
                    Used by trace_callers() to show confidence scores.
    """

    root: NodeRow
    visited: list[tuple[NodeRow, int]]
    edges: list[EdgeRow]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class Store:
    """
    SQLite graph store handle.

    Wraps a sqlite3.Connection. All database operations go through
    methods on this class — no SQL outside store.py.

    Do not instantiate directly. Use open_memory(), open_path(), or
    open_path_readonly().
    """

    def __init__(self, conn: sqlite3.Connection, db_path: str = ":memory:"):
        """
        Initialise a Store wrapping an open sqlite3.Connection.

        Called by the open_* factory functions after the connection is
        configured and the schema is initialised.

        Args:
            conn:    open sqlite3.Connection with WAL mode and FK support
            db_path: path string for logging and error messages;
                     ":memory:" for in-memory databases
        """
        self._conn: sqlite3.Connection = conn
        self._db_path: str = db_path
        self._in_bulk: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Close the underlying SQLite connection.

        Safe to call multiple times — subsequent calls are no-ops.
        After close(), all other methods raise sqlite3.ProgrammingError.

        Examples:
            >>> store = open_memory()
            >>> store.close()
            >>> store.close()   # no-op, no error
        """
        with contextlib.suppress(Exception):
            self._conn.close()

    def checkpoint(self) -> None:
        """
        Force a WAL checkpoint and run PRAGMA optimize.

        Should be called after end_bulk() and before dump_to_file() to
        ensure all WAL frames are folded back into the main database
        file and SQLite's query planner statistics are up to date.

        No-op on in-memory databases.
        """
        if self._db_path == ":memory:":
            return
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.execute("PRAGMA optimize")
        except Exception:
            logger.warning("checkpoint failed on %s", self._db_path, exc_info=True)

    def check_integrity(self) -> bool:
        """
        Run PRAGMA integrity_check and verify the schema version.

        Returns:
            True if the database passes all checks.
            False if corruption is detected or schema_ver in any project
            row exceeds SCHEMA_VERSION (forward-incompatible artifact).

        Examples:
            >>> store = open_memory()
            >>> store.check_integrity()
            True
        """
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        if row[0] != "ok":
            logger.error("integrity_check failed: %s", row[0])
            return False
        rows = self._conn.execute("SELECT schema_ver FROM projects").fetchall()
        for r in rows:
            if r[0] > SCHEMA_VERSION:
                logger.error(
                    "forward-incompatible schema_ver %d (max supported: %d)",
                    r[0],
                    SCHEMA_VERSION,
                )
                return False
        return True

    # ── Bulk write optimisation ────────────────────────────────────────────

    def begin_bulk(self) -> None:
        """
        Set SQLite pragmas for maximum bulk-insert throughput.

        Sets synchronous=OFF and cache_size=-65536 (64 MB). WAL mode is
        preserved throughout. Must be paired with end_bulk().

        Call drop_indexes() after begin_bulk() and before inserting
        data, then create_indexes() before end_bulk().

        Raises:
            StoreOperationError: if begin_bulk() is called while already in
            bulk mode (missing end_bulk() call).
        """
        if self._in_bulk:
            raise StoreOperationError(
                op="begin_bulk", message="called while already in bulk mode"
            )
        self._conn.execute("PRAGMA synchronous = OFF")
        self._conn.execute("PRAGMA cache_size = -65536")
        self._in_bulk = True

    def end_bulk(self) -> None:
        """
        Restore normal SQLite pragmas after bulk insert.

        Resets synchronous=NORMAL and cache_size=-2000. Runs a WAL
        checkpoint. Must be called after create_indexes().

        Raises:
            StoreOperationError: if end_bulk() is called without a preceding
            begin_bulk().
        """
        if not self._in_bulk:
            raise StoreOperationError(
                op="end_bulk", message="called without a preceding begin_bulk()"
            )
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA cache_size = -2000")
        self._in_bulk = False
        self.checkpoint()

    def drop_indexes(self) -> None:
        """
        Drop all non-unique indexes before bulk insert.

        The UNIQUE constraints on nodes(project, qualified_name) and
        edges(source_id, target_id, type) are part of the table DDL and
        are NOT dropped — they continue to enforce integrity during bulk
        inserts.

        Must be called after begin_bulk() and before inserting data.
        """
        self._conn.executescript(_schema.DROP_INDEXES)

    def create_indexes(self) -> None:
        """
        Recreate all indexes after bulk insert.

        Must be called after all data has been committed and before
        end_bulk(). Runs PRAGMA optimize after index creation.
        """
        self._conn.executescript(_schema.INDEXES)
        self._conn.execute("PRAGMA optimize")

    # ── Transaction ────────────────────────────────────────────────────────

    def begin(self) -> None:
        """
        Begin an explicit transaction.

        SQLite's default autocommit is off when a transaction is open.
        Must be paired with commit() or rollback().

        Raises:
            sqlite3.OperationalError: if a transaction is already open.
        """
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        """
        Commit the current transaction.

        Raises:
            sqlite3.OperationalError: if no transaction is open.
        """
        self._conn.execute("COMMIT")

    def rollback(self) -> None:
        """
        Roll back the current transaction.

        Safe to call in an exception handler even if no transaction is
        open — sqlite3 ignores the rollback in that case.
        """
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("ROLLBACK")

    # ── Project CRUD ───────────────────────────────────────────────────────

    def upsert_project(
        self,
        name: str,
        root_path: str,
        language: str | None = None,
    ) -> None:
        """
        Insert or update a project record.

        Updates indexed_at to the current UTC time on conflict so the
        record reflects the most recent index run.

        Args:
            name:      project name, e.g. "my-app"
            root_path: absolute path to the repo root on disk
            language:  dominant language, e.g. "python". Optional.

        Examples:
            >>> store.upsert_project("my-app", "/home/user/my-app", "python")
        """
        self._conn.execute(
            """
            INSERT INTO projects (name, root_path, language, schema_ver)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                root_path  = excluded.root_path,
                language   = excluded.language,
                indexed_at = datetime('now'),
                schema_ver = excluded.schema_ver
            """,
            (name, root_path, language, SCHEMA_VERSION),
        )

    def get_project(self, name: str) -> dict | None:
        """
        Fetch a project record by name.

        Args:
            name: project name

        Returns:
            Dict with keys: name, root_path, language, indexed_at,
            schema_ver. None if not found.

        Examples:
            >>> store.upsert_project("p", "/repo")
            >>> store.get_project("p")["name"]
            'p'
            >>> store.get_project("nonexistent") is None
            True
        """
        row = self._conn.execute(
            "SELECT * FROM projects WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        """
        Return all project records ordered by name.

        Returns:
            List of dicts, each with keys: name, root_path, language,
            indexed_at, schema_ver. Empty list if no projects indexed.

        Examples:
            >>> store.upsert_project("a", "/a")
            >>> store.upsert_project("b", "/b")
            >>> [p["name"] for p in store.list_projects()]
            ['a', 'b']
        """
        rows = self._conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def delete_project(self, name: str) -> int:
        """
        Delete a project and all its associated data.

        Cascades to nodes, edges, files, file_hashes, and adr via
        ON DELETE CASCADE foreign keys.

        Args:
            name: project name to delete

        Returns:
            Number of rows deleted from the projects table (0 or 1).

        Examples:
            >>> store.upsert_project("p", "/repo")
            >>> store.delete_project("p")
            1
            >>> store.delete_project("nonexistent")
            0
        """
        cur = self._conn.execute("DELETE FROM projects WHERE name = ?", (name,))
        return cur.rowcount

    # ── Node writes ────────────────────────────────────────────────────────

    def insert_nodes(
        self,
        records: list[NodeRecord],
        project: str,
    ) -> dict[str, int]:
        """
        Bulk-insert NodeRecord objects into the nodes table.

        Uses INSERT ... ON CONFLICT DO UPDATE so node IDs remain stable
        when a symbol keeps the same (project, qualified_name) across
        re-index runs.

        Inserts in batches of _INSERT_BATCH_SIZE to avoid hitting
        SQLite's maximum variable count limit.

        Args:
            records: list of NodeRecord objects. qualified_name must be
                     set on every record (raises ValueError otherwise).
            project: project name. Stored on every inserted row.

        Returns:
            Dict mapping qualified_name → SQLite row ID for every
            inserted/replaced node. Used by insert_edges() to resolve
            QN references to IDs.

        Raises:
            ValueError: if any record has an empty qualified_name.

        Examples:
            >>> qn_to_id = store.insert_nodes(records, "my-app")
            >>> qn_to_id["my_app.src.payments.service.charge"]
            42
        """
        if any(not r.qualified_name for r in records):
            raise InvalidNodeRecordError(
                message="All NodeRecord objects must have a non-empty qualified_name"
            )
        qn_to_id: dict[str, int] = {}
        for batch in _batched(records, _INSERT_BATCH_SIZE):
            rows = [_node_record_to_row(r, project) for r in batch]
            self._conn.executemany(
                """
                INSERT INTO nodes
                    (project, label, name, qualified_name, file_path,
                     start_line, end_line, signature, source, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, qualified_name) DO UPDATE SET
                    label      = excluded.label,
                    name       = excluded.name,
                    file_path  = excluded.file_path,
                    start_line = excluded.start_line,
                    end_line   = excluded.end_line,
                    signature  = excluded.signature,
                    source     = excluded.source,
                    properties = excluded.properties
                """,
                rows,
            )
            qns = [r.qualified_name for r in batch]
            placeholders = ",".join("?" * len(qns))
            id_rows = self._conn.execute(
                f"SELECT id, qualified_name FROM nodes"
                f" WHERE project = ? AND qualified_name IN ({placeholders})",
                (project, *qns),
            ).fetchall()
            for id_row in id_rows:
                qn_to_id[id_row["qualified_name"]] = id_row["id"]
        return qn_to_id

    def delete_nodes_for_file(self, project: str, file_path: str) -> int:
        """
        Delete all nodes belonging to a specific file.

        Used by the incremental pipeline to remove stale nodes before
        re-extracting a changed file. Cascades to edges via ON DELETE
        CASCADE.

        Args:
            project:   project name
            file_path: repo-relative file path

        Returns:
            Number of nodes deleted.
        """
        cur = self._conn.execute(
            "DELETE FROM nodes WHERE project = ? AND file_path = ?",
            (project, file_path),
        )
        return cur.rowcount

    def delete_nodes_by_qns(self, project: str, qualified_names: list[str]) -> int:
        """
        Delete nodes by qualified name.

        Used by incremental indexing to remove symbols that disappeared
        from a changed file while preserving stable IDs for symbols that
        still exist.

        Args:
            project: project name
            qualified_names: node qualified names to delete

        Returns:
            Number of nodes deleted.
        """
        if not qualified_names:
            return 0
        total_deleted = 0
        for batch in _batched(qualified_names, _INSERT_BATCH_SIZE):
            placeholders = ",".join("?" * len(batch))
            cur = self._conn.execute(
                f"DELETE FROM nodes WHERE project = ?"
                f" AND qualified_name IN ({placeholders})",
                [project, *batch],
            )
            total_deleted += cur.rowcount
        return total_deleted

    def get_qns_for_file(self, project: str, file_path: str) -> set[str]:
        """
        Return all qualified names currently stored for a file.

        Args:
            project: project name
            file_path: repo-relative file path

        Returns:
            Set of qualified names for nodes in that file.
        """
        rows = self._conn.execute(
            "SELECT qualified_name FROM nodes WHERE project = ? AND file_path = ?",
            (project, file_path),
        ).fetchall()
        return {row["qualified_name"] for row in rows}

    # ── Edge writes ────────────────────────────────────────────────────────

    def insert_edges(
        self,
        edges: list[tuple[str, str, str, dict]],
        qn_to_id: dict[str, int],
        project: str,
    ) -> int:
        """
        Bulk-insert edges into the edges table.

        Each edge is a (source_qn, target_qn, type, properties) tuple.
        source_qn and target_qn are resolved to IDs via qn_to_id. Edges
        where either endpoint is not in qn_to_id are silently skipped
        (unresolved external calls are expected and normal).

        Uses INSERT OR IGNORE to skip duplicate edges (same source_id,
        target_id, type).

        Args:
            edges:     list of (source_qn, target_qn, edge_type, props)
            qn_to_id:  mapping from qualified_name → node ID, as
                       returned by insert_nodes()
            project:   project name stored on each edge row

        Returns:
            Number of edges actually inserted (excluding skipped
            duplicates and unresolved endpoints).

        Examples:
            >>> inserted = store.insert_edges(
            ...     [("a.foo", "a.bar", "CALLS", {"confidence": 0.95})],
            ...     qn_to_id,
            ...     "my-app",
            ... )
            >>> inserted
            1
        """
        valid: list[tuple] = []
        for src_qn, tgt_qn, edge_type, props in edges:
            src_id = qn_to_id.get(src_qn)
            tgt_id = qn_to_id.get(tgt_qn)
            if src_id is None or tgt_id is None:
                continue
            valid.append((project, src_id, tgt_id, edge_type, json.dumps(props)))
        inserted = 0
        for batch in _batched(valid, _INSERT_BATCH_SIZE):
            cur = self._conn.executemany(
                """
                INSERT OR IGNORE INTO edges
                    (project, source_id, target_id, type, properties)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch,
            )
            inserted += cur.rowcount
        return inserted

    # ── File writes ────────────────────────────────────────────────────────

    def insert_files(
        self,
        file_contents: dict[str, str],
        project: str,
        languages: dict[str, str | None] | None = None,
    ) -> None:
        """
        Insert or replace raw file content into the files table.

        Args:
            file_contents: dict mapping repo-relative path → source text
            project:       project name
            languages:     optional dict mapping path → language name.
                           If None or a path is missing, language is NULL.

        Examples:
            >>> store.insert_files(
            ...     {"src/payments/service.py": "def charge(): pass"},
            ...     "my-app",
            ...     {"src/payments/service.py": "python"},
            ... )
        """
        rows: list[tuple] = []
        for path, source in file_contents.items():
            lang = (languages or {}).get(path)
            line_count = source.count("\n") + (
                1 if source and not source.endswith("\n") else 0
            )
            size_bytes = len(source.encode("utf-8"))
            rows.append((project, path, lang, source, line_count, size_bytes))
        for batch in _batched(rows, _INSERT_BATCH_SIZE):
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO files
                    (project, path, language, source, line_count, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                batch,
            )

    def insert_file_hashes(
        self,
        hashes: list[tuple[str, str, int, int]],
        project: str,
    ) -> None:
        """
        Insert or replace file hash records for incremental re-indexing.

        Args:
            hashes:  list of (rel_path, sha256_hex, mtime_ns, size_bytes)
            project: project name

        Examples:
            >>> store.insert_file_hashes(
            ...     [("src/service.py", "abc123...", 1234567890, 512)],
            ...     "my-app",
            ... )
        """
        out: list[tuple] = [
            (project, path, sha, mtime_ns, size) for path, sha, mtime_ns, size in hashes
        ]
        for batch in _batched(out, _INSERT_BATCH_SIZE):
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO file_hashes
                    (project, path, sha256, mtime_ns, size_bytes)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch,
            )

    def delete_files(self, project: str, paths: list[str]) -> int:
        """
        Delete file content rows for the provided paths.

        Args:
            project: project name
            paths: repo-relative file paths

        Returns:
            Number of file rows deleted.
        """
        if not paths:
            return 0
        total_deleted = 0
        for batch in _batched(paths, _INSERT_BATCH_SIZE):
            placeholders = ",".join("?" * len(batch))
            cur = self._conn.execute(
                f"DELETE FROM files WHERE project = ? AND path IN ({placeholders})",
                [project, *batch],
            )
            total_deleted += cur.rowcount
        return total_deleted

    def delete_file_hashes(self, project: str, paths: list[str]) -> int:
        """
        Delete file hash rows for the provided paths.

        Args:
            project: project name
            paths: repo-relative file paths

        Returns:
            Number of hash rows deleted.
        """
        if not paths:
            return 0
        total_deleted = 0
        for batch in _batched(paths, _INSERT_BATCH_SIZE):
            placeholders = ",".join("?" * len(batch))
            cur = self._conn.execute(
                f"DELETE FROM file_hashes WHERE project = ?"
                f" AND path IN ({placeholders})",
                [project, *batch],
            )
            total_deleted += cur.rowcount
        return total_deleted

    def delete_edges_for_project(self, project: str) -> int:
        """
        Delete all edges for a project.

        Used by incremental indexing to rebuild all relationships from
        stored call/import properties after node updates.

        Args:
            project: project name

        Returns:
            Number of edges deleted.
        """
        cur = self._conn.execute("DELETE FROM edges WHERE project = ?", (project,))
        return cur.rowcount

    def get_qn_to_id(self, project: str) -> dict[str, int]:
        """
        Return a mapping from qualified name to node ID for a project.
        """
        rows = self._conn.execute(
            "SELECT id, qualified_name FROM nodes WHERE project = ?",
            (project,),
        ).fetchall()
        return {row["qualified_name"]: row["id"] for row in rows}

    def get_node_records(self, project: str) -> list[NodeRecord]:
        """
        Return all project nodes as NodeRecord objects.

        This is used by incremental indexing to rebuild edges from the
        node properties already persisted in the database.
        """
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE project = ? ORDER BY file_path, start_line",
            (project,),
        ).fetchall()
        records: list[NodeRecord] = []
        for row in rows:
            props = row["properties"]
            records.append(
                NodeRecord(
                    label=row["label"],
                    name=row["name"],
                    qualified_name=row["qualified_name"],
                    file_path=row["file_path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    signature=row["signature"],
                    source=row["source"],
                    language="",
                    properties=json.loads(props) if props else {},
                )
            )
        return records

    # ── Node reads ─────────────────────────────────────────────────────────

    def get_node_by_qn(
        self,
        qualified_name: str,
        project: str | None = None,
    ) -> NodeRow | None:
        """
        Fetch a single node by qualified name.

        Args:
            qualified_name: exact QN string, e.g.
                            "my_app.src.payments.service.charge"
            project:        optional project filter. If None, searches
                            all projects (QNs are globally unique).

        Returns:
            NodeRow if found, None otherwise.

        Examples:
            >>> node = store.get_node_by_qn("my_app.src.auth.views.login")
            >>> node.label
            'Function'
            >>> store.get_node_by_qn("nonexistent") is None
            True
        """
        if project:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? AND qualified_name = ?",
                (project, qualified_name),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE qualified_name = ?",
                (qualified_name,),
            ).fetchone()
        return _row_to_node(row) if row else None

    def get_node_by_id(self, node_id: int) -> NodeRow | None:
        """
        Fetch a single node by its SQLite row ID.

        Args:
            node_id: integer primary key

        Returns:
            NodeRow if found, None otherwise.
        """
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return _row_to_node(row) if row else None

    def get_nodes_by_file(
        self,
        project: str,
        file_path: str,
    ) -> list[NodeRow]:
        """
        Return all nodes belonging to a specific file, ordered by
        start_line ascending.

        Args:
            project:   project name
            file_path: repo-relative file path

        Returns:
            List of NodeRow objects. Empty list if no nodes found.
        """
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE project = ? AND file_path = ?"
            " ORDER BY start_line ASC",
            (project, file_path),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def count_nodes(self, project: str) -> int:
        """
        Return the total number of nodes for a project.

        Args:
            project: project name

        Returns:
            Integer count >= 0.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project = ?", (project,)
        ).fetchone()
        return row[0]

    def count_edges(self, project: str) -> int:
        """
        Return the total number of edges for a project.

        Args:
            project: project name

        Returns:
            Integer count >= 0.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE project = ?", (project,)
        ).fetchone()
        return row[0]

    # ── Search ─────────────────────────────────────────────────────────────

    def search_nodes(self, params: SearchParams) -> SearchResult:
        """
        Search nodes using a combination of filters and FTS5.

        When params.fts_query is set, performs an FTS5 match across
        name, signature, and source, then applies any additional column
        filters. When fts_query is None, falls back to a plain SQL
        query with LIKE patterns.

        Results are ordered by:
          - FTS5 rank (best match first) when fts_query is set
          - name ASC when fts_query is None

        Args:
            params: SearchParams dataclass controlling all filters

        Returns:
            SearchResult with rows (paginated) and total (unpaginated
            count for the same filters).

        Examples:
            >>> result = store.search_nodes(SearchParams(
            ...     project="my-app",
            ...     fts_query="sql injection",
            ... ))
            >>> result.total >= 0
            True
            >>> all(isinstance(r, NodeRow) for r in result.rows)
            True
        """
        conditions: list[str] = []
        args: list = []

        if params.fts_query:
            fts_cond = ["nodes_fts MATCH ?"]
            fts_args: list = [params.fts_query]
            if params.project:
                fts_cond.append("n.project = ?")
                fts_args.append(params.project)
            if params.label:
                fts_cond.append("n.label = ?")
                fts_args.append(params.label)
            if params.name_pattern:
                fts_cond.append("n.name LIKE ?")
                fts_args.append(params.name_pattern)
            if params.file_pattern:
                fts_cond.append("n.file_path LIKE ?")
                fts_args.append(params.file_pattern)
            where = " AND ".join(fts_cond)
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM nodes_fts"
                f" JOIN nodes n ON nodes_fts.rowid = n.id WHERE {where}",
                fts_args,
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT n.* FROM nodes_fts"
                f" JOIN nodes n ON nodes_fts.rowid = n.id"
                f" WHERE {where} ORDER BY rank LIMIT ? OFFSET ?",
                [*fts_args, params.limit, params.offset],
            ).fetchall()
        else:
            if params.project:
                conditions.append("project = ?")
                args.append(params.project)
            if params.label:
                conditions.append("label = ?")
                args.append(params.label)
            if params.name_pattern:
                conditions.append("name LIKE ?")
                args.append(params.name_pattern)
            if params.file_pattern:
                conditions.append("file_path LIKE ?")
                args.append(params.file_pattern)
            where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM nodes {where_clause}", args
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM nodes {where_clause}"
                f" ORDER BY name ASC LIMIT ? OFFSET ?",
                [*args, params.limit, params.offset],
            ).fetchall()
        return SearchResult(rows=[_row_to_node(r) for r in rows], total=total)

    # ── Graph traversal ────────────────────────────────────────────────────

    def bfs_callers(
        self,
        start_qn: str,
        project: str | None = None,
        max_depth: int = 3,
        max_nodes: int = 50,
        edge_types: list[str] | None = None,
    ) -> BFSResult | None:
        """
        Breadth-first search UP the call graph from a starting node.

        Finds all nodes that (directly or indirectly) call the node at
        start_qn. Useful for blast-radius analysis: "what will break if
        I change this function?"

        Args:
            start_qn:   qualified name of the starting node
            project:    optional project filter
            max_depth:  maximum BFS hops (1 = direct callers only).
                        Capped at 10 to prevent runaway traversal.
            max_nodes:  maximum total nodes to visit (including root).
                        Returns partial results if exceeded.
            edge_types: edge types to follow. Defaults to ["CALLS"].

        Returns:
            BFSResult with root, visited (caller, hop_depth) pairs, and
            traversed edges. None if start_qn is not found.

        Examples:
            >>> result = store.bfs_callers("my_app.src.payments.service.charge")
            >>> result.root.name
            'charge'
            >>> [(r.name, hop) for r, hop in result.visited[:3]]
            [('create_order', 1), ('checkout', 1), ('process_cart', 2)]
        """
        if edge_types is None:
            edge_types = ["CALLS"]
        max_depth = min(max_depth, 10)

        root = self.get_node_by_qn(start_qn, project)
        if root is None:
            return None

        visited_nodes: list[tuple[NodeRow, int]] = []
        visited_edges: list[EdgeRow] = []
        visited_ids: set[int] = {root.id}
        queue: list[tuple[int, int]] = [(root.id, 0)]
        et_ph = ",".join("?" * len(edge_types))
        proj_clause = " AND project = ?" if project else ""

        while queue and len(visited_ids) < max_nodes:
            node_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            proj_args = [project] if project else []
            edge_rows = self._conn.execute(
                f"SELECT * FROM edges WHERE target_id = ?"
                f" AND type IN ({et_ph}){proj_clause}",
                [node_id, *edge_types, *proj_args],
            ).fetchall()
            for edge_row in edge_rows:
                edge = _row_to_edge(edge_row)
                caller_id = edge.source_id
                if caller_id in visited_ids:
                    continue
                visited_ids.add(caller_id)
                caller = self.get_node_by_id(caller_id)
                if caller is None:
                    continue
                visited_nodes.append((caller, depth + 1))
                visited_edges.append(edge)
                if len(visited_ids) < max_nodes:
                    queue.append((caller_id, depth + 1))
        return BFSResult(root=root, visited=visited_nodes, edges=visited_edges)

    def bfs_callees(
        self,
        start_qn: str,
        project: str | None = None,
        max_depth: int = 3,
        max_nodes: int = 50,
        edge_types: list[str] | None = None,
    ) -> BFSResult | None:
        """
        Breadth-first search DOWN the call graph from a starting node.

        Finds all nodes that the node at start_qn (directly or
        indirectly) calls. Useful for dependency analysis: "what does
        this function depend on?"

        Args:
            start_qn:   qualified name of the starting node
            project:    optional project filter
            max_depth:  maximum BFS hops. Capped at 10.
            max_nodes:  maximum total nodes to visit.
            edge_types: edge types to follow. Defaults to ["CALLS"].

        Returns:
            BFSResult with root, visited (callee, hop_depth) pairs, and
            traversed edges. None if start_qn is not found.
        """
        if edge_types is None:
            edge_types = ["CALLS"]
        max_depth = min(max_depth, 10)

        root = self.get_node_by_qn(start_qn, project)
        if root is None:
            return None

        visited_nodes: list[tuple[NodeRow, int]] = []
        visited_edges: list[EdgeRow] = []
        visited_ids: set[int] = {root.id}
        queue: list[tuple[int, int]] = [(root.id, 0)]
        et_ph = ",".join("?" * len(edge_types))
        proj_clause = " AND project = ?" if project else ""

        while queue and len(visited_ids) < max_nodes:
            node_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            proj_args = [project] if project else []
            edge_rows = self._conn.execute(
                f"SELECT * FROM edges WHERE source_id = ?"
                f" AND type IN ({et_ph}){proj_clause}",
                [node_id, *edge_types, *proj_args],
            ).fetchall()
            for edge_row in edge_rows:
                edge = _row_to_edge(edge_row)
                callee_id = edge.target_id
                if callee_id in visited_ids:
                    continue
                visited_ids.add(callee_id)
                callee = self.get_node_by_id(callee_id)
                if callee is None:
                    continue
                visited_nodes.append((callee, depth + 1))
                visited_edges.append(edge)
                if len(visited_ids) < max_nodes:
                    queue.append((callee_id, depth + 1))
        return BFSResult(root=root, visited=visited_nodes, edges=visited_edges)

    # ── File reads ─────────────────────────────────────────────────────────

    def get_file_source(
        self,
        project: str,
        file_path: str,
    ) -> str | None:
        """
        Return the raw source content of a file from the files table.

        Args:
            project:   project name
            file_path: repo-relative file path

        Returns:
            Source text string, or None if the file is not in the index.
        """
        row = self._conn.execute(
            "SELECT source FROM files WHERE project = ? AND path = ?",
            (project, file_path),
        ).fetchone()
        return row[0] if row else None

    def get_file_hashes(
        self,
        project: str,
    ) -> dict[str, tuple[str, int, int]]:
        """
        Return all file hashes for a project.

        Used by the incremental pipeline to identify changed files.

        Args:
            project: project name

        Returns:
            Dict mapping rel_path → (sha256_hex, mtime_ns, size_bytes).
            Empty dict if no hashes are stored.
        """
        rows = self._conn.execute(
            "SELECT path, sha256, mtime_ns, size_bytes"
            " FROM file_hashes WHERE project = ?",
            (project,),
        ).fetchall()
        return {r["path"]: (r["sha256"], r["mtime_ns"], r["size_bytes"]) for r in rows}

    # ── Schema introspection ───────────────────────────────────────────────

    def get_schema_summary(self, project: str) -> dict:
        """
        Return a summary of the graph schema for a project.

        Queries the nodes and edges tables to produce label/type counts
        and sample qualified names. Used by the agent's get_graph_schema
        tool to orient itself before making targeted queries.

        Args:
            project: project name

        Returns:
            Dict with keys:
              node_labels:   list of {"label": str, "count": int}
              edge_types:    list of {"type": str, "count": int}
              sample_qns:    list of up to 10 sample qualified names
              total_nodes:   int
              total_edges:   int

        Examples:
            >>> summary = store.get_schema_summary("my-app")
            >>> summary["total_nodes"] > 0
            True
            >>> any(l["label"] == "Function"
            ...     for l in summary["node_labels"])
            True
        """
        node_labels = [
            {"label": r["label"], "count": r["count"]}
            for r in self._conn.execute(
                "SELECT label, COUNT(*) AS count FROM nodes"
                " WHERE project = ? GROUP BY label ORDER BY count DESC",
                (project,),
            ).fetchall()
        ]
        edge_types = [
            {"type": r["type"], "count": r["count"]}
            for r in self._conn.execute(
                "SELECT type, COUNT(*) AS count FROM edges"
                " WHERE project = ? GROUP BY type ORDER BY count DESC",
                (project,),
            ).fetchall()
        ]
        sample_qns = [
            r[0]
            for r in self._conn.execute(
                "SELECT qualified_name FROM nodes WHERE project = ? LIMIT 10",
                (project,),
            ).fetchall()
        ]
        return {
            "node_labels": node_labels,
            "edge_types": edge_types,
            "sample_qns": sample_qns,
            "total_nodes": self.count_nodes(project),
            "total_edges": self.count_edges(project),
        }

    # ── Skeleton ───────────────────────────────────────────────────────────

    def iter_skeleton(
        self,
        project: str,
        labels: list[str] | None = None,
    ) -> Iterator[tuple[str, str, str]]:
        """
        Yield (file_path, signature, qualified_name) rows for skeleton
        rendering, ordered by file_path ASC, start_line ASC.

        Designed for streaming — does not load all rows into memory at
        once. The skeleton renderer in context.py consumes this iterator.

        Args:
            project: project name
            labels:  optional list of labels to include.
                     Defaults to all labels except "File" (File nodes
                     are listed separately at the top of the skeleton).

        Yields:
            (file_path, signature, qualified_name) tuples.

        Examples:
            >>> for fp, sig, qn in store.iter_skeleton("my-app"):
            ...     print(f"{fp}: {sig}")
        """
        if labels is None:
            label_clause = "AND label != 'File'"
            label_args: list = []
        else:
            label_clause = f"AND label IN ({','.join('?' * len(labels))})"
            label_args = list(labels)
        cursor = self._conn.execute(
            f"SELECT file_path, signature, qualified_name FROM nodes"
            f" WHERE project = ? {label_clause}"
            f" ORDER BY file_path ASC, start_line ASC",
            [project, *label_args],
        )
        yield from ((row[0], row[1], row[2]) for row in cursor)

    # ── Dump / restore ─────────────────────────────────────────────────────

    def dump_to_file(self, dest_path: str) -> None:
        """
        Dump the database to a file using SQLite's backup API.

        Works for both in-memory and file-backed databases. Creates
        parent directories if they don't exist.

        For in-memory databases this is the primary persistence
        mechanism — the pipeline indexes into memory then calls
        dump_to_file() at the end.

        Args:
            dest_path: absolute path to write the .db file.

        Raises:
            OSError: if the destination directory cannot be created.
            sqlite3.Error: on backup failure.

        Examples:
            >>> store = open_memory()
            >>> store.dump_to_file("/tmp/my-app.db")
        """
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        dest_conn = sqlite3.connect(dest_path)
        try:
            self._conn.backup(dest_conn)
        finally:
            dest_conn.close()

    def restore_from_file(self, src_path: str) -> None:
        """
        Restore the database from a file using SQLite's backup API.

        Replaces the current database contents with those from src_path.
        Used by artifact.import_artifact() to load a decompressed
        snapshot into the working cache database.

        Args:
            src_path: absolute path to an existing .db file.

        Raises:
            FileNotFoundError: if src_path does not exist.
            sqlite3.Error: on backup failure.
        """
        if not Path(src_path).exists():
            raise FileNotFoundError(src_path)
        src_conn = sqlite3.connect(src_path)
        try:
            src_conn.backup(self._conn)
        finally:
            src_conn.close()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def open_memory() -> Store:
    """
    Open a new in-memory SQLite database and initialise the schema.

    Used by the pipeline during indexing — all writes happen in memory
    for speed, then dump_to_file() persists the result.

    Returns:
        A fully initialised Store backed by an in-memory database.

    Examples:
        >>> store = open_memory()
        >>> store.check_integrity()
        True
    """
    conn = sqlite3.connect(":memory:")
    _configure_connection(conn)
    _schema.initialize(conn)
    return Store(conn, ":memory:")


def open_path(db_path: str) -> Store:
    """
    Open a file-backed SQLite database, creating it if it doesn't exist.

    Initialises the schema if the file is new. If the file already
    exists, the schema is applied with IF NOT EXISTS so existing data
    is not disturbed.

    Args:
        db_path: absolute path to the .db file.

    Returns:
        A fully initialised Store backed by a file database.

    Raises:
        OSError: if the parent directory cannot be created.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _configure_connection(conn)
    _schema.initialize(conn)
    return Store(conn, db_path)


def open_path_readonly(db_path: str) -> Store:
    """
    Open an existing file-backed SQLite database in read-only mode.

    Used by agent tools (get_source, search, trace_callers) which only
    read data. Read-only mode prevents accidental writes and allows
    multiple processes to read the same database concurrently.

    Args:
        db_path: absolute path to an existing .db file.

    Returns:
        A Store in read-only mode.

    Raises:
        FileNotFoundError: if db_path does not exist.

    Examples:
        >>> store = open_path_readonly("/home/user/.cache/codebase-indexer/my-app.db")
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(db_path)
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    _configure_connection(conn, readonly=True)
    return Store(conn, db_path)


def default_db_path(project: str, cache_dir: str | None = None) -> str:
    """
    Return the default database file path for a project.

    Constructs the path as <cache_dir>/<project>.db, creating the cache
    directory if it does not exist.

    Args:
        project:   project name, e.g. "my-app"
        cache_dir: override for the cache directory. Defaults to
                   DEFAULT_CACHE_DIR (~/.cache/codebase-indexer/).

    Returns:
        Absolute path string, e.g.
        "/home/user/.cache/codebase-indexer/my-app.db"

    Examples:
        >>> default_db_path("my-app")
        '/home/user/.cache/codebase-indexer/my-app.db'
    """
    cache = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache / f"{project}.db")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _configure_connection(conn: sqlite3.Connection, readonly: bool = False) -> None:
    """
    Apply connection-level SQLite configuration.

    Sets:
      - row_factory = sqlite3.Row for dict-like row access
      - WAL journal mode for concurrent read access (skipped for readonly)
      - Foreign key enforcement
      - 64 MB mmap_size for faster reads on large databases

    Args:
        conn:     open sqlite3.Connection to configure
        readonly: if True, skip PRAGMAs that require write access
    """
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None  # autocommit; transactions are explicit
    if not readonly:
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA mmap_size = 67108864")


def _row_to_node(row: sqlite3.Row) -> NodeRow:
    """
    Convert a sqlite3.Row from the nodes table to a NodeRow dataclass.

    Deserialises the properties column from JSON. Handles NULL
    properties gracefully (returns empty dict).

    Args:
        row: a sqlite3.Row with all columns of the nodes table

    Returns:
        NodeRow dataclass instance.
    """
    props = row["properties"]
    return NodeRow(
        id=row["id"],
        project=row["project"],
        label=row["label"],
        name=row["name"],
        qualified_name=row["qualified_name"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        signature=row["signature"],
        source=row["source"],
        properties=json.loads(props) if props else {},
    )


def _row_to_edge(row: sqlite3.Row) -> EdgeRow:
    """
    Convert a sqlite3.Row from the edges table to an EdgeRow dataclass.

    Deserialises the properties column from JSON.

    Args:
        row: a sqlite3.Row with all columns of the edges table

    Returns:
        EdgeRow dataclass instance.
    """
    props = row["properties"]
    return EdgeRow(
        id=row["id"],
        project=row["project"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        type=row["type"],
        properties=json.loads(props) if props else {},
    )


def _node_record_to_row(
    record: NodeRecord,
    project: str,
) -> tuple:
    """
    Convert a NodeRecord to a tuple suitable for parameterised INSERT.

    Serialises the properties dict to JSON. Returns a tuple matching
    the column order of the nodes INSERT statement:
        (project, label, name, qualified_name, file_path,
         start_line, end_line, signature, source, properties_json)

    Args:
        record:  NodeRecord with all fields set (qualified_name must
                 be non-empty)
        project: project name

    Returns:
        10-element tuple for use as executemany() parameters.

    Raises:
        ValueError: if record.qualified_name is empty.
    """
    if not record.qualified_name:
        raise InvalidNodeRecordError(
            message=(
                f"NodeRecord has empty qualified_name: name={record.name!r}, "
                f"file={record.file_path!r}"
            ),
        )
    props = dict(record.properties)
    if record.parent:
        props["parent"] = record.parent
    return (
        project,
        record.label,
        record.name,
        record.qualified_name,
        record.file_path,
        record.start_line,
        record.end_line,
        record.signature,
        record.source,
        json.dumps(props),
    )


def _batched(items: list, size: int) -> Iterator[list]:
    """
    Yield successive slices of `items` of length `size`.

    Used to split large node/edge lists into batches for executemany()
    to avoid SQLite's maximum variable count (SQLITE_MAX_VARIABLE_NUMBER,
    default 999).

    Args:
        items: list to slice
        size:  maximum batch size (must be >= 1)

    Yields:
        Successive sub-lists, each of length <= size.
        The last sub-list may be shorter than size.

    Examples:
        >>> list(_batched([1,2,3,4,5], 2))
        [[1, 2], [3, 4], [5]]
        >>> list(_batched([], 10))
        []
    """
    for i in range(0, max(len(items), 1), size):
        batch = items[i : i + size]
        if batch:
            yield batch
