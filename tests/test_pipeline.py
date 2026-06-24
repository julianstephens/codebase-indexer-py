"""Tests for v2 call/import handling in pipeline pass 6."""

from pathlib import Path

from src.indexer.extractor import ExtractionResult
from src.indexer.pipeline import (
    PipelineConfig,
    _collect_calls_from_records,
    _collect_imports_from_records,
    _pass_resolve_calls,
    _pass_store,
    run,
)
from src.indexer.registry import build
from src.indexer.store import SearchParams, open_memory, open_path_readonly
from src.indexer.treesitter import NodeRecord


def make_record(
    *,
    name: str,
    qn: str,
    file_path: str,
    label: str = "Function",
    properties: dict | None = None,
) -> NodeRecord:
    return NodeRecord(
        label=label,
        name=name,
        qualified_name=qn,
        file_path=file_path,
        start_line=1,
        end_line=3,
        signature=f"def {name}():",
        source=f"def {name}():\n    pass",
        language="python",
        properties=properties or {},
    )


def test_collect_imports_from_records_parses_supported_shapes():
    record = make_record(
        name="checkout",
        qn="src.orders.views.checkout",
        file_path="src/orders/views.py",
        properties={
            "imports": [
                "src.payments.service",
                {
                    "module_path": "src.auth.models",
                    "names": ["User", "Role"],
                    "alias": "models",
                    "line": 7,
                },
                {
                    "module": "src.orders.helpers",
                    "names": "normalize, clean",
                },
            ]
        },
    )

    imports, malformed = _collect_imports_from_records([record])

    assert len(imports) == 3
    assert malformed == 0
    assert imports[0].module_path == "src.payments.service"
    assert imports[1].module_path == "src.auth.models"
    assert imports[1].names == ["User", "Role"]
    assert imports[1].alias == "models"
    assert imports[1].line == 7
    assert imports[2].module_path == "src.orders.helpers"
    assert imports[2].names == ["normalize", "clean"]


def test_collect_imports_from_records_preserves_scope():
    record = make_record(
        name="checkout",
        qn="src.orders.views.checkout",
        file_path="src/orders/views.py",
        properties={
            "imports": [
                {
                    "module_path": "src.payments.service",
                    "names": ["charge"],
                    "scope": "file",
                },
                {
                    "module_path": "src.payments.service",
                    "names": ["charge"],
                    "scope": "local",
                },
            ]
        },
    )

    imports, malformed = _collect_imports_from_records([record])

    assert malformed == 0
    assert len(imports) == 2
    by_scope = {imp.in_function for imp in imports}
    assert "" in by_scope
    assert "src.orders.views.checkout" in by_scope


def test_collect_calls_from_records_defaults_in_function():
    record = make_record(
        name="checkout",
        qn="src.orders.views.checkout",
        file_path="src/orders/views.py",
        properties={
            "calls": [
                "charge",
                {
                    "callee": "svc.refund",
                    "line": 12,
                    "qualifier": "svc",
                },
            ]
        },
    )

    calls, malformed, unsupported = _collect_calls_from_records([record])

    assert len(calls) == 2
    assert malformed == 0
    assert unsupported == 0
    assert calls[0].callee == "charge"
    assert calls[0].in_function == "src.orders.views.checkout"
    assert calls[1].callee == "svc.refund"
    assert calls[1].line == 12
    assert calls[1].qualifier == "svc"
    assert calls[1].in_function == "src.orders.views.checkout"


def test_pass_resolve_calls_uses_v2_properties_and_tracks_stats():
    caller = make_record(
        name="checkout",
        qn="src.orders.views.checkout",
        file_path="src/orders/views.py",
        properties={
            "imports": [{"module_path": "src.payments.service", "names": ["charge"]}],
            "calls": [
                {"callee": "charge", "line": 10},
                {"callee": "missing", "line": 11},
            ],
        },
    )
    target = make_record(
        name="charge",
        qn="src.payments.service.charge",
        file_path="src/payments/service.py",
    )

    reg = build([caller, target])
    results = {
        "src/orders/views.py": ExtractionResult(
            path="src/orders/views.py",
            records=[caller],
            language="python",
            extractor="treesitter",
        )
    }

    (
        edges,
        calls_discovered,
        calls_resolved,
        calls_unresolved,
        calls_unsupported,
        malformed_payloads,
    ) = _pass_resolve_calls(
        results,
        reg,
        "my-app",
        PipelineConfig(max_workers=1),
    )

    assert len(edges) == 1
    assert calls_discovered == 2
    assert edges[0][0] == "src.orders.views.checkout"
    assert edges[0][1] == "src.payments.service.charge"
    assert edges[0][2] == "CALLS"
    assert edges[0][3]["line"] == 10
    assert calls_resolved == 1
    assert calls_unresolved == 1
    assert calls_unsupported == 0
    assert malformed_payloads == 0


def test_pass_store_inserts_only_valid_unique_edges():
    caller = make_record(
        name="checkout",
        qn="src.orders.views.checkout",
        file_path="src/orders/views.py",
    )
    target = make_record(
        name="charge",
        qn="src.payments.service.charge",
        file_path="src/payments/service.py",
    )

    # Includes one valid edge, one unresolved endpoint, and one duplicate.
    edges = [
        (
            "src.orders.views.checkout",
            "src.payments.service.charge",
            "CALLS",
            {"confidence": 0.95, "strategy": "same_module", "line": 10},
        ),
        (
            "src.orders.views.checkout",
            "src.external.missing",
            "CALLS",
            {"confidence": 0.85, "strategy": "import_map", "line": 11},
        ),
        (
            "src.orders.views.checkout",
            "src.payments.service.charge",
            "CALLS",
            {"confidence": 0.95, "strategy": "same_module", "line": 10},
        ),
    ]

    db = open_memory()
    try:
        _, edges_inserted = _pass_store(
            db=db,
            project="my-app",
            repo_path="/tmp/my-app",
            records=[caller, target],
            edges=edges,
            file_contents={},
            file_hashes=[],
            file_languages={},
        )
    finally:
        db.close()

    assert edges_inserted == 1


def test_pass_resolve_calls_counts_malformed_and_unsupported_payloads():
    caller = make_record(
        name="checkout",
        qn="src.orders.views.checkout",
        file_path="src/orders/views.py",
        properties={
            "imports": [
                {"module_path": "src.payments.service", "names": ["charge"]},
                42,
                {"line": 7},
            ],
            "calls": [
                {"callee": "charge", "line": 10},
                {"line": 11},
                "",
            ],
            "unsupported_calls": 2,
        },
    )
    target = make_record(
        name="charge",
        qn="src.payments.service.charge",
        file_path="src/payments/service.py",
    )

    reg = build([caller, target])
    results = {
        "src/orders/views.py": ExtractionResult(
            path="src/orders/views.py",
            records=[caller],
            language="python",
            extractor="treesitter",
        )
    }

    (
        edges,
        calls_discovered,
        calls_resolved,
        calls_unresolved,
        calls_unsupported,
        malformed_payloads,
    ) = _pass_resolve_calls(
        results,
        reg,
        "my-app",
        PipelineConfig(max_workers=1),
    )

    assert len(edges) == 1
    assert calls_discovered == 1
    assert calls_resolved == 1
    assert calls_unresolved == 0
    assert calls_unsupported == 2
    assert malformed_payloads == 4


def test_run_reports_relationship_unavailable_language(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)

    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )
    (repo / "README.md").write_text("hello\n", encoding="utf-8")

    result = run(
        str(repo),
        PipelineConfig(
            project="diag-run",
            cache_dir=str(tmp_path / "cache"),
            export_artifact=False,
            incremental=False,
            max_workers=1,
        ),
    )

    assert "unknown" in result.relationship_unavailable_languages


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_incremental_run_preserves_graph_state_and_rebuilds_edges(tmp_path):
    repo = tmp_path / "repo"
    cache_dir = tmp_path / "cache"

    _write(
        repo / "src" / "a.py",
        """
from src.b import target


def caller():
    return target()
""".strip()
        + "\n",
    )
    _write(
        repo / "src" / "b.py",
        """
def stable():
    return 1


def target():
    return stable()
""".strip()
        + "\n",
    )
    _write(
        repo / "src" / "removed.py",
        """
def orphan():
    return 0
""".strip()
        + "\n",
    )

    first = run(
        str(repo),
        PipelineConfig(
            project="incremental-state",
            cache_dir=str(cache_dir),
            export_artifact=False,
            incremental=False,
            max_workers=1,
        ),
    )

    store = open_path_readonly(first.db_path)
    try:
        caller = store.search_nodes(
            SearchParams(project="incremental-state", name_pattern="%caller%", limit=1)
        ).rows[0]
        stable = store.search_nodes(
            SearchParams(project="incremental-state", name_pattern="%stable%", limit=1)
        ).rows[0]
        target = store.search_nodes(
            SearchParams(project="incremental-state", name_pattern="%target%", limit=2)
        ).rows
        assert len(target) == 1

        caller_qn = caller.qualified_name
        caller_id_before = caller.id
        stable_qn = stable.qualified_name
        stable_id_before = stable.id
        target_qn = target[0].qualified_name

        callees_before = store.bfs_callees(
            caller_qn,
            project="incremental-state",
            max_depth=1,
        )
        assert callees_before is not None
        assert {node.qualified_name for node, _ in callees_before.visited} == {
            target_qn
        }
        removed_before = store.search_nodes(
            SearchParams(project="incremental-state", name_pattern="%orphan%", limit=1)
        )
        assert removed_before.rows
        caller_file_path = caller.file_path
        removed_file_path = removed_before.rows[0].file_path
    finally:
        store.close()

    _write(
        repo / "src" / "b.py",
        """
def stable():
    return 2


def new_target():
    return stable()
""".strip()
        + "\n",
    )
    (repo / "src" / "removed.py").unlink()

    second = run(
        str(repo),
        PipelineConfig(
            project="incremental-state",
            cache_dir=str(cache_dir),
            export_artifact=False,
            incremental=True,
            max_workers=1,
        ),
    )

    assert second.files_unchanged == 1
    assert second.files_changed == 1
    assert second.files_added == 0
    assert second.files_removed == 1

    store = open_path_readonly(second.db_path)
    try:
        caller_after = store.get_node_by_qn(caller_qn, project="incremental-state")
        stable_after = store.get_node_by_qn(stable_qn, project="incremental-state")
        assert caller_after is not None
        assert stable_after is not None

        # Unchanged file nodes stay present, and stable symbols in changed
        # files retain IDs via ON CONFLICT DO UPDATE.
        assert caller_after.id == caller_id_before
        assert stable_after.id == stable_id_before

        target_after = store.search_nodes(
            SearchParams(project="incremental-state", name_pattern="%target%", limit=10)
        ).rows
        assert all(row.name != "target" for row in target_after)
        assert any(row.name == "new_target" for row in target_after)

        removed_after = store.search_nodes(
            SearchParams(project="incremental-state", name_pattern="%orphan%", limit=1)
        )
        assert removed_after.rows == []

        hashes = store.get_file_hashes("incremental-state")
        assert caller_file_path in hashes
        assert removed_file_path not in hashes

        # Edges are rebuilt from stored payloads, so unchanged callers do
        # not keep stale links to removed symbols.
        callees_after = store.bfs_callees(
            caller_qn,
            project="incremental-state",
            max_depth=1,
        )
        assert callees_after is not None
        assert callees_after.visited == []
    finally:
        store.close()
