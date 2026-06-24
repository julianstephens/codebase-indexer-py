"""Tests for v2 call/import handling in pipeline pass 6."""

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
from src.indexer.store import open_memory
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
