"""End-to-end semantic CALLS extraction/resolution using real source files."""

from pathlib import Path

from src.indexer.pipeline import PipelineConfig, run
from src.indexer.store import SearchParams, open_path_readonly


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_real_source_calls_are_discovered_resolved_and_queryable(tmp_path):
    repo = tmp_path / "repo"
    cache_dir = tmp_path / "cache"

    _write(
        repo / "src/payments/service.py",
        """
def charge(amount):
    return amount
""".strip()
        + "\n",
    )

    _write(
        repo / "src/orders/views.py",
        """
from src.payments import service as pay_service


def checkout(amount):
    pay_service.charge(amount)
    missing(amount)


def with_local(amount):
    from src.payments.service import charge as local_charge
    return local_charge(amount)


def without_local(amount):
    return local_charge(amount)
""".strip()
        + "\n",
    )

    result = run(
        str(repo),
        PipelineConfig(
            project="semantic-e2e",
            cache_dir=str(cache_dir),
            export_artifact=False,
            incremental=False,
            max_workers=1,
        ),
    )

    assert result.calls_discovered > 0
    assert result.calls_resolved >= 1
    assert result.calls_unresolved >= 1
    assert result.edges_by_type.get("CALLS", 0) >= 1

    store = open_path_readonly(result.db_path)
    try:
        checkout_hit = store.search_nodes(
            SearchParams(
                project="semantic-e2e",
                name_pattern="%checkout%",
                limit=1,
            )
        )
        charge_hit = store.search_nodes(
            SearchParams(
                project="semantic-e2e",
                name_pattern="%charge%",
                limit=1,
            )
        )
        store.search_nodes(
            SearchParams(
                project="semantic-e2e",
                name_pattern="%with_local%",
                limit=1,
            )
        )
        without_local_hit = store.search_nodes(
            SearchParams(
                project="semantic-e2e",
                name_pattern="%without_local%",
                limit=1,
            )
        )

        assert checkout_hit.rows
        assert charge_hit.rows
        assert without_local_hit.rows

        checkout_qn = checkout_hit.rows[0].qualified_name
        charge_qn = charge_hit.rows[0].qualified_name
        without_local_qn = without_local_hit.rows[0].qualified_name

        callees = store.bfs_callees(
            checkout_qn,
            project="semantic-e2e",
            max_depth=1,
        )
        assert callees is not None
        callee_qns = {node.qualified_name for node, _ in callees.visited}
        assert charge_qn in callee_qns

        callers = store.bfs_callers(
            charge_qn,
            project="semantic-e2e",
            max_depth=1,
        )
        assert callers is not None
        caller_qns = {node.qualified_name for node, _ in callers.visited}
        assert checkout_qn in caller_qns
        assert without_local_qn not in caller_qns
    finally:
        store.close()
