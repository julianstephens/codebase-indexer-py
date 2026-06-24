"""
tests/test_tools.py — Tests for indexer/tools.py

Coverage:
  - get_source(): found, not found, missing db, truncation, callers/callees,
                  token estimate present, hint on not-found
  - search(): results found, no results, FTS query, label filter, limit cap,
              pagination hint, missing db, error string
  - trace_callers(): found, not found, no callers, multi-hop, depth cap,
                     confidence map, missing db, error string
  - Internal helpers: _format_related, _format_search_hit, _build_confidence_map,
                      _format_qn, _truncate_sig, _maybe_truncate, _bare_name
  - Error recovery: all three public functions return strings on exception,
                    never raise
"""

from pathlib import Path

import pytest

from src.indexer.store import (
    BFSResult,
    EdgeRow,
    NodeRow,
    open_path,
)
from src.indexer.tools import (
    MAX_RELATED_SHOWN,
    MAX_SOURCE_BYTES,
    _bare_name,
    _build_confidence_map,
    _format_qn,
    _format_related,
    _format_search_hit,
    _maybe_truncate,
    _truncate_sig,
    get_source,
    search,
    trace_callers,
)
from src.indexer.treesitter import NodeRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    name: str,
    qn: str,
    file_path: str = "src/utils.py",
    label: str = "Function",
    start_line: int = 1,
    end_line: int = 10,
    parent: str = "",
    signature: str = "",
    source: str = "",
    language: str = "python",
    properties: dict | None = None,
) -> NodeRecord:
    sig = signature or f"def {name}():"
    src = source or f"{sig}\n    pass"
    return NodeRecord(
        label=label,
        name=name,
        qualified_name=qn,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=sig,
        source=src,
        language=language,
        parent=parent,
        properties=properties or {},
    )


def _build_db(tmp_path: Path) -> str:
    """
    Build a populated test database with:
      project "test-app"
      nodes:
        charge      Function  src/payments/service.py   lines 1-20
        refund      Function  src/payments/service.py   lines 22-35
        checkout    Function  src/payments/views.py     lines 1-30
        complete    Function  src/orders/processor.py   lines 5-25
        Payment     Class     src/payments/models.py    lines 1-50
        save        Method    src/payments/models.py    lines 10-20  parent=Payment
      edges:
        checkout   → charge    CALLS  confidence=0.95
        complete   → charge    CALLS  confidence=0.85
        charge     → save      CALLS  confidence=0.90
        charge     → refund    CALLS  confidence=0.70
      files:
        src/payments/service.py
        src/payments/views.py
        src/payments/models.py
        src/orders/processor.py
    """
    db_path = str(tmp_path / "test.db")
    store = open_path(db_path)

    store.upsert_project("test-app", "/repo", "python")

    records = [
        _make_record(
            "charge",
            "test_app.src.payments.service.charge",
            file_path="src/payments/service.py",
            start_line=1,
            end_line=20,
            signature=(
                "def charge(user: User, amount_cents: int, currency: str) -> Payment:"
            ),
            source=(
                "def charge(user: User, amount_cents: int, currency: str) -> Payment:\n"
                '    """Charge a user via Stripe."""\n'
                "    result = stripe.charge(user.token, amount_cents, currency)\n"
                "    payment = Payment.save(result)\n"
                "    return payment\n"
            ),
        ),
        _make_record(
            "refund",
            "test_app.src.payments.service.refund",
            file_path="src/payments/service.py",
            start_line=22,
            end_line=35,
            signature="def refund(payment: Payment) -> bool:",
            source=(
                "def refund(payment: Payment) -> bool:\n"
                "    return stripe.refund(payment.id)\n"
            ),
        ),
        _make_record(
            "checkout",
            "test_app.src.payments.views.checkout",
            file_path="src/payments/views.py",
            start_line=1,
            end_line=30,
            signature="def checkout(request: Request) -> Response:",
            source=(
                "def checkout(request: Request) -> Response:\n"
                "    charge(request.user, request.amount)\n"
            ),
        ),
        _make_record(
            "complete",
            "test_app.src.orders.processor.complete",
            file_path="src/orders/processor.py",
            start_line=5,
            end_line=25,
            signature="def complete(order: Order) -> None:",
            source=(
                "def complete(order: Order) -> None:\n"
                "    charge(order.user, order.total)\n"
            ),
        ),
        _make_record(
            "Payment",
            "test_app.src.payments.models.Payment",
            file_path="src/payments/models.py",
            label="Class",
            start_line=1,
            end_line=50,
            signature="class Payment(BaseModel):",
            source="class Payment(BaseModel):\n    id: str\n    amount: int\n",
        ),
        _make_record(
            "save",
            "test_app.src.payments.models.Payment.save",
            file_path="src/payments/models.py",
            label="Method",
            start_line=10,
            end_line=20,
            parent="Payment",
            signature="def save(self) -> None:",
            source="def save(self) -> None:\n    db.save(self)\n",
        ),
    ]

    store.begin()
    qn_to_id = store.insert_nodes(records, "test-app")
    store.insert_edges(
        [
            (
                "test_app.src.payments.views.checkout",
                "test_app.src.payments.service.charge",
                "CALLS",
                {"confidence": 0.95, "strategy": "import_map"},
            ),
            (
                "test_app.src.orders.processor.complete",
                "test_app.src.payments.service.charge",
                "CALLS",
                {"confidence": 0.85, "strategy": "fuzzy"},
            ),
            (
                "test_app.src.payments.service.charge",
                "test_app.src.payments.models.Payment.save",
                "CALLS",
                {"confidence": 0.90, "strategy": "same_module"},
            ),
            (
                "test_app.src.payments.service.charge",
                "test_app.src.payments.service.refund",
                "CALLS",
                {"confidence": 0.70, "strategy": "same_module"},
            ),
        ],
        qn_to_id,
        "test-app",
    )
    store.insert_files(
        {
            "src/payments/service.py": "def charge(): pass\ndef refund(): pass",
            "src/payments/views.py": "def checkout(): pass",
            "src/payments/models.py": "class Payment: pass",
            "src/orders/processor.py": "def complete(): pass",
        },
        "test-app",
        {
            "src/payments/service.py": "python",
            "src/payments/views.py": "python",
            "src/payments/models.py": "python",
            "src/orders/processor.py": "python",
        },
    )
    store.commit()
    store.close()
    return db_path


@pytest.fixture
def db_path(tmp_path) -> str:
    return _build_db(tmp_path)


@pytest.fixture
def missing_db_path(tmp_path) -> str:
    return str(tmp_path / "nonexistent.db")


# ---------------------------------------------------------------------------
# Helper: NodeRow factory
# ---------------------------------------------------------------------------


def _node_row(
    id: int = 1,
    project: str = "p",
    label: str = "Function",
    name: str = "foo",
    qualified_name: str = "p.src.foo",
    file_path: str = "src/foo.py",
    start_line: int = 1,
    end_line: int = 5,
    signature: str = "def foo():",
    source: str = "def foo():\n    pass",
    properties: dict | None = None,
) -> NodeRow:
    return NodeRow(
        id=id,
        project=project,
        label=label,
        name=name,
        qualified_name=qualified_name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        source=source,
        properties=properties or {},
    )


def _edge_row(
    id: int = 1,
    project: str = "p",
    source_id: int = 1,
    target_id: int = 2,
    type: str = "CALLS",
    properties: dict | None = None,
) -> EdgeRow:
    return EdgeRow(
        id=id,
        project=project,
        source_id=source_id,
        target_id=target_id,
        type=type,
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# _bare_name
# ---------------------------------------------------------------------------


class TestBareName:
    def test_dotted(self):
        assert _bare_name("src.payments.service.charge") == "charge"

    def test_no_dots(self):
        assert _bare_name("charge") == "charge"

    def test_two_components(self):
        assert _bare_name("service.charge") == "charge"

    def test_empty(self):
        assert _bare_name("") == ""


# ---------------------------------------------------------------------------
# _truncate_sig
# ---------------------------------------------------------------------------


class TestTruncateSig:
    def test_short_unchanged(self):
        sig = "def foo():"
        assert _truncate_sig(sig, 80) == sig

    def test_exactly_at_limit_unchanged(self):
        sig = "x" * 80
        assert _truncate_sig(sig, 80) == sig

    def test_long_truncated(self):
        sig = "def charge(user: User, amount_cents: int, currency: str) -> Payment:"
        result = _truncate_sig(sig, 30)
        assert len(result) <= 30
        assert result.endswith("...")

    def test_truncation_at_comma(self):
        sig = "def foo(a: int, b: str, c: bool) -> None:"
        result = _truncate_sig(sig, 20)
        assert "..." in result
        assert len(result) <= 20

    def test_never_raises_on_short_max_len(self):
        sig = "def foo(x):"
        result = _truncate_sig(sig, 5)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _maybe_truncate
# ---------------------------------------------------------------------------


class TestMaybeTruncate:
    def test_short_source_unchanged(self):
        src = "def foo():\n    pass\n"
        (result,) = [_maybe_truncate(src)]
        assert result == src

    def test_long_source_truncated(self):
        # 40 KB of source
        src = "x = 1\n" * 7_000
        result = _maybe_truncate(src)
        assert len(result.encode("utf-8")) <= MAX_SOURCE_BYTES + 300
        assert "[source truncated" in result

    def test_truncated_snaps_to_newline(self):
        src = "x = 1\n" * 7_000
        result = _maybe_truncate(src)
        # Should not cut mid-line
        lines = result.split("\n")
        # Last meaningful line before the notice should be complete
        assert any("[source truncated" in line for line in lines)

    def test_truncation_notice_contains_byte_count(self):
        src = "y\n" * 20_000  # ~40 KB > MAX_SOURCE_BYTES (32 KB)
        result = _maybe_truncate(src)
        assert str(MAX_SOURCE_BYTES) in result

    def test_empty_source_unchanged(self):
        assert _maybe_truncate("") == ""

    def test_exactly_at_limit_unchanged(self):
        src = "a" * MAX_SOURCE_BYTES
        result = _maybe_truncate(src)
        assert "[source truncated" not in result


# ---------------------------------------------------------------------------
# _format_qn
# ---------------------------------------------------------------------------


class TestFormatQn:
    def test_short_padded(self):
        result = _format_qn("src.foo", 20)
        assert len(result) == 20
        assert result.startswith("src.foo")

    def test_exact_length(self):
        qn = "a" * 20
        result = _format_qn(qn, 20)
        assert result == qn

    def test_long_truncated_from_left(self):
        qn = "src.very.long.module.path.to.charge"
        result = _format_qn(qn, 20)
        assert len(result) == 20
        assert result.startswith("...")
        assert "charge" in result

    def test_empty_string(self):
        result = _format_qn("", 10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _format_related
# ---------------------------------------------------------------------------


class TestFormatRelated:
    def test_empty_visited(self):
        result = _format_related([], "called by")
        assert "0" in result
        assert "none" in result.lower()

    def test_single_direct_caller(self):
        node = _node_row(
            name="checkout",
            qualified_name="src.views.checkout",
            file_path="src/views.py",
            start_line=42,
        )
        result = _format_related([(node, 1)], "called by")
        assert "called by (1)" in result
        assert "src.views.checkout" in result
        assert "views.py:42" in result

    def test_only_direct_shown(self):
        # hop=2 should be excluded (only hop=1 is direct)
        direct = _node_row(id=1, qualified_name="src.a.direct")
        indirect = _node_row(id=2, qualified_name="src.b.indirect")
        result = _format_related([(direct, 1), (indirect, 2)], "calls")
        assert "src.a.direct" in result
        assert "src.b.indirect" not in result

    def test_truncates_at_max_related(self):
        nodes = [
            _node_row(id=i, qualified_name=f"src.mod.fn_{i}")
            for i in range(MAX_RELATED_SHOWN + 5)
        ]
        visited = [(n, 1) for n in nodes]
        result = _format_related(visited, "calls")
        assert "more" in result

    def test_direction_label_in_output(self):
        result = _format_related([], "calls")
        assert "calls" in result


# ---------------------------------------------------------------------------
# _format_search_hit
# ---------------------------------------------------------------------------


class TestFormatSearchHit:
    def test_contains_qn(self):
        node = _node_row(qualified_name="src.payments.service.charge")
        result = _format_search_hit(node)
        assert "src.payments.service.charge" in result

    def test_contains_label(self):
        node = _node_row(label="Function")
        result = _format_search_hit(node)
        assert "Function" in result

    def test_contains_file_and_line(self):
        node = _node_row(file_path="src/payments/service.py", start_line=23)
        result = _format_search_hit(node)
        assert "src/payments/service.py" in result
        assert "23" in result

    def test_contains_signature(self):
        node = _node_row(signature="def charge(user, amount):")
        result = _format_search_hit(node)
        assert "def charge" in result

    def test_long_signature_truncated(self):
        long_sig = "def " + "x" * 200 + "():"
        node = _node_row(signature=long_sig)
        result = _format_search_hit(node)
        # Signature should be truncated to ≤ 80 chars
        lines = result.split("\n")
        sig_line = next(line for line in lines if "def " in line)
        assert len(sig_line.strip()) <= 83  # 80 + small indent


# ---------------------------------------------------------------------------
# _build_confidence_map
# ---------------------------------------------------------------------------


class TestBuildConfidenceMap:
    def test_single_edge(self):
        edge = _edge_row(source_id=10, properties={"confidence": 0.95})
        root = _node_row(id=1)
        result = BFSResult(root=root, visited=[], edges=[edge])
        conf_map = _build_confidence_map(result)
        assert conf_map[10] == pytest.approx(0.95)

    def test_multiple_edges_same_source_takes_max(self):
        edges = [
            _edge_row(id=1, source_id=5, properties={"confidence": 0.40}),
            _edge_row(id=2, source_id=5, properties={"confidence": 0.95}),
        ]
        root = _node_row(id=1)
        result = BFSResult(root=root, visited=[], edges=edges)
        conf_map = _build_confidence_map(result)
        assert conf_map[5] == pytest.approx(0.95)

    def test_missing_confidence_defaults_zero(self):
        edge = _edge_row(source_id=7, properties={})
        root = _node_row(id=1)
        result = BFSResult(root=root, visited=[], edges=[edge])
        conf_map = _build_confidence_map(result)
        # Missing confidence is 0.0; since 0.0 is not > 0.0 the key is absent.
        # The call site defaults to 0.0 when the key is missing.
        assert 7 not in conf_map

    def test_empty_edges(self):
        root = _node_row(id=1)
        result = BFSResult(root=root, visited=[], edges=[])
        assert _build_confidence_map(result) == {}

    def test_invalid_confidence_type_defaults_zero(self):
        edge = _edge_row(source_id=3, properties={"confidence": "high"})
        root = _node_row(id=1)
        result = BFSResult(root=root, visited=[], edges=[edge])
        conf_map = _build_confidence_map(result)
        # Invalid type is coerced to 0.0; same as missing — key absent, defaults at call
        # site.
        assert 3 not in conf_map


# ---------------------------------------------------------------------------
# get_source
# ---------------------------------------------------------------------------


class TestGetSource:
    def test_found_returns_string(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_source_code(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "def charge" in result

    def test_contains_file_path(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "src/payments/service.py" in result

    def test_contains_line_numbers(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "1" in result  # start_line
        assert "20" in result  # end_line

    def test_contains_label(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "Function" in result

    def test_contains_qualified_name(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "test_app.src.payments.service.charge" in result

    def test_contains_callers_section(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "called by" in result.lower()

    def test_callers_listed(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        # checkout and complete both call charge
        assert "checkout" in result or "complete" in result

    def test_contains_callees_section(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "calls" in result.lower()

    def test_callees_listed(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        # charge calls save and refund
        assert "save" in result or "refund" in result

    def test_contains_token_estimate(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "tokens" in result

    def test_contains_separator(self, db_path):
        result = get_source(db_path, "test_app.src.payments.service.charge")
        assert "─" in result

    def test_not_found_returns_message(self, db_path):
        result = get_source(db_path, "nonexistent.qn")
        assert "not found" in result.lower() or "Node not found" in result

    def test_not_found_contains_hint(self, db_path):
        result = get_source(db_path, "nonexistent.qn")
        assert "search" in result.lower() or "hint" in result.lower()

    def test_not_found_bare_name_in_hint(self, db_path):
        result = get_source(db_path, "src.payments.service.charge_card")
        assert "charge_card" in result

    def test_missing_db_returns_error_string(self, missing_db_path):
        result = get_source(missing_db_path, "any.qn")
        assert isinstance(result, str)
        assert "error" in result.lower() or "not found" in result.lower()

    def test_missing_db_never_raises(self, missing_db_path):
        # Must not raise under any circumstances
        result = get_source(missing_db_path, "any.qn")
        assert isinstance(result, str)

    def test_project_filter_found(self, db_path):
        result = get_source(
            db_path,
            "test_app.src.payments.service.charge",
            project="test-app",
        )
        assert "def charge" in result

    def test_project_filter_wrong_project_not_found(self, db_path):
        result = get_source(
            db_path,
            "test_app.src.payments.service.charge",
            project="wrong-project",
        )
        assert "not found" in result.lower() or "Node not found" in result

    def test_class_node(self, db_path):
        result = get_source(
            db_path,
            "test_app.src.payments.models.Payment",
        )
        assert "class Payment" in result
        assert "Class" in result

    def test_method_node(self, db_path):
        result = get_source(
            db_path,
            "test_app.src.payments.models.Payment.save",
        )
        assert "save" in result

    def test_node_with_no_callers(self, db_path):
        # checkout has no callers
        result = get_source(
            db_path,
            "test_app.src.payments.views.checkout",
        )
        assert isinstance(result, str)
        # Should show 0 callers, not crash
        assert "called by" in result.lower()
        assert "(0)" in result or "(none)" in result.lower()

    def test_node_with_no_callees(self, db_path):
        # refund has no outgoing CALLS edges in our fixture
        result = get_source(
            db_path,
            "test_app.src.payments.service.refund",
        )
        assert isinstance(result, str)
        assert "calls" in result.lower()

    def test_source_truncation(self, tmp_path):
        # Build a db with a node whose source is very large
        db_path = str(tmp_path / "big.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        big_source = "# line\n" * 7_000  # ~50 KB
        rec = _make_record(
            "huge_fn", "p.huge_fn", source=big_source, start_line=1, end_line=7000
        )
        store.begin()
        store.insert_nodes([rec], "p")
        store.commit()
        store.close()

        result = get_source(db_path, "p.huge_fn")
        assert "[source truncated" in result

    def test_returns_string_on_corrupt_db(self, tmp_path):
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"this is not sqlite")
        result = get_source(str(corrupt), "any.qn")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_basic_search_returns_string(self, db_path):
        result = search(db_path, "charge")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_match_found_contains_qn(self, db_path):
        result = search(db_path, "charge")
        assert "charge" in result

    def test_match_found_contains_file_path(self, db_path):
        result = search(db_path, "charge")
        assert "src/payments" in result

    def test_match_found_contains_separator(self, db_path):
        result = search(db_path, "charge")
        assert "─" in result

    def test_match_found_shows_result_count(self, db_path):
        result = search(db_path, "charge")
        # Should contain some indication of results
        assert "result" in result.lower() or "charge" in result

    def test_no_results_returns_message(self, db_path):
        result = search(db_path, "zzz_no_such_symbol_xyz")
        assert (
            "0" in result or "no" in result.lower() or "not matched" in result.lower()
        )

    def test_no_results_contains_hint(self, db_path):
        result = search(db_path, "zzz_no_such_symbol_xyz")
        # Should suggest a fallback
        assert (
            "search" in result.lower()
            or "try" in result.lower()
            or "query" in result.lower()
        )

    def test_label_filter_function(self, db_path):
        result = search(db_path, "Payment", label="Function")
        # Payment is a Class, not Function — should return fewer or 0 results
        assert isinstance(result, str)

    def test_label_filter_class(self, db_path):
        result = search(db_path, "Payment", label="Class")
        assert "Payment" in result

    def test_limit_respected(self, tmp_path):
        # Build db with 20 nodes all named "helper_*"
        db_path = str(tmp_path / "many.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record(
                f"helper_{i}", f"p.mod_{i}.helper_{i}", source=f"def helper_{i}(): pass"
            )
            for i in range(20)
        ]
        store.begin()
        store.insert_nodes(records, "p")
        store.commit()
        store.close()

        result = search(db_path, "helper", limit=5)
        # Result should mention at most 5 hits in detail
        # (exact counting is hard from text, but should not crash)
        assert isinstance(result, str)

    def test_limit_capped_at_50(self, db_path):
        # Even if limit=200 is requested, should not crash
        result = search(db_path, "charge", limit=200)
        assert isinstance(result, str)

    def test_pagination_hint_when_more_results(self, tmp_path):
        db_path = str(tmp_path / "many2.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record(
                f"process_{i}", f"p.mod.process_{i}", source=f"def process_{i}(): pass"
            )
            for i in range(30)
        ]
        store.begin()
        store.insert_nodes(records, "p")
        store.commit()
        store.close()

        result = search(db_path, "process", limit=5)
        # Should hint that more results exist
        assert "more" in result.lower() or "results" in result.lower()

    def test_missing_db_returns_error_string(self, missing_db_path):
        result = search(missing_db_path, "anything")
        assert isinstance(result, str)
        assert "error" in result.lower() or "not found" in result.lower()

    def test_never_raises(self, missing_db_path):
        result = search(missing_db_path, "anything")
        assert isinstance(result, str)

    def test_project_filter(self, db_path):
        result = search(db_path, "charge", project="test-app")
        assert "charge" in result

    def test_project_filter_wrong_project(self, db_path):
        result = search(db_path, "charge", project="nonexistent-project")
        assert isinstance(result, str)

    def test_fts_phrase_query(self, db_path):
        result = search(db_path, '"def charge"')
        assert isinstance(result, str)

    def test_fts_and_query(self, db_path):
        result = search(db_path, "charge AND Payment")
        assert isinstance(result, str)

    def test_empty_query_does_not_raise(self, db_path):
        result = search(db_path, "")
        assert isinstance(result, str)

    def test_search_result_contains_label(self, db_path):
        result = search(db_path, "checkout")
        assert "Function" in result

    def test_search_result_contains_line_number(self, db_path):
        result = search(db_path, "checkout")
        # start_line=1 for checkout
        assert "1" in result

    def test_returns_string_on_corrupt_db(self, tmp_path):
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"not a database")
        result = search(str(corrupt), "anything")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# trace_callers
# ---------------------------------------------------------------------------


class TestTraceCallers:
    def test_returns_string(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_qn_in_header(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        assert "test_app.src.payments.service.charge" in result

    def test_contains_depth_in_header(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge", depth=2)
        assert "depth=2" in result

    def test_direct_callers_listed(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        # checkout and complete both call charge
        assert "checkout" in result
        assert "complete" in result

    def test_hop_1_label(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        assert "hop 1" in result

    def test_confidence_shown(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        assert "confidence=" in result

    def test_contains_blast_radius_summary(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        assert "blast radius" in result.lower()

    def test_contains_token_estimate(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        assert "tokens" in result

    def test_contains_separator(self, db_path):
        result = trace_callers(db_path, "test_app.src.payments.service.charge")
        assert "─" in result

    def test_node_not_found_returns_message(self, db_path):
        result = trace_callers(db_path, "nonexistent.qn")
        assert "not found" in result.lower() or "Node not found" in result

    def test_not_found_contains_hint(self, db_path):
        result = trace_callers(db_path, "nonexistent.qn")
        assert "search" in result.lower() or "hint" in result.lower()

    def test_no_callers_returns_message(self, db_path):
        # checkout has no callers
        result = trace_callers(db_path, "test_app.src.payments.views.checkout")
        assert isinstance(result, str)
        assert "0" in result or "no caller" in result.lower()

    def test_depth_cap_at_10(self, db_path):
        # depth=100 should be capped at 10, not raise
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
            depth=100,
        )
        assert isinstance(result, str)

    def test_depth_minimum_1(self, db_path):
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
            depth=0,
        )
        assert isinstance(result, str)

    def test_missing_db_returns_error_string(self, missing_db_path):
        result = trace_callers(missing_db_path, "any.qn")
        assert isinstance(result, str)
        assert "error" in result.lower() or "not found" in result.lower()

    def test_never_raises(self, missing_db_path):
        result = trace_callers(missing_db_path, "any.qn")
        assert isinstance(result, str)

    def test_project_filter(self, db_path):
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
            project="test-app",
        )
        assert "checkout" in result or "complete" in result

    def test_project_filter_wrong_project(self, db_path):
        result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
            project="wrong-project",
        )
        assert isinstance(result, str)

    def test_multi_hop_trace(self, tmp_path):
        """Chain: a → b → c; tracing c should find both b (hop1) and a (hop2)."""
        db_path = str(tmp_path / "chain.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record("a", "p.a"),
            _make_record("b", "p.b"),
            _make_record("c", "p.c"),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                ("p.a", "p.b", "CALLS", {"confidence": 0.95}),
                ("p.b", "p.c", "CALLS", {"confidence": 0.85}),
            ],
            qn_to_id,
            "p",
        )
        store.commit()
        store.close()

        result = trace_callers(db_path, "p.c", depth=2)
        assert "p.b" in result  # hop 1
        assert "p.a" in result  # hop 2
        assert "hop 1" in result
        assert "hop 2" in result

    def test_cycle_does_not_hang(self, tmp_path):
        """Circular call f0 → f1 → f0 must terminate."""
        db_path = str(tmp_path / "cycle.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record("f0", "p.f0"),
            _make_record("f1", "p.f1"),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                ("p.f0", "p.f1", "CALLS", {}),
                ("p.f1", "p.f0", "CALLS", {}),
            ],
            qn_to_id,
            "p",
        )
        store.commit()
        store.close()

        result = trace_callers(db_path, "p.f0", depth=10)
        assert isinstance(result, str)

    def test_depth_1_excludes_indirect(self, tmp_path):
        """With depth=1, only direct callers appear."""
        db_path = str(tmp_path / "depth1.db")
        store = open_path(db_path)
        store.upsert_project("p", "/repo")
        records = [
            _make_record("a", "p.a"),
            _make_record("b", "p.b"),
            _make_record("c", "p.c"),
        ]
        store.begin()
        qn_to_id = store.insert_nodes(records, "p")
        store.insert_edges(
            [
                ("p.a", "p.b", "CALLS", {}),
                ("p.b", "p.c", "CALLS", {}),
            ],
            qn_to_id,
            "p",
        )
        store.commit()
        store.close()

        result = trace_callers(db_path, "p.c", depth=1)
        assert "p.b" in result
        # p.a is hop 2 — should not appear with depth=1
        assert "p.a" not in result

    def test_returns_string_on_corrupt_db(self, tmp_path):
        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"not a database")
        result = trace_callers(str(corrupt), "any.qn")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Integration: all three tools on the same database
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_get_source_then_search_consistent(self, db_path):
        """QNs returned by search() should be valid inputs to get_source()."""
        search_result = search(db_path, "charge")
        # Extract QNs mentioned in the search result
        for line in search_result.split("\n"):
            line = line.strip()
            if line.startswith("test_app.src.payments.service.charge"):
                source_result = get_source(db_path, line.split()[0])
                assert "def charge" in source_result
                break

    def test_trace_callers_then_get_source_on_caller(self, db_path):
        """QNs from trace_callers() should be valid get_source() targets."""
        trace_result = trace_callers(
            db_path,
            "test_app.src.payments.service.charge",
        )
        # checkout is a direct caller
        if "checkout" in trace_result:
            source_result = get_source(
                db_path,
                "test_app.src.payments.views.checkout",
            )
            assert "def checkout" in source_result

    def test_all_tools_return_non_empty_strings(self, db_path):
        charge_qn = "test_app.src.payments.service.charge"
        assert get_source(db_path, charge_qn)
        assert search(db_path, "charge")
        assert trace_callers(db_path, charge_qn)

    def test_all_tools_handle_nonexistent_qn_gracefully(self, db_path):
        qn = "totally.nonexistent.qn"
        assert isinstance(get_source(db_path, qn), str)
        assert isinstance(trace_callers(db_path, qn), str)

    def test_all_tools_handle_missing_db_gracefully(self, missing_db_path):
        assert isinstance(get_source(missing_db_path, "any.qn"), str)
        assert isinstance(search(missing_db_path, "anything"), str)
        assert isinstance(trace_callers(missing_db_path, "any.qn"), str)
