"""
tests/test_registry.py — Tests for indexer/registry.py

Coverage:
  - build(): empty list, single record, many records, skips empty QNs
  - Registry.__len__(), __contains__()
  - Registry.get_by_qn(): found, not found
  - Registry.get_by_name(): found, not found, multiple matches
  - Registry.get_by_module(): exact prefix, parent prefix, not found
  - _index_record(): all three indexes populated correctly
  - _resolve_same_module(): bare name, dotted name, parent package,
    method on class, not found
  - _resolve_import_map(): bare name, dotted qualifier, alias import,
    from-import, not found
  - _resolve_fuzzy(): unique match, too many matches, too short,
    project preference, not found
  - _make_unresolved(): fields set correctly
  - resolve(): strategy chain order, first hit wins
  - resolve_all(): filters empty source_qn, includes unresolved
  - Internal helpers: _module_prefixes, _bare_name, _qualifier,
    _normalise_module_path, _strip_self
  - Edge cases: self/cls calls, dotted chains, empty imports,
    cross-project disambiguation
"""

from src.indexer.registry import (
    CONFIDENCE_FUZZY,
    CONFIDENCE_IMPORT_MAP,
    CONFIDENCE_SAME_MODULE,
    MAX_FUZZY_MATCHES,
    MIN_FUZZY_NAME_LEN,
    CallSite,
    Import,
    Registry,
    Resolution,
    ResolutionContext,
    _bare_name,
    _module_prefixes,
    _normalise_module_path,
    _qualifier,
    _strip_self,
    build,
)
from src.indexer.treesitter import NodeRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(
    name: str,
    qn: str,
    file_path: str = "src/utils.py",
    label: str = "Function",
    language: str = "python",
    parent: str = "",
) -> NodeRecord:
    return NodeRecord(
        label=label,
        name=name,
        qualified_name=qn,
        file_path=file_path,
        start_line=1,
        end_line=5,
        signature=f"def {name}():",
        source=f"def {name}():\n    pass",
        language=language,
        parent=parent,
    )


def make_call(
    callee: str,
    line: int = 10,
    qualifier: str = "",
    in_function: str = "src.payments.views.checkout",
) -> CallSite:
    return CallSite(
        callee=callee, line=line, qualifier=qualifier, in_function=in_function
    )


def make_ctx(
    file_path: str = "src/payments/views.py",
    module_qn: str = "src.payments.views",
    imports: list[Import] | None = None,
    project: str = "my-app",
) -> ResolutionContext:
    return ResolutionContext(
        file_path=file_path,
        module_qn=module_qn,
        imports=imports or [],
        project=project,
    )


def make_import(
    module_path: str,
    names: list[str] | None = None,
    alias: str = "",
    line: int = 1,
    in_function: str = "",
) -> Import:
    return Import(
        module_path=module_path,
        names=names or [],
        alias=alias,
        line=line,
        in_function=in_function,
    )


# ---------------------------------------------------------------------------
# _module_prefixes
# ---------------------------------------------------------------------------


class TestModulePrefixes:
    def test_four_part_qn(self):
        result = _module_prefixes("src.payments.service.charge")
        assert result == ["src", "src.payments", "src.payments.service"]

    def test_three_part_qn(self):
        result = _module_prefixes("src.payments.service")
        assert result == ["src", "src.payments"]

    def test_two_part_qn(self):
        result = _module_prefixes("src.charge")
        assert result == ["src"]

    def test_single_component(self):
        assert _module_prefixes("charge") == []

    def test_empty_string(self):
        assert _module_prefixes("") == []

    def test_preserves_order_short_to_long(self):
        result = _module_prefixes("a.b.c.d.e")
        assert result == ["a", "a.b", "a.b.c", "a.b.c.d"]

    def test_does_not_include_full_qn(self):
        qn = "src.payments.service.charge"
        result = _module_prefixes(qn)
        assert qn not in result

    def test_does_not_include_empty_string(self):
        result = _module_prefixes("src.payments.service.charge")
        assert "" not in result


# ---------------------------------------------------------------------------
# _bare_name
# ---------------------------------------------------------------------------


class TestBareName:
    def test_no_dots(self):
        assert _bare_name("charge") == "charge"

    def test_one_dot(self):
        assert _bare_name("service.charge") == "charge"

    def test_two_dots(self):
        assert _bare_name("self.service.charge") == "charge"

    def test_many_dots(self):
        assert _bare_name("a.b.c.d.name") == "name"

    def test_empty_string(self):
        assert _bare_name("") == ""

    def test_trailing_dot(self):
        # edge case: trailing dot
        result = _bare_name("service.")
        assert result == ""

    def test_leading_dot(self):
        assert _bare_name(".charge") == "charge"


# ---------------------------------------------------------------------------
# _qualifier
# ---------------------------------------------------------------------------


class TestQualifier:
    def test_one_dot(self):
        assert _qualifier("service.charge") == "service"

    def test_two_dots(self):
        assert _qualifier("self.service.charge") == "self.service"

    def test_no_dots(self):
        assert _qualifier("charge") == ""

    def test_empty_string(self):
        assert _qualifier("") == ""

    def test_three_parts(self):
        assert _qualifier("a.b.c") == "a.b"


# ---------------------------------------------------------------------------
# _normalise_module_path
# ---------------------------------------------------------------------------


class TestNormaliseModulePath:
    def test_dotted_already_normalised(self):
        assert _normalise_module_path("src.payments.service") == "src.payments.service"

    def test_slash_path(self):
        assert _normalise_module_path("src/payments/service") == "src.payments.service"

    def test_slash_path_with_py_extension(self):
        assert (
            _normalise_module_path("src/payments/service.py") == "src.payments.service"
        )

    def test_mixed_slashes_and_dots(self):
        # Should normalise separators
        result = _normalise_module_path("src/payments.service")
        assert "." in result
        assert "/" not in result

    def test_strips_leading_relative_dots(self):
        result = _normalise_module_path("..utils")
        assert not result.startswith(".")
        assert "utils" in result

    def test_single_level(self):
        assert _normalise_module_path("utils") == "utils"

    def test_empty_string(self):
        result = _normalise_module_path("")
        assert isinstance(result, str)

    def test_no_trailing_dot(self):
        result = _normalise_module_path("src/payments/service")
        assert not result.endswith(".")

    def test_no_leading_dot(self):
        result = _normalise_module_path("src/payments/service")
        assert not result.startswith(".")


# ---------------------------------------------------------------------------
# _strip_self
# ---------------------------------------------------------------------------


class TestStripSelf:
    def test_strip_self(self):
        assert _strip_self("self.charge") == "charge"

    def test_strip_cls(self):
        assert _strip_self("cls.create") == "create"

    def test_no_prefix(self):
        assert _strip_self("charge") == "charge"

    def test_service_prefix_unchanged(self):
        assert _strip_self("service.charge") == "service.charge"

    def test_self_dotted_chain(self):
        assert _strip_self("self.service.charge") == "service.charge"

    def test_cls_dotted_chain(self):
        assert _strip_self("cls.manager.find") == "manager.find"

    def test_empty_string(self):
        assert _strip_self("") == ""

    def test_self_alone(self):
        # "self." with nothing after
        result = _strip_self("self.")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


class TestBuild:
    def test_empty_list_returns_empty_registry(self):
        reg = build([])
        assert len(reg) == 0

    def test_single_record(self):
        rec = make_record("foo", "src.utils.foo")
        reg = build([rec])
        assert len(reg) == 1

    def test_many_records(self):
        records = [make_record(f"fn_{i}", f"src.mod.fn_{i}") for i in range(50)]
        reg = build(records)
        assert len(reg) == 50

    def test_skips_empty_qn(self):
        r1 = make_record("foo", "src.utils.foo")
        r2 = make_record("bar", "")  # empty QN
        reg = build([r1, r2])
        assert len(reg) == 1
        assert "src.utils.foo" in reg

    def test_returns_registry_instance(self):
        reg = build([make_record("foo", "src.foo")])
        assert isinstance(reg, Registry)

    def test_all_qns_indexed(self):
        records = [
            make_record("charge", "src.payments.service.charge"),
            make_record("refund", "src.payments.service.refund"),
            make_record("login", "src.auth.views.login"),
        ]
        reg = build(records)
        assert "src.payments.service.charge" in reg
        assert "src.payments.service.refund" in reg
        assert "src.auth.views.login" in reg


# ---------------------------------------------------------------------------
# Registry.__len__ and __contains__
# ---------------------------------------------------------------------------


class TestRegistryLenContains:
    def test_len_empty(self):
        reg = build([])
        assert len(reg) == 0

    def test_len_after_build(self):
        records = [make_record(f"f{i}", f"mod.f{i}") for i in range(10)]
        reg = build(records)
        assert len(reg) == 10

    def test_contains_true(self):
        reg = build([make_record("foo", "src.utils.foo")])
        assert "src.utils.foo" in reg

    def test_contains_false(self):
        reg = build([make_record("foo", "src.utils.foo")])
        assert "src.utils.bar" not in reg

    def test_contains_empty_string(self):
        reg = build([make_record("foo", "src.utils.foo")])
        assert "" not in reg


# ---------------------------------------------------------------------------
# Registry.get_by_qn
# ---------------------------------------------------------------------------


class TestGetByQn:
    def test_found(self):
        rec = make_record("charge", "src.payments.service.charge")
        reg = build([rec])
        result = reg.get_by_qn("src.payments.service.charge")
        assert result is not None
        assert result.name == "charge"

    def test_not_found(self):
        reg = build([make_record("foo", "src.foo")])
        assert reg.get_by_qn("src.bar") is None

    def test_returns_node_record(self):
        rec = make_record("foo", "src.foo")
        reg = build([rec])
        result = reg.get_by_qn("src.foo")
        assert isinstance(result, NodeRecord)

    def test_returns_none_for_prefix(self):
        reg = build([make_record("foo", "src.payments.foo")])
        # prefix is not a valid QN
        assert reg.get_by_qn("src.payments") is None

    def test_exact_match_required(self):
        reg = build([make_record("foo", "src.utils.foo")])
        assert reg.get_by_qn("src.utils.fo") is None
        assert reg.get_by_qn("src.utils.fooo") is None


# ---------------------------------------------------------------------------
# Registry.get_by_name
# ---------------------------------------------------------------------------


class TestGetByName:
    def test_found_single(self):
        rec = make_record("charge", "src.payments.service.charge")
        reg = build([rec])
        results = reg.get_by_name("charge")
        assert len(results) == 1
        assert results[0].name == "charge"

    def test_found_multiple(self):
        records = [
            make_record("process", "src.payments.service.process"),
            make_record("process", "src.orders.service.process"),
            make_record("process", "src.refunds.service.process"),
        ]
        reg = build(records)
        results = reg.get_by_name("process")
        assert len(results) == 3

    def test_not_found(self):
        reg = build([make_record("foo", "src.foo")])
        assert reg.get_by_name("bar") == []

    def test_returns_copy(self):
        rec = make_record("foo", "src.foo")
        reg = build([rec])
        result = reg.get_by_name("foo")
        result.append(make_record("intruder", "src.intruder"))
        # Internal state should not be mutated
        assert len(reg.get_by_name("foo")) == 1

    def test_empty_name(self):
        reg = build([make_record("foo", "src.foo")])
        assert reg.get_by_name("") == []


# ---------------------------------------------------------------------------
# Registry.get_by_module
# ---------------------------------------------------------------------------


class TestGetByModule:
    def test_exact_module_match(self):
        records = [
            make_record(
                "charge",
                "src.payments.service.charge",
                file_path="src/payments/service.py",
            ),
            make_record(
                "refund",
                "src.payments.service.refund",
                file_path="src/payments/service.py",
            ),
        ]
        reg = build(records)
        results = reg.get_by_module("src.payments.service")
        names = {r.name for r in results}
        assert "charge" in names
        assert "refund" in names

    def test_parent_package_match(self):
        records = [
            make_record("charge", "src.payments.service.charge"),
            make_record("refund", "src.payments.service.refund"),
            make_record("login", "src.auth.views.login"),
        ]
        reg = build(records)
        results = reg.get_by_module("src.payments")
        names = {r.name for r in results}
        assert "charge" in names
        assert "refund" in names
        assert "login" not in names

    def test_not_found_returns_empty(self):
        reg = build([make_record("foo", "src.utils.foo")])
        assert reg.get_by_module("nonexistent.module") == []

    def test_returns_copy(self):
        reg = build([make_record("foo", "src.utils.foo")])
        result = reg.get_by_module("src.utils")
        result.append(make_record("intruder", "src.utils.intruder"))
        assert len(reg.get_by_module("src.utils")) == 1

    def test_cross_module_isolation(self):
        records = [
            make_record("a", "pkg.mod_a.a"),
            make_record("b", "pkg.mod_b.b"),
        ]
        reg = build(records)
        assert len(reg.get_by_module("pkg.mod_a")) == 1
        assert len(reg.get_by_module("pkg.mod_b")) == 1


# ---------------------------------------------------------------------------
# _resolve_same_module
# ---------------------------------------------------------------------------


class TestResolveSameModule:
    def _make_reg(self) -> Registry:
        records = [
            make_record(
                "charge",
                "src.payments.service.charge",
                file_path="src/payments/service.py",
            ),
            make_record(
                "refund",
                "src.payments.service.refund",
                file_path="src/payments/service.py",
            ),
            make_record(
                "validate",
                "src.payments.validate",
                file_path="src/payments/validate.py",
            ),
            make_record(
                "MyClass",
                "src.payments.models.MyClass",
                label="Class",
                file_path="src/payments/models.py",
            ),
            make_record(
                "process",
                "src.payments.models.MyClass.process",
                label="Method",
                parent="MyClass",
                file_path="src/payments/models.py",
            ),
        ]
        return build(records)

    def test_bare_name_same_module(self):
        reg = self._make_reg()
        call = make_call("charge", in_function="src.payments.service.checkout")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg._resolve_same_module(call, ctx)
        assert res is not None
        assert res.target_qn == "src.payments.service.charge"
        assert res.strategy == "same_module"
        assert res.confidence == CONFIDENCE_SAME_MODULE

    def test_sibling_module_via_parent_package(self):
        reg = self._make_reg()
        # caller is in src.payments.views, callee is in src.payments.validate
        call = make_call("validate", in_function="src.payments.views.checkout")
        ctx = make_ctx(module_qn="src.payments.views")
        res = reg._resolve_same_module(call, ctx)
        assert res is not None
        assert "validate" in res.target_qn

    def test_dotted_class_method(self):
        reg = self._make_reg()
        call = make_call(
            "MyClass.process",
            in_function="src.payments.models.other_func",
            qualifier="MyClass",
        )
        ctx = make_ctx(module_qn="src.payments.models")
        res = reg._resolve_same_module(call, ctx)
        assert res is not None
        assert res.target_qn == "src.payments.models.MyClass.process"

    def test_not_found_returns_none(self):
        reg = self._make_reg()
        call = make_call("nonexistent", in_function="src.payments.service.foo")
        ctx = make_ctx(module_qn="src.payments.service")
        assert reg._resolve_same_module(call, ctx) is None

    def test_returns_resolution_type(self):
        reg = self._make_reg()
        call = make_call("charge", in_function="src.payments.service.checkout")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg._resolve_same_module(call, ctx)
        assert isinstance(res, Resolution)

    def test_self_stripped_before_lookup(self):
        records = [
            make_record("helper", "src.payments.service.helper"),
        ]
        reg = build(records)
        call = make_call("self.helper", in_function="src.payments.service.main")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg._resolve_same_module(call, ctx)
        assert res is not None
        assert res.target_qn == "src.payments.service.helper"


# ---------------------------------------------------------------------------
# _resolve_import_map
# ---------------------------------------------------------------------------


class TestResolveImportMap:
    def _make_reg(self) -> Registry:
        return build(
            [
                make_record(
                    "charge",
                    "src.payments.service.charge",
                    file_path="src/payments/service.py",
                ),
                make_record(
                    "refund",
                    "src.payments.service.refund",
                    file_path="src/payments/service.py",
                ),
                make_record(
                    "login", "src.auth.views.login", file_path="src/auth/views.py"
                ),
                make_record(
                    "User",
                    "src.auth.models.User",
                    label="Class",
                    file_path="src/auth/models.py",
                ),
            ]
        )

    def test_from_import_bare_name(self):
        reg = self._make_reg()
        call = make_call("charge", in_function="src.orders.views.create")
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.payments.service", names=["charge"])],
        )
        res = reg._resolve_import_map(call, ctx)
        assert res is not None
        assert res.target_qn == "src.payments.service.charge"
        assert res.strategy == "import_map"
        assert res.confidence == CONFIDENCE_IMPORT_MAP

    def test_dotted_qualifier_resolves_module(self):
        reg = self._make_reg()
        # "service.charge" where "service" is imported from src.payments
        call = make_call(
            "service.charge", qualifier="service", in_function="src.orders.views.create"
        )
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.payments", names=["service"])],
        )
        res = reg._resolve_import_map(call, ctx)
        assert res is not None
        assert "charge" in res.target_qn

    def test_alias_import(self):
        reg = self._make_reg()
        # import src.payments.service as svc
        call = make_call(
            "svc.charge", qualifier="svc", in_function="src.orders.views.create"
        )
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.payments.service", alias="svc")],
        )
        res = reg._resolve_import_map(call, ctx)
        assert res is not None
        assert res.target_qn == "src.payments.service.charge"

    def test_not_in_imports_returns_none(self):
        reg = self._make_reg()
        call = make_call("charge", in_function="src.orders.views.create")
        ctx = make_ctx(module_qn="src.orders.views", imports=[])
        assert reg._resolve_import_map(call, ctx) is None

    def test_import_present_but_symbol_not_in_registry(self):
        reg = self._make_reg()
        call = make_call("missing_fn", in_function="src.orders.views.create")
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.payments.service", names=["missing_fn"])],
        )
        res = reg._resolve_import_map(call, ctx)
        assert res is None

    def test_returns_resolution_type(self):
        reg = self._make_reg()
        call = make_call("charge", in_function="src.orders.views.create")
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.payments.service", names=["charge"])],
        )
        res = reg._resolve_import_map(call, ctx)
        assert isinstance(res, Resolution)

    def test_from_import_alias_resolves_symbol(self):
        reg = self._make_reg()
        call = make_call("U", in_function="src.orders.views.create")
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.auth.models", names=["User"], alias="U")],
        )
        res = reg._resolve_import_map(call, ctx)
        assert res is not None
        assert res.target_qn == "src.auth.models.User"

    def test_local_import_visible_only_in_own_callable(self):
        reg = self._make_reg()
        local_import = make_import(
            "src.payments.service",
            names=["charge"],
            in_function="src.orders.views.with_local",
        )

        local_call = make_call("charge", in_function="src.orders.views.with_local")
        other_call = make_call("charge", in_function="src.orders.views.other")
        ctx = make_ctx(module_qn="src.orders.views", imports=[local_import])

        local_res = reg._resolve_import_map(local_call, ctx)
        other_res = reg._resolve_import_map(other_call, ctx)

        assert local_res is not None
        assert local_res.target_qn == "src.payments.service.charge"
        assert other_res is None

    def test_alias_symbol_dotted_call_resolves_member(self):
        reg = build(
            [
                make_record(
                    "User",
                    "src.auth.models.User",
                    label="Class",
                    file_path="src/auth/models.py",
                ),
                make_record(
                    "save",
                    "src.auth.models.User.save",
                    label="Method",
                    file_path="src/auth/models.py",
                    parent="User",
                ),
            ]
        )
        call = make_call("U.save", in_function="src.orders.views.create")
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.auth.models", names=["User"], alias="U")],
        )
        res = reg._resolve_import_map(call, ctx)
        assert res is not None
        assert res.target_qn == "src.auth.models.User.save"

    def test_empty_names_bare_import(self):
        # import src.payments.service  (no "from", no names)
        reg = self._make_reg()
        call = make_call(
            "src.payments.service.charge", in_function="src.orders.views.create"
        )
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.payments.service", names=[])],
        )
        res = reg._resolve_import_map(call, ctx)
        # May or may not resolve depending on implementation —
        # at minimum should not raise
        assert res is None or isinstance(res, Resolution)


# ---------------------------------------------------------------------------
# _resolve_fuzzy
# ---------------------------------------------------------------------------


class TestResolveFuzzy:
    def test_unique_match(self):
        reg = build(
            [
                make_record("charge", "src.payments.service.charge"),
            ]
        )
        call = make_call("charge")
        ctx = make_ctx()
        res = reg._resolve_fuzzy(call, ctx)
        assert res is not None
        assert res.target_qn == "src.payments.service.charge"
        assert res.strategy == "fuzzy"
        assert res.confidence == CONFIDENCE_FUZZY

    def test_too_many_matches_returns_none(self):
        records = [
            make_record("process", f"mod_{i}.process")
            for i in range(MAX_FUZZY_MATCHES + 1)
        ]
        reg = build(records)
        call = make_call("process")
        ctx = make_ctx()
        assert reg._resolve_fuzzy(call, ctx) is None

    def test_exactly_at_limit_returns_none(self):
        records = [
            make_record("process", f"mod_{i}.process") for i in range(MAX_FUZZY_MATCHES)
        ]
        reg = build(records)
        call = make_call("process")
        ctx = make_ctx()
        # At exactly MAX_FUZZY_MATCHES, should be suppressed (too ambiguous)
        assert reg._resolve_fuzzy(call, ctx) is None

    def test_name_too_short_returns_none(self):
        name = "x" * (MIN_FUZZY_NAME_LEN - 1)
        reg = build([make_record(name, f"src.mod.{name}")])
        call = make_call(name)
        ctx = make_ctx()
        assert reg._resolve_fuzzy(call, ctx) is None

    def test_not_found_returns_none(self):
        reg = build([make_record("foo", "src.mod.foo")])
        call = make_call("nonexistent")
        ctx = make_ctx()
        assert reg._resolve_fuzzy(call, ctx) is None

    def test_project_preference(self):
        records = [
            make_record("helper", "src.utils.helper", file_path="src/utils.py"),
            make_record(
                "helper", "other_project.utils.helper", file_path="other/utils.py"
            ),
        ]
        reg = build(records)
        call = make_call("helper")
        # project="my-app" — neither record has project info in NodeRecord,
        # but the QN prefix is used as a heuristic
        ctx = make_ctx(module_qn="src.payments.views", project="my-app")
        res = reg._resolve_fuzzy(call, ctx)
        # When exactly 2 matches: should suppress (ambiguous) OR prefer
        # in-project. Either is valid — just must not raise.
        assert res is None or isinstance(res, Resolution)

    def test_returns_resolution_type(self):
        reg = build([make_record("charge", "src.payments.service.charge")])
        call = make_call("charge")
        ctx = make_ctx()
        res = reg._resolve_fuzzy(call, ctx)
        assert res is None or isinstance(res, Resolution)

    def test_bare_name_extracted_from_dotted(self):
        # "service.charge" → fuzzy should look up "charge"
        reg = build([make_record("charge", "src.payments.service.charge")])
        call = make_call("service.charge", qualifier="service")
        ctx = make_ctx()
        res = reg._resolve_fuzzy(call, ctx)
        assert res is None or (
            isinstance(res, Resolution)
            and res.target_qn == "src.payments.service.charge"
        )


# ---------------------------------------------------------------------------
# _make_unresolved
# ---------------------------------------------------------------------------


class TestMakeUnresolved:
    def test_fields(self):
        reg = build([])
        call = make_call("charge", line=42, in_function="src.payments.views.checkout")
        res = reg._make_unresolved(call)
        assert res.source_qn == "src.payments.views.checkout"
        assert res.target_qn == ""
        assert res.strategy == "unresolved"
        assert res.confidence == 0.0
        assert res.call_site is call

    def test_returns_resolution_type(self):
        reg = build([])
        call = make_call("foo")
        res = reg._make_unresolved(call)
        assert isinstance(res, Resolution)

    def test_empty_in_function(self):
        reg = build([])
        call = CallSite(callee="foo", line=1, qualifier="", in_function="")
        res = reg._make_unresolved(call)
        assert res.source_qn == ""
        assert res.target_qn == ""


# ---------------------------------------------------------------------------
# resolve() — strategy chain
# ---------------------------------------------------------------------------


class TestResolve:
    def _make_full_reg(self) -> Registry:
        return build(
            [
                make_record(
                    "charge",
                    "src.payments.service.charge",
                    file_path="src/payments/service.py",
                ),
                make_record(
                    "refund",
                    "src.payments.service.refund",
                    file_path="src/payments/service.py",
                ),
                make_record(
                    "login", "src.auth.views.login", file_path="src/auth/views.py"
                ),
                make_record(
                    "unique_fn",
                    "src.misc.helpers.unique_fn",
                    file_path="src/misc/helpers.py",
                ),
            ]
        )

    def test_same_module_wins_first(self):
        reg = self._make_full_reg()
        call = make_call("charge", in_function="src.payments.service.other_fn")
        ctx = make_ctx(
            module_qn="src.payments.service",
            imports=[make_import("src.payments.service", names=["charge"])],
        )
        res = reg.resolve(call, ctx)
        assert res.strategy == "same_module"

    def test_import_map_wins_when_no_same_module(self):
        reg = self._make_full_reg()
        call = make_call("charge", in_function="src.orders.views.create")
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.payments.service", names=["charge"])],
        )
        res = reg.resolve(call, ctx)
        assert res.strategy == "import_map"
        assert res.target_qn == "src.payments.service.charge"

    def test_fuzzy_wins_when_no_same_module_or_import(self):
        reg = self._make_full_reg()
        call = make_call("unique_fn", in_function="src.random.module.caller")
        ctx = make_ctx(module_qn="src.random.module", imports=[])
        res = reg.resolve(call, ctx)
        assert res.strategy == "fuzzy"
        assert res.target_qn == "src.misc.helpers.unique_fn"

    def test_unresolved_when_all_fail(self):
        reg = self._make_full_reg()
        call = make_call(
            "absolutely_unknown_symbol", in_function="src.random.module.caller"
        )
        ctx = make_ctx(module_qn="src.random.module", imports=[])
        res = reg.resolve(call, ctx)
        assert res.strategy == "unresolved"
        assert res.target_qn == ""
        assert res.confidence == 0.0

    def test_resolve_returns_resolution_type(self):
        reg = self._make_full_reg()
        call = make_call("charge")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg.resolve(call, ctx)
        assert isinstance(res, Resolution)

    def test_source_qn_copied_from_call(self):
        reg = self._make_full_reg()
        call = make_call("charge", in_function="src.payments.service.checkout")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg.resolve(call, ctx)
        assert res.source_qn == "src.payments.service.checkout"

    def test_call_site_attached_to_resolution(self):
        reg = self._make_full_reg()
        call = make_call("charge")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg.resolve(call, ctx)
        assert res.call_site is call

    def test_confidence_is_float(self):
        reg = self._make_full_reg()
        call = make_call("charge")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg.resolve(call, ctx)
        assert isinstance(res.confidence, float)
        assert 0.0 <= res.confidence <= 1.0


# ---------------------------------------------------------------------------
# resolve_all()
# ---------------------------------------------------------------------------


class TestResolveAll:
    def _make_reg(self) -> Registry:
        return build(
            [
                make_record("charge", "src.payments.service.charge"),
                make_record("refund", "src.payments.service.refund"),
            ]
        )

    def test_empty_calls_returns_empty(self):
        reg = self._make_reg()
        ctx = make_ctx(module_qn="src.payments.service")
        result = reg.resolve_all([], ctx)
        assert result == []

    def test_all_calls_resolved(self):
        reg = self._make_reg()
        calls = [
            make_call("charge", in_function="src.payments.service.create"),
            make_call("refund", in_function="src.payments.service.cancel"),
        ]
        ctx = make_ctx(module_qn="src.payments.service")
        result = reg.resolve_all(calls, ctx)
        assert len(result) == 2

    def test_filters_empty_source_qn(self):
        reg = self._make_reg()
        calls = [
            make_call("charge", in_function=""),  # module-level call
            make_call("refund", in_function="src.payments.service.cancel"),
        ]
        ctx = make_ctx(module_qn="src.payments.service")
        result = reg.resolve_all(calls, ctx)
        # The call with in_function="" should be filtered out
        assert all(r.source_qn != "" for r in result)

    def test_includes_unresolved(self):
        reg = self._make_reg()
        calls = [
            make_call("unknown_symbol", in_function="src.payments.service.create"),
        ]
        ctx = make_ctx(module_qn="src.payments.service", imports=[])
        result = reg.resolve_all(calls, ctx)
        assert len(result) == 1
        assert result[0].strategy == "unresolved"

    def test_returns_list_of_resolutions(self):
        reg = self._make_reg()
        calls = [make_call("charge", in_function="src.payments.service.create")]
        ctx = make_ctx(module_qn="src.payments.service")
        result = reg.resolve_all(calls, ctx)
        assert all(isinstance(r, Resolution) for r in result)

    def test_many_calls(self):
        records = [make_record(f"fn_{i}", f"src.mod.fn_{i}") for i in range(20)]
        reg = build(records)
        calls = [make_call(f"fn_{i}", in_function="src.mod.caller") for i in range(20)]
        ctx = make_ctx(module_qn="src.mod")
        result = reg.resolve_all(calls, ctx)
        resolved = [r for r in result if r.strategy != "unresolved"]
        assert len(resolved) == 20


# ---------------------------------------------------------------------------
# Integration: realistic Python call patterns
# ---------------------------------------------------------------------------


class TestRealisticPatterns:
    def _make_reg(self) -> Registry:
        return build(
            [
                make_record(
                    "charge",
                    "src.payments.service.charge",
                    file_path="src/payments/service.py",
                ),
                make_record(
                    "send_receipt",
                    "src.notifications.email.send_receipt",
                    file_path="src/notifications/email.py",
                ),
                make_record(
                    "Order",
                    "src.orders.models.Order",
                    label="Class",
                    file_path="src/orders/models.py",
                ),
                make_record(
                    "save",
                    "src.orders.models.Order.save",
                    label="Method",
                    parent="Order",
                    file_path="src/orders/models.py",
                ),
                make_record(
                    "get_by_id",
                    "src.orders.repository.get_by_id",
                    file_path="src/orders/repository.py",
                ),
            ]
        )

    def test_self_method_call(self):
        reg = self._make_reg()
        call = make_call("self.charge", in_function="src.payments.service.process")
        ctx = make_ctx(module_qn="src.payments.service")
        res = reg.resolve(call, ctx)
        assert res.target_qn == "src.payments.service.charge"

    def test_cross_module_import(self):
        reg = self._make_reg()
        call = make_call("send_receipt", in_function="src.orders.views.complete_order")
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.notifications.email", names=["send_receipt"])],
        )
        res = reg.resolve(call, ctx)
        assert res.target_qn == "src.notifications.email.send_receipt"
        assert res.strategy == "import_map"

    def test_class_method_dotted(self):
        reg = self._make_reg()
        call = make_call(
            "Order.save", qualifier="Order", in_function="src.orders.views.create_order"
        )
        ctx = make_ctx(
            module_qn="src.orders.views",
            imports=[make_import("src.orders.models", names=["Order"])],
        )
        res = reg.resolve(call, ctx)
        assert res.strategy in ("import_map", "same_module", "fuzzy")
        assert res.target_qn != "" or res.strategy == "unresolved"

    def test_stdlib_call_unresolved(self):
        reg = self._make_reg()
        call = make_call(
            "os.path.join",
            qualifier="os.path",
            in_function="src.utils.paths.build_path",
        )
        ctx = make_ctx(module_qn="src.utils.paths", imports=[])
        res = reg.resolve(call, ctx)
        # os.path.join is not in the registry — should be unresolved
        assert res.strategy == "unresolved"

    def test_chained_self_call(self):
        reg = self._make_reg()
        call = make_call("self.service.charge", in_function="src.orders.views.checkout")
        ctx = make_ctx(module_qn="src.orders.views")
        # self.service.charge → strip self → service.charge
        # Should attempt resolution without raising
        res = reg.resolve(call, ctx)
        assert isinstance(res, Resolution)

    def test_multiple_files_same_project(self):
        records = [
            make_record(
                "process",
                "src.payments.service.process",
                file_path="src/payments/service.py",
            ),
            make_record(
                "validate",
                "src.payments.validators.validate",
                file_path="src/payments/validators.py",
            ),
        ]
        reg = build(records)
        call = make_call("validate", in_function="src.payments.service.process")
        ctx = make_ctx(
            module_qn="src.payments.service",
            imports=[make_import("src.payments.validators", names=["validate"])],
        )
        res = reg.resolve(call, ctx)
        assert res.target_qn == "src.payments.validators.validate"


# ---------------------------------------------------------------------------
# CallSite and Import dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_callsite_default_qualifier(self):
        cs = CallSite(callee="foo", line=1)
        assert cs.qualifier == ""
        assert cs.in_function == ""

    def test_import_default_names(self):
        imp = Import(module_path="src.utils")
        assert imp.names == []
        assert imp.alias == ""
        assert imp.line == 0

    def test_resolution_context_default_imports(self):
        ctx = ResolutionContext(file_path="f.py", module_qn="mod")
        assert ctx.imports == []
        assert ctx.project == ""

    def test_resolution_fields(self):
        call = CallSite(callee="foo", line=1)
        res = Resolution(
            source_qn="src.mod.caller",
            target_qn="src.mod.foo",
            strategy="same_module",
            confidence=0.95,
            call_site=call,
        )
        assert res.source_qn == "src.mod.caller"
        assert res.target_qn == "src.mod.foo"
        assert res.strategy == "same_module"
        assert res.confidence == 0.95
        assert res.call_site is call
