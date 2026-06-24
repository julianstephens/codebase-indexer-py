"""
tests/test_extractor.py — Tests for indexer/extractor.py

Coverage:
  - extract_file() routing: treesitter path, fallback paths, skip path
  - extract_file_detailed() metadata: extractor, reason, error, language
  - extract_file() on a real .py file and a real .ts file (checkpoint 5)
  - extract_files() batch API: normal files, read errors, oversized files
  - NodeRecord field correctness coming out of extract_file()
"""

import textwrap

from src.indexer.extractor import (
    ExtractionResult,
    FileInfo,
    extract_file,
    extract_file_detailed,
    extract_files,
)
from src.indexer.fallback import MAX_FILE_BYTES
from src.indexer.treesitter import NodeRecord


def dedent(source: str) -> str:
    return textwrap.dedent(source).strip()


# ---------------------------------------------------------------------------
# Routing: treesitter path
# ---------------------------------------------------------------------------


class TestExtractFileTreesitterPath:
    """extract_file() calls tree-sitter for recognised languages."""

    PY_SOURCE = dedent(
        """
        def add(x: int, y: int) -> int:
            return x + y

        class Calculator:
            def multiply(self, x: int, y: int) -> int:
                return x * y
        """
    )

    TS_SOURCE = dedent(
        """
        function greet(name: string): string {
            return `Hello, ${name}`;
        }

        class Greeter {
            greet(name: string): string {
                return `Hi, ${name}`;
            }
        }
        """
    )

    def test_py_returns_list_of_node_records(self):
        records = extract_file("src/calc.py", self.PY_SOURCE)
        assert isinstance(records, list)
        assert all(isinstance(r, NodeRecord) for r in records)

    def test_py_extracts_function(self):
        records = extract_file("src/calc.py", self.PY_SOURCE)
        names = [r.name for r in records]
        assert "add" in names

    def test_py_extracts_class(self):
        records = extract_file("src/calc.py", self.PY_SOURCE)
        names = [r.name for r in records]
        assert "Calculator" in names

    def test_py_extracts_method(self):
        records = extract_file("src/calc.py", self.PY_SOURCE)
        names = [r.name for r in records]
        assert "multiply" in names

    def test_py_node_record_fields(self):
        records = extract_file("src/calc.py", self.PY_SOURCE)
        fn = next(r for r in records if r.name == "add")
        assert fn.label == "Function"
        assert fn.file_path == "src/calc.py"
        assert fn.language == "python"
        assert fn.qualified_name == ""  # pipeline sets this
        assert fn.start_line >= 1
        assert fn.end_line >= fn.start_line
        assert "add" in fn.signature
        assert "return x + y" in fn.source

    def test_py_method_has_parent(self):
        records = extract_file("src/calc.py", self.PY_SOURCE)
        method = next(r for r in records if r.name == "multiply")
        assert method.parent == "Calculator"

    def test_ts_returns_list_of_node_records(self):
        records = extract_file("src/greeter.ts", self.TS_SOURCE)
        assert isinstance(records, list)
        assert all(isinstance(r, NodeRecord) for r in records)

    def test_ts_extracts_function(self):
        records = extract_file("src/greeter.ts", self.TS_SOURCE)
        names = [r.name for r in records]
        assert "greet" in names

    def test_ts_extracts_class(self):
        records = extract_file("src/greeter.ts", self.TS_SOURCE)
        names = [r.name for r in records]
        assert "Greeter" in names

    def test_ts_node_record_fields(self):
        records = extract_file("src/greeter.ts", self.TS_SOURCE)
        fn = next(r for r in records if r.name == "greet" and r.parent == "")
        assert fn.label == "Function"
        assert fn.file_path == "src/greeter.ts"
        assert fn.language == "typescript"
        assert fn.start_line >= 1
        assert fn.end_line >= fn.start_line
        assert "greet" in fn.signature
        assert "Hello" in fn.source

    def test_detailed_treesitter_extractor_label(self):
        result = extract_file_detailed("src/calc.py", self.PY_SOURCE)
        assert result.extractor == "treesitter"
        assert result.reason == ""
        assert result.error == ""
        assert result.language == "python"

    def test_detailed_ts_extractor_label(self):
        result = extract_file_detailed("src/greeter.ts", self.TS_SOURCE)
        assert result.extractor == "treesitter"
        assert result.language == "typescript"


# ---------------------------------------------------------------------------
# Routing: skip path
# ---------------------------------------------------------------------------


class TestExtractFileSkipPath:
    """extract_file() returns [] for files that should be skipped."""

    def test_lock_file_returns_empty(self):
        assert extract_file("package-lock.json", '{"lockfileVersion": 2}') == []

    def test_lock_file_detailed_skip(self):
        result = extract_file_detailed("package-lock.json", "{}")
        assert result.extractor == "skip"
        assert result.records == []

    def test_binary_extension_returns_empty(self):
        assert extract_file("module.pyc", b"\x00" * 10) == []

    def test_oversized_file_returns_empty(self):
        big_source = "x = 1\n" * (MAX_FILE_BYTES // 6 + 1)
        assert extract_file("src/huge.py", big_source) == []


# ---------------------------------------------------------------------------
# Routing: fallback path — unrecognised language
# ---------------------------------------------------------------------------


class TestExtractFileFallbackNoLanguage:
    """Unrecognised extensions produce a single File fallback record."""

    def test_dockerfile_returns_one_record(self):
        source = "FROM python:3.13\nRUN pip install .\n"
        records = extract_file("Dockerfile", source)
        assert len(records) == 1

    def test_dockerfile_fallback_label(self):
        source = "FROM python:3.13\nRUN pip install .\n"
        records = extract_file("Dockerfile", source)
        assert records[0].label == "File"

    def test_dockerfile_fallback_properties(self):
        source = "FROM python:3.13\nRUN pip install .\n"
        records = extract_file("Dockerfile", source)
        assert records[0].properties.get("fallback") is True
        assert records[0].properties.get("reason") == "no_language"

    def test_yaml_fallback(self):
        source = "version: '3'\nservices:\n  web:\n    image: nginx\n"
        records = extract_file("docker-compose.yml", source)
        assert len(records) == 1
        assert records[0].label == "File"

    def test_detailed_fallback_reason_no_language(self):
        source = "hello world\n"
        result = extract_file_detailed("README.md", source)
        assert result.extractor == "fallback"
        assert result.reason == "no_language"


# ---------------------------------------------------------------------------
# Routing: fallback path — no definitions in recognised language
# ---------------------------------------------------------------------------


class TestExtractFileFallbackNoDefinitions:
    """
    A Python file with no functions or classes falls back because
    tree-sitter returns an empty result.
    """

    IMPORTS_ONLY = dedent(
        """
        import os
        import sys

        PATH = os.environ.get("HOME", "/tmp")
        DEBUG = True
        """
    )

    def test_constants_file_falls_back(self):
        result = extract_file_detailed("src/constants.py", self.IMPORTS_ONLY)
        assert result.extractor == "fallback"
        assert result.reason == "no_definitions"

    def test_constants_file_has_one_record(self):
        records = extract_file("src/constants.py", self.IMPORTS_ONLY)
        assert len(records) == 1
        assert records[0].label == "File"

    def test_empty_py_file_falls_back(self):
        result = extract_file_detailed("src/empty.py", "")
        assert result.extractor == "fallback"


# ---------------------------------------------------------------------------
# Semantic relationship extraction (real source)
# ---------------------------------------------------------------------------


class TestExtractSemanticRelationshipsPython:
    SOURCE = dedent(
        """
        import pkg.mod as pm
        from pkg.sub import thing as local_thing

        def outer(x):
            import pkg.inner as inner_mod

            def nested():
                return inner_mod.run(x)

            pm.call(x)
            local_thing(x)
            return nested()
        """
    )

    def test_extracts_calls_from_real_source(self):
        records = extract_file("src/example.py", self.SOURCE)
        outer = next(r for r in records if r.name == "outer")

        calls = outer.properties.get("calls", [])
        callees = {item["callee"] for item in calls if isinstance(item, dict)}
        assert "pm.call" in callees
        assert "local_thing" in callees
        assert "nested" in callees

    def test_nested_call_not_attributed_to_outer(self):
        records = extract_file("src/example.py", self.SOURCE)
        outer = next(r for r in records if r.name == "outer")
        nested = next(r for r in records if r.name == "nested")

        outer_callees = {
            item["callee"]
            for item in outer.properties.get("calls", [])
            if isinstance(item, dict)
        }
        nested_callees = {
            item["callee"]
            for item in nested.properties.get("calls", [])
            if isinstance(item, dict)
        }

        assert "inner_mod.run" not in outer_callees
        assert "inner_mod.run" in nested_callees

    def test_extracts_import_bindings(self):
        records = extract_file("src/example.py", self.SOURCE)
        outer = next(r for r in records if r.name == "outer")

        imports = outer.properties.get("imports", [])
        modules = {
            item["module_path"]
            for item in imports
            if isinstance(item, dict) and "module_path" in item
        }
        assert "pkg.mod" in modules
        assert "pkg.sub" in modules
        assert "pkg.inner" in modules


class TestExtractSemanticRelationshipsTypescript:
    SOURCE = dedent(
        """
        import * as svc from "src/payments/service";
        import { charge as payCharge } from "src/payments/service";

        function checkout(amount: number): number {
            svc.charge(amount);
            payCharge(amount);
            return amount;
        }
        """
    )

    def test_extracts_calls_from_real_ts_source(self):
        records = extract_file("src/checkout.ts", self.SOURCE)
        checkout = next(r for r in records if r.name == "checkout")

        calls = checkout.properties.get("calls", [])
        callees = {item["callee"] for item in calls if isinstance(item, dict)}
        assert "svc.charge" in callees
        assert "payCharge" in callees

    def test_extracts_ts_import_aliases(self):
        records = extract_file("src/checkout.ts", self.SOURCE)
        checkout = next(r for r in records if r.name == "checkout")

        imports = checkout.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("src/payments/service", tuple(), "svc") in as_map
        assert ("src/payments/service", ("charge",), "payCharge") in as_map


class TestExtractSemanticRelationshipsOtherLanguages:
    def test_extracts_go_import_alias(self):
        source = dedent(
            """
            package main

            import svc "example.com/payments/service"

            func checkout(amount int) int {
                return svc.Charge(amount)
            }
            """
        )
        records = extract_file("src/checkout.go", source)
        checkout = next(r for r in records if r.name == "checkout")

        imports = checkout.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("example.com/payments/service", ("service",), "svc") in as_map

    def test_extracts_java_import_symbol(self):
        source = dedent(
            """
            import java.util.List;

            class Checkout {
                void run() {}
            }
            """
        )
        records = extract_file("src/Checkout.java", source)
        run_method = next(r for r in records if r.name == "run")

        imports = run_method.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("java.util", ("List",), "") in as_map

    def test_extracts_java_static_wildcard_import(self):
        source = dedent(
            """
            import static java.util.Collections.*;

            class Checkout {
                void run() {}
            }
            """
        )
        records = extract_file("src/Checkout.java", source)
        run_method = next(r for r in records if r.name == "run")

        imports = run_method.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("java.util.Collections", tuple(), "") in as_map

    def test_extracts_csharp_alias_using(self):
        source = dedent(
            """
            using SB = System.Text.StringBuilder;

            class Checkout {
                void Run() {}
            }
            """
        )
        records = extract_file("src/Checkout.cs", source)
        run_method = next(r for r in records if r.name == "Run")

        imports = run_method.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("System.Text", ("StringBuilder",), "SB") in as_map

    def test_extracts_csharp_using_static(self):
        source = dedent(
            """
            using static System.Math;

            class Checkout {
                void Run() {}
            }
            """
        )
        records = extract_file("src/Checkout.cs", source)
        run_method = next(r for r in records if r.name == "Run")

        imports = run_method.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("System", ("Math",), "") in as_map

    def test_extracts_rust_use_alias(self):
        source = dedent(
            """
            use crate::payments::service::Charge as PayCharge;

            fn checkout(amount: i32) -> i32 {
                amount
            }
            """
        )
        records = extract_file("src/checkout.rs", source)
        checkout = next(r for r in records if r.name == "checkout")

        imports = checkout.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("crate::payments::service", ("Charge",), "PayCharge") in as_map

    def test_extracts_rust_grouped_use_items(self):
        source = dedent(
            """
            use crate::payments::{service::Charge as PayCharge, Refund};

            fn checkout(amount: i32) -> i32 {
                amount
            }
            """
        )
        records = extract_file("src/checkout.rs", source)
        checkout = next(r for r in records if r.name == "checkout")

        imports = checkout.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("crate::payments::service", ("Charge",), "PayCharge") in as_map
        assert ("crate::payments", ("Refund",), "") in as_map

    def test_extracts_kotlin_import_alias(self):
        source = dedent(
            """
            import com.example.payments.Charge as PayCharge

            class Checkout {
                fun run() {}
            }
            """
        )
        records = extract_file("src/Checkout.kt", source)
        owner = next(
            (r for r in records if r.name in {"run", "Checkout"}),
            records[0],
        )

        imports = owner.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("com.example.payments", ("Charge",), "PayCharge") in as_map

    def test_extracts_kotlin_wildcard_import(self):
        source = dedent(
            """
            import com.example.payments.*

            class Checkout {
                fun run() {}
            }
            """
        )
        records = extract_file("src/Checkout.kt", source)
        owner = next(
            (r for r in records if r.name in {"run", "Checkout"}),
            records[0],
        )

        imports = owner.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("com.example.payments", tuple(), "") in as_map

    def test_extracts_swift_typed_import(self):
        source = dedent(
            """
            import struct Foundation.Date

            struct Checkout {
                func run() {}
            }
            """
        )
        records = extract_file("src/Checkout.swift", source)
        owner = next(
            (r for r in records if r.name in {"run", "Checkout"}),
            records[0],
        )

        imports = owner.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("Foundation", ("Date",), "") in as_map

    def test_extracts_scala_import_selector_alias(self):
        source = dedent(
            """
            import payments.service.{Charge => PayCharge}

            class Checkout {
              def run(): Unit = {}
            }
            """
        )
        records = extract_file("src/Checkout.scala", source)
        owner = next(
            (r for r in records if r.name in {"run", "Checkout"}),
            records[0],
        )

        imports = owner.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("payments.service", ("Charge",), "PayCharge") in as_map

    def test_extracts_scala_wildcard_and_skips_hidden_selector(self):
        source = dedent(
            """
            import payments.service._
            import payments.model.{Charge => _, Refund => BillRefund}

            class Checkout {
              def run(): Unit = {}
            }
            """
        )
        records = extract_file("src/Checkout.scala", source)
        owner = next(
            (r for r in records if r.name in {"run", "Checkout"}),
            records[0],
        )

        imports = owner.properties.get("imports", [])
        as_map = {
            (item.get("module_path"), tuple(item.get("names", [])), item.get("alias"))
            for item in imports
            if isinstance(item, dict)
        }
        assert ("payments.service", tuple(), "") in as_map
        assert ("payments.model", ("Refund",), "BillRefund") in as_map
        assert ("payments.model", ("Charge",), "_") not in as_map


# ---------------------------------------------------------------------------
# Real .py file (checkpoint 5)
# ---------------------------------------------------------------------------


class TestExtractRealPyFile:
    """
    Use extractor.py itself as the test fixture — a non-trivial real
    Python file with classes, functions, and decorators.
    """

    _path = "src/indexer/extractor.py"
    _source: str | None = None

    @classmethod
    def get_source(cls) -> str:
        if cls._source is None:
            import pathlib

            here = pathlib.Path(__file__).parent.parent
            cls._source = (here / cls._path).read_text(encoding="utf-8")
        return cls._source

    def test_returns_non_empty(self):
        records = extract_file(self._path, self.get_source())
        assert len(records) > 0

    def test_contains_extract_file_function(self):
        records = extract_file(self._path, self.get_source())
        names = [r.name for r in records]
        assert "extract_file" in names

    def test_contains_extract_file_detailed_function(self):
        records = extract_file(self._path, self.get_source())
        names = [r.name for r in records]
        assert "extract_file_detailed" in names

    def test_all_records_have_correct_file_path(self):
        records = extract_file(self._path, self.get_source())
        for r in records:
            assert r.file_path == self._path

    def test_all_records_have_python_language(self):
        records = extract_file(self._path, self.get_source())
        for r in records:
            assert r.language == "python"

    def test_line_numbers_are_consistent(self):
        records = extract_file(self._path, self.get_source())
        for r in records:
            assert r.start_line >= 1
            assert r.end_line >= r.start_line

    def test_signature_and_source_both_contain_name(self):
        records = extract_file(self._path, self.get_source())
        for r in records:
            # The symbol name must appear in both the signature and the source.
            # (Signatures are whitespace-collapsed single lines, so they can't
            # be used as substrings of the multi-line source directly.)
            assert r.name in r.signature, f"{r.name!r} missing from signature"
            assert r.name in r.source, f"{r.name!r} missing from source"

    def test_qualified_name_is_empty(self):
        records = extract_file(self._path, self.get_source())
        for r in records:
            assert r.qualified_name == ""  # pipeline hasn't run yet


# ---------------------------------------------------------------------------
# Real .ts file (checkpoint 5)
# ---------------------------------------------------------------------------


class TestExtractRealTsFile:
    """
    Use treesitter.py's companion test fixtures — or a non-trivial
    inline TypeScript source — to validate the TS path end-to-end.
    """

    TS_PATH = "src/greeter.ts"
    TS_SOURCE = dedent(
        """
        interface Logger {
            log(message: string): void;
        }

        type UserId = string;

        async function fetchUser(id: UserId): Promise<string> {
            return `user-${id}`;
        }

        class UserService {
            private logger: Logger;

            constructor(logger: Logger) {
                this.logger = logger;
            }

            async getUser(id: UserId): Promise<string> {
                this.logger.log(`Fetching ${id}`);
                return fetchUser(id);
            }
        }

        export default UserService;
        """
    )

    def test_returns_non_empty(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        assert len(records) > 0

    def test_extracts_interface(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        labels = [r.label for r in records]
        assert "Interface" in labels or any(r.name == "Logger" for r in records)

    def test_extracts_async_function(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        names = [r.name for r in records]
        assert "fetchUser" in names

    def test_extracts_class(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        names = [r.name for r in records]
        assert "UserService" in names

    def test_extracts_method(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        names = [r.name for r in records]
        assert "getUser" in names

    def test_method_parent_is_class(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        method = next((r for r in records if r.name == "getUser"), None)
        assert method is not None
        assert method.parent == "UserService"

    def test_all_records_have_correct_file_path(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        for r in records:
            assert r.file_path == self.TS_PATH

    def test_all_records_have_typescript_language(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        for r in records:
            assert r.language == "typescript"

    def test_qualified_name_is_empty(self):
        records = extract_file(self.TS_PATH, self.TS_SOURCE)
        for r in records:
            assert r.qualified_name == ""


# ---------------------------------------------------------------------------
# extract_files() batch API
# ---------------------------------------------------------------------------


class TestExtractFiles:
    """extract_files() batch processing over a list of FileInfo objects."""

    def _make_fi(
        self, path: str, abs_path: str, language: str | None, size: int = 100
    ) -> FileInfo:
        return FileInfo(
            path=path, abs_path=abs_path, language=language, size_bytes=size
        )

    def test_returns_dict_keyed_by_path(self):
        fi = self._make_fi("src/a.py", "/repo/src/a.py", "python")
        source = "def f(): pass\n"
        result = extract_files([fi], read_file=lambda _: source)
        assert "src/a.py" in result
        assert isinstance(result["src/a.py"], ExtractionResult)

    def test_multiple_files(self):
        fi1 = self._make_fi("a.py", "/repo/a.py", "python")
        fi2 = self._make_fi("b.ts", "/repo/b.ts", "typescript")
        sources = {"/repo/a.py": "def a(): pass\n", "/repo/b.ts": "function b() {}\n"}
        result = extract_files([fi1, fi2], read_file=lambda p: sources[p])
        assert set(result.keys()) == {"a.py", "b.ts"}

    def test_read_error_produces_skip_result(self):
        fi = self._make_fi("broken.py", "/repo/broken.py", "python")

        def bad_read(_path: str) -> str:
            raise PermissionError

        result = extract_files([fi], read_file=bad_read)
        assert result["broken.py"].extractor == "skip"
        assert result["broken.py"].reason == "read_error"

    def test_oversized_file_skipped_without_read(self):
        fi = self._make_fi(
            "huge.py", "/repo/huge.py", "python", size=MAX_FILE_BYTES + 1
        )
        read_called = []

        def tracking_read(path: str) -> str:
            read_called.append(path)
            return ""

        result = extract_files([fi], read_file=tracking_read)
        assert read_called == []  # read_file never called
        assert result["huge.py"].extractor == "skip"
