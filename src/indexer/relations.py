"""
relations.py - Semantic relationship extraction from tree-sitter ASTs.

This module extracts source facts used by the resolution pass:
  - call sites inside callable definitions
  - imports/bindings at file scope and callable-local scope

Extraction is intentionally target-agnostic: it reports what appears in
source without trying to resolve calls to indexed symbols. Resolution and
confidence decisions happen later in registry.py / pipeline.py.
"""

import re
from dataclasses import dataclass

from tree_sitter import Node

REL_STATE_DEFS_AND_RELS = "definitions_and_relationships"
REL_STATE_DEFS_ONLY = "definitions_only"
REL_STATE_FALLBACK_ONLY = "fallback_only"


@dataclass(frozen=True)
class RelationshipCapability:
    """Per-language relationship extraction capability metadata."""

    language: str
    state: str


@dataclass
class RelationshipExtraction:
    """Relationship facts extracted for one callable scope."""

    calls: list[dict[str, object]]
    imports: list[dict[str, object]]
    unsupported_calls: int


_CALL_FIELDS: tuple[str, ...] = (
    "function",
    "callee",
    "name",
    "call",
    "value",
    "target",
)


# Tree-sitter call nodes vary heavily by grammar; this set captures the
# common invocation node names used by supported grammars.
_CALL_NODE_TYPES: frozenset[str] = frozenset(
    {
        "call",
        "call_expression",
        "function_call_expression",
        "invocation_expression",
        "method_invocation",
    }
)


# Import/binding nodes by language. A conservative map is used so callers can
# distinguish "no imports discovered" from "language extraction unavailable".
_IMPORT_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"import_statement", "import_from_statement"}),
    "javascript": frozenset({"import_statement"}),
    "typescript": frozenset({"import_statement"}),
    "go": frozenset({"import_spec", "import_declaration"}),
    "rust": frozenset({"use_declaration"}),
    "java": frozenset({"import_declaration"}),
    "c": frozenset(),
    "cpp": frozenset(),
    "ruby": frozenset(),
    "php": frozenset({"namespace_use_declaration"}),
    "c_sharp": frozenset({"using_directive"}),
    "bash": frozenset(),
    "kotlin": frozenset({"import", "import_header"}),
    "swift": frozenset({"import_declaration"}),
    "scala": frozenset({"import_declaration"}),
    "lua": frozenset(),
    "elixir": frozenset({"alias", "import", "require", "use"}),
}


def get_relationship_capability(language: str | None) -> RelationshipCapability:
    """Return the relationship extraction state for a language."""
    if not language:
        return RelationshipCapability(language="unknown", state=REL_STATE_FALLBACK_ONLY)
    if language in _IMPORT_NODE_TYPES:
        return RelationshipCapability(
            language=language,
            state=REL_STATE_DEFS_AND_RELS,
        )
    return RelationshipCapability(language=language, state=REL_STATE_DEFS_ONLY)


def extract_file_imports(
    root: Node,
    language: str,
    lines: list[str],
) -> list[dict[str, object]]:
    """Extract file-scope imports/bindings from a parsed file root."""
    imports: list[dict[str, object]] = []
    import_types = _IMPORT_NODE_TYPES.get(language, frozenset())
    if not import_types:
        return imports

    for child in root.children:
        if child.type not in import_types:
            continue
        parsed = _parse_import_node(child, language, lines, scope="file")
        imports.extend(parsed)
    return imports


def extract_relationships_for_symbol(
    symbol_node: Node,
    language: str,
    lines: list[str],
    definition_types: set[str],
) -> RelationshipExtraction:
    """Extract calls/imports from one callable symbol scope.

    Nested callable definitions are not traversed, so calls in nested
    functions/methods are not attributed to the enclosing callable.
    """
    calls: list[dict[str, object]] = []
    imports: list[dict[str, object]] = []
    unsupported_calls = 0

    import_types = _IMPORT_NODE_TYPES.get(language, frozenset())

    def walk(node: Node) -> None:
        nonlocal unsupported_calls

        if node is not symbol_node and node.type in definition_types:
            return

        if node.type in import_types:
            parsed_imports = _parse_import_node(node, language, lines, scope="local")
            imports.extend(parsed_imports)

        if _is_call_node(node):
            parsed_call = _parse_call_node(node, lines)
            if parsed_call is None:
                unsupported_calls += 1
            else:
                calls.append(parsed_call)

        for child in node.children:
            walk(child)

    walk(symbol_node)
    return RelationshipExtraction(
        calls=calls,
        imports=imports,
        unsupported_calls=unsupported_calls,
    )


def _is_call_node(node: Node) -> bool:
    """Heuristic check for call/invocation nodes across grammars."""
    if node.type in _CALL_NODE_TYPES:
        return True
    return node.type.endswith("_call") or node.type.endswith("_invocation")


def _parse_call_node(node: Node, lines: list[str]) -> dict[str, object] | None:
    """Parse a tree-sitter call node into a call-site payload."""
    target = _find_call_target_node(node)
    if target is None:
        return None

    callee = _node_text(target).strip()
    if not callee or "(" in callee or "\n" in callee:
        return None

    return {
        "callee": callee,
        "line": node.start_point[0] + 1,
        "qualifier": callee.rpartition(".")[0],
        "source_line": _safe_line(lines, node.start_point[0]),
    }


def _find_call_target_node(node: Node) -> Node | None:
    for field in _CALL_FIELDS:
        field_node = node.child_by_field_name(field)
        if field_node is not None:
            return field_node

    # Fallback for grammars without named call fields: choose the first
    # named child that looks like a callee expression.
    for child in node.children:
        if child.is_named and child.type not in {"arguments", "argument_list"}:
            return child
    return None


def _parse_import_node(
    node: Node,
    language: str,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    """Parse one import/binding node into a normalized payload."""
    if language == "python":
        return _parse_python_import(node, lines, scope)
    if language in ("javascript", "typescript"):
        return _parse_js_ts_import(node, lines, scope)
    if language == "go":
        return _parse_go_import(node, lines, scope)
    if language == "rust":
        return _parse_rust_import(node, lines, scope)
    if language == "java":
        return _parse_java_import(node, lines, scope)
    if language == "c_sharp":
        return _parse_csharp_import(node, lines, scope)
    if language == "kotlin":
        return _parse_kotlin_import(node, lines, scope)
    if language == "swift":
        return _parse_swift_import(node, lines, scope)
    if language == "scala":
        return _parse_scala_import(node, lines, scope)

    raw = _node_text(node).strip()
    if not raw:
        return []

    module_path = _guess_module_path(raw)
    if not module_path:
        return []

    alias = ""
    as_name = node.child_by_field_name("alias")
    if as_name is not None:
        alias = _node_text(as_name).strip()

    return [
        {
            "module_path": module_path,
            "names": [],
            "alias": alias,
            "line": node.start_point[0] + 1,
            "source_line": _safe_line(lines, node.start_point[0]),
            "scope": scope,
        }
    ]


def _parse_python_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip()
    if not raw:
        return []

    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    if raw.startswith("import "):
        payloads = _parse_python_import_statement(raw)
        if not payloads:
            return []
        return [
            {
                **payload,
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
            for payload in payloads
        ]

    if raw.startswith("from "):
        payload = _parse_python_from_import(raw)
        if payload is None:
            return []
        return [
            {
                **payload,
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        ]

    return []


def _parse_python_import_statement(raw: str) -> list[dict[str, object]]:
    # import a, b as c
    tail = raw.removeprefix("import ").strip()
    parts = [part.strip() for part in tail.split(",") if part.strip()]
    parsed: list[dict[str, object]] = []
    for part in parts:
        if " as " in part:
            module_path, alias = [p.strip() for p in part.split(" as ", 1)]
        else:
            module_path, alias = part, ""
        if not module_path:
            continue
        parsed.append(
            {
                "module_path": module_path,
                "names": [],
                "alias": alias,
            }
        )
    return parsed


def _parse_python_from_import(raw: str) -> dict[str, object] | None:
    # from pkg.sub import x, y as z
    try:
        head, tail = raw.split(" import ", 1)
    except ValueError:
        return None

    module_path = head.removeprefix("from ").strip()
    if not module_path:
        return None

    names: list[str] = []
    alias = ""
    for part in [piece.strip() for piece in tail.split(",") if piece.strip()]:
        if " as " in part:
            name, maybe_alias = [p.strip() for p in part.split(" as ", 1)]
            if name:
                names.append(name)
            if maybe_alias and not alias:
                alias = maybe_alias
        else:
            names.append(part)

    return {
        "module_path": module_path,
        "names": names,
        "alias": alias,
    }


def _guess_module_path(raw: str) -> str:
    # Best-effort extraction for non-python grammars.
    for token in ("import", "using", "use", "require", "alias"):
        if raw.startswith(token + " "):
            remainder = raw[len(token) + 1 :].strip().rstrip(";")
            if remainder:
                return remainder.split()[0].strip("\"'")
    return ""


def _parse_js_ts_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    """Parse JS/TS import statements into normalized import payloads."""
    raw = _node_text(node).strip().rstrip(";")
    if not raw.startswith("import"):
        return []

    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    # Side-effect import: import "module"
    side_effect_match = re.match(r"^import\s+[\"']([^\"']+)[\"']$", raw)
    if side_effect_match:
        return [
            {
                "module_path": side_effect_match.group(1),
                "names": [],
                "alias": "",
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        ]

    match = re.match(r"^import\s+(.+?)\s+from\s+[\"']([^\"']+)[\"']$", raw)
    if not match:
        return []

    spec = match.group(1).strip()
    module_path = match.group(2).strip()
    if not module_path:
        return []

    payloads: list[dict[str, object]] = []

    # Namespace import: import * as svc from "module"
    ns_match = re.match(r"^\*\s+as\s+([A-Za-z_$][\w$]*)$", spec)
    if ns_match:
        payloads.append(
            {
                "module_path": module_path,
                "names": [],
                "alias": ns_match.group(1),
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        )
        return payloads

    # Split default and named parts in: defaultName, { a as b }
    default_part = ""
    named_part = ""
    if "{" in spec and "}" in spec:
        brace_start = spec.find("{")
        brace_end = spec.rfind("}")
        default_part = spec[:brace_start].strip().rstrip(",").strip()
        named_part = spec[brace_start + 1 : brace_end].strip()
    else:
        default_part = spec.strip()

    # Default import alias: import foo from "module"
    if default_part and default_part != "*":
        payloads.append(
            {
                "module_path": module_path,
                "names": [],
                "alias": default_part,
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        )

    # Named imports: import { charge, refund as doRefund } from "module"
    if named_part:
        for item in [piece.strip() for piece in named_part.split(",") if piece.strip()]:
            if " as " in item:
                name, alias = [p.strip() for p in item.split(" as ", 1)]
            else:
                name, alias = item, ""
            if not name:
                continue
            payloads.append(
                {
                    "module_path": module_path,
                    "names": [name],
                    "alias": alias,
                    "line": line,
                    "source_line": source_line,
                    "scope": scope,
                }
            )

    return payloads


def _parse_go_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip().rstrip(";")
    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    # import_spec commonly looks like:
    #   "fmt"
    #   alias "my/module"
    #   _ "my/module"
    #   . "my/module"
    # import_declaration can also appear as:
    #   import alias "my/module"
    #   import (
    #       "fmt"
    #       svc "my/module"
    #   )
    spec_matches = re.findall(
        r'(?:^|\n)\s*(?:(?:import)\s+)?(?:(\w+|\.|_)\s+)?"([^"]+)"',
        raw,
    )
    if not spec_matches:
        return []

    payloads: list[dict[str, object]] = []
    for raw_alias, raw_module in spec_matches:
        module_path = raw_module.strip()
        if not module_path:
            continue
        alias = (
            raw_alias.strip()
            if raw_alias and raw_alias.strip() not in {".", "_"}
            else ""
        )
        name = module_path.rsplit("/", 1)[-1] if "/" in module_path else module_path
        payloads.append(
            {
                "module_path": module_path,
                "names": [name] if name else [],
                "alias": alias,
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        )

    return payloads


def _parse_rust_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip().rstrip(";")
    if not raw.startswith("use "):
        return []

    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])
    body = raw.removeprefix("use ").strip()

    payloads: list[dict[str, object]] = []
    if "{" in body and "}" in body:
        prefix, _, tail = body.partition("{")
        items = tail.rsplit("}", 1)[0]
        module_prefix = prefix.rstrip(":").strip()
        for item in [piece.strip() for piece in items.split(",") if piece.strip()]:
            if item in {"*", "self"}:
                continue
            name, alias = _split_alias(item, " as ")
            if name.startswith("self::"):
                name = name.removeprefix("self::")
            if "::" in name:
                item_module, item_names = _split_parent_and_name(name, sep="::")
                module_path = (
                    f"{module_prefix}::{item_module}"
                    if module_prefix and item_module
                    else module_prefix or item_module
                )
                names = item_names
            else:
                module_path = module_prefix
                names = [name] if name else []
            payloads.append(
                {
                    "module_path": module_path,
                    "names": names,
                    "alias": alias,
                    "line": line,
                    "source_line": source_line,
                    "scope": scope,
                }
            )
        return payloads

    name_path, alias = _split_alias(body, " as ")
    module_path, names = _split_parent_and_name(name_path, sep="::")
    return [
        {
            "module_path": module_path,
            "names": names,
            "alias": alias,
            "line": line,
            "source_line": source_line,
            "scope": scope,
        }
    ]


def _parse_java_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip().rstrip(";")
    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    # import java.util.List
    # import static java.util.Collections.emptyList
    match = re.match(r"^import\s+(static\s+)?([\w.]+(?:\.\*)?)$", raw)
    if not match:
        return []

    static_prefix = bool(match.group(1))
    path = match.group(2)
    if path.endswith(".*"):
        module_path = path[:-2]
        names: list[str] = []
    else:
        module_path, names = _split_parent_and_name(path)
        if static_prefix and names:
            # Keep static member as imported symbol for import_map.
            pass

    return [
        {
            "module_path": module_path,
            "names": names,
            "alias": "",
            "line": line,
            "source_line": source_line,
            "scope": scope,
        }
    ]


def _parse_csharp_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip().rstrip(";")
    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    if not raw.startswith("using "):
        return []

    body = raw.removeprefix("using ").strip()

    # Alias using: using SB = System.Text.StringBuilder
    alias_match = re.match(r"^([A-Za-z_][\w]*)\s*=\s*([\w.]+)$", body)
    if alias_match:
        alias = alias_match.group(1)
        target = alias_match.group(2)
        module_path, names = _split_parent_and_name(target)
        return [
            {
                "module_path": module_path,
                "names": names,
                "alias": alias,
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        ]

    # Static import: using static System.Math
    static_match = re.match(r"^static\s+([\w.]+)$", body)
    if static_match:
        target = static_match.group(1)
        module_path, names = _split_parent_and_name(target)
        return [
            {
                "module_path": module_path,
                "names": names,
                "alias": "",
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        ]

    # Namespace using: using Foo.Bar
    ns_match = re.match(r"^([\w.]+)$", body)
    if not ns_match:
        return []
    target = ns_match.group(1)
    module_path, names = _split_parent_and_name(target)
    return [
        {
            "module_path": module_path,
            "names": names,
            "alias": "",
            "line": line,
            "source_line": source_line,
            "scope": scope,
        }
    ]


def _parse_kotlin_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip().rstrip(";")
    if not raw.startswith("import "):
        return []
    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    body = raw.removeprefix("import ").strip()
    if body.endswith(".*"):
        module_path = body[:-2]
        names: list[str] = []
        alias = ""
    else:
        name_path, alias = _split_alias(body, " as ")
        module_path, names = _split_parent_and_name(name_path)
    return [
        {
            "module_path": module_path,
            "names": names,
            "alias": alias,
            "line": line,
            "source_line": source_line,
            "scope": scope,
        }
    ]


def _parse_swift_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip().rstrip(";")
    if not raw.startswith("import "):
        return []
    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    body = raw.removeprefix("import ").strip()
    for kind in ("struct", "class", "enum", "protocol", "func", "var", "let"):
        prefix = kind + " "
        if body.startswith(prefix):
            body = body[len(prefix) :].strip()
            break

    module_path, names = _split_parent_and_name(body)
    return [
        {
            "module_path": module_path,
            "names": names,
            "alias": "",
            "line": line,
            "source_line": source_line,
            "scope": scope,
        }
    ]


def _parse_scala_import(
    node: Node,
    lines: list[str],
    scope: str,
) -> list[dict[str, object]]:
    raw = _node_text(node).strip().rstrip(";")
    if not raw.startswith("import "):
        return []
    line = node.start_point[0] + 1
    source_line = _safe_line(lines, node.start_point[0])

    body = raw.removeprefix("import ").strip()
    payloads: list[dict[str, object]] = []

    if body.endswith("._"):
        return [
            {
                "module_path": body[:-2],
                "names": [],
                "alias": "",
                "line": line,
                "source_line": source_line,
                "scope": scope,
            }
        ]

    if "{" in body and "}" in body:
        prefix, _, tail = body.partition("{")
        items = tail.rsplit("}", 1)[0]
        module_prefix = prefix.rstrip(".").strip()
        for item in [piece.strip() for piece in items.split(",") if piece.strip()]:
            if item in {"_"}:
                continue
            name, alias = _split_alias(item, "=>")
            if alias == "_":
                continue
            payloads.append(
                {
                    "module_path": module_prefix,
                    "names": [name] if name else [],
                    "alias": alias,
                    "line": line,
                    "source_line": source_line,
                    "scope": scope,
                }
            )
        return payloads

    module_path, names = _split_parent_and_name(body)
    if names == ["_"]:
        names = []
    payloads.append(
        {
            "module_path": module_path,
            "names": names,
            "alias": "",
            "line": line,
            "source_line": source_line,
            "scope": scope,
        }
    )
    return payloads


def _split_parent_and_name(path: str, sep: str = ".") -> tuple[str, list[str]]:
    if sep not in path:
        return path, []
    parent, _, leaf = path.rpartition(sep)
    if not parent or not leaf:
        return path, []
    return parent, [leaf]


def _split_alias(value: str, token: str) -> tuple[str, str]:
    if token in value:
        name, alias = [part.strip() for part in value.split(token, 1)]
        return name, alias
    return value.strip(), ""


def _node_text(node: Node) -> str:
    if node.text:
        return node.text.decode("utf-8", errors="replace")
    return ""


def _safe_line(lines: list[str], index: int) -> str:
    if 0 <= index < len(lines):
        return lines[index]
    return ""
