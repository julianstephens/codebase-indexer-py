"""
treesitter.py — Universal AST extractor built on tree-sitter.

Converts source files into lists of NodeRecord objects — one per
extractable symbol (Function, Class, Method, Interface, Type) — each
carrying the full source of its body and a signature line suitable for
the skeleton renderer.

Public API:
    extract(path, source)  →  list[NodeRecord]
        Main entry point. Routes to the correct language config and
        walks the AST. Returns an empty list (not an error) for files
        with no extractable definitions.

Internal structure:
    _walk()                 Recursive AST walker. Matches node types
                            against LANG_CONFIG definitions, delegates
                            name extraction and signature slicing, then
                            recurses into child nodes with updated parent.

    _extract_name()         Reads a symbol name from a tree-sitter node
                            using the configured name_field, or falls
                            back to _CUSTOM_NAME_EXTRACTORS for cases
                            where name_field=None.

    _extract_signature()    Slices source lines from the definition start
                            up to (but not including) the body opener.
                            Language-aware: knows what node types open a
                            body in each language family.

    _extract_source()       Slices source lines for the full node extent
                            (start_line to end_line inclusive).

    _find_body_start()      Locates the line where the body begins by
                            scanning child node types for known body
                            openers (block, statement_block, class_body,
                            compound_statement, etc.).

Custom name extractors (_CUSTOM_NAME_EXTRACTORS):
    Some node types cannot have their name read from a single named
    field. Examples:
      - Python decorated_definition: unwrap to inner function/class
      - JS/TS arrow_function: look at parent assignment for name
      - Go method_declaration: receiver type becomes the parent
      - Rust impl_item: read "type" field for the implemented type
      - C function_definition: declarator is nested, walk to identifier
    Each entry is a callable (node, source_lines) → str | None.

Thread safety:
    _get_parser() creates a new Parser instance on every call. The
    pipeline creates one extractor call per file in a thread pool —
    each call creates its own parser instance, so Parser objects are
    never shared across threads.
"""

import importlib
from dataclasses import dataclass, field
from typing import Callable

from tree_sitter import Language, Node, Parser

from .relations import extract_file_imports, extract_relationships_for_symbol

# ---------------------------------------------------------------------------
# NodeRecord
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    """
    A single extractable symbol extracted from a source file.

    All fields are populated by _walk() and its helpers. The pipeline
    consumes NodeRecord objects directly — store.py maps them to the
    nodes table, registry.py indexes them by qualified_name.

    Attributes:
        label:          Graph node label. One of: Function | Class |
                        Method | Interface | Type | File.
        name:           Short symbol name, e.g. "charge".
        qualified_name: Fully qualified address, e.g.
                        "src.payments.service.charge". Set by the caller
                        (pipeline.py) after extraction using fqn.compute().
                        Empty string until set.
        file_path:      Repo-relative file path, e.g.
                        "src/payments/service.py".
        start_line:     1-based line number where the definition begins
                        (the decorator line for decorated definitions).
        end_line:       1-based line number of the closing delimiter.
        signature:      Everything from start_line up to (but not
                        including) the body opener, joined to a single
                        line. E.g.:
                          "def charge(user: User, amount_cents: int) -> Payment:"
        source:         Full source text of the node from start_line to
                        end_line inclusive, preserving original indentation.
        parent:         Short name of the enclosing class or struct, or
                        empty string for top-level definitions. Used by
                        fqn.compute() and by the skeleton renderer to
                        group methods under their class.
        language:       Canonical language name, e.g. "python". Copied
                        from the per-file detection result.
        properties:     Dict of language-specific extras. Populated by
                        _collect_properties(). Examples:
                          {"async": True}
                          {"decorators": ["@login_required", "@csrf_exempt"]}
                          {"visibility": "public", "return_type": "Payment"}
                          {"receiver": "*PaymentService"}   # Go methods
    """

    label: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    source: str
    language: str
    qualified_name: str = ""
    parent: str = ""
    properties: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Body-opener node types
# ---------------------------------------------------------------------------
#
# _find_body_start() scans a definition node's children for the first
# child whose type is in this set. The body opener's start line is where
# the signature ends and the body begins.
#
# These are tree-sitter internal node type names, not language keywords.

_BODY_OPENERS: frozenset[str] = frozenset(
    {
        # Python
        "block",
        # JavaScript / TypeScript
        "statement_block",
        # Java, Kotlin, Swift, Scala, PHP, C#
        "class_body",
        "interface_body",
        "enum_body",
        "declaration_list",
        # C / C++
        "compound_statement",
        "field_declaration_list",
        # Rust
        "enum_variant_list",
        # Ruby
        "body_statement",
        # Go
        # Elixir
        "do_block",
    }
)

# ---------------------------------------------------------------------------
# Wrapper node types
# ---------------------------------------------------------------------------
#
# These node types wrap an inner function or class definition. When _walk
# matches one of these:
#   1. The label is overridden from the inner node's entry in defs (so
#      a decorated class gets label "Class", not the wrapper's "Function").
#   2. _walk does NOT recurse into the wrapper's children — the wrapper
#      already spans the full inner definition, recursing would create a
#      duplicate record for the inner node type.

_WRAPPER_NODE_TYPES: frozenset[str] = frozenset(
    {
        "decorated_definition",  # Python: @decorator def/class
        "export_statement",  # JS/TS: export function/class
        "template_declaration",  # C++: template<T> function/class
    }
)

# ---------------------------------------------------------------------------
# Custom name extractors
# ---------------------------------------------------------------------------
#
# Registered for (language, node_type) pairs where name_field=None in
# LANG_CONFIG, meaning the name cannot be read from a single named field.
#
# Signature: (node: Node, lines: list[str]) -> str | None
#   Return the short symbol name, or None to skip this node entirely.
#
# Keys are (language, tree_sitter_node_type) tuples.

_CustomExtractor = Callable[[Node, list[str]], str | None]
_CUSTOM_NAME_EXTRACTORS: dict[tuple[str, str], _CustomExtractor] = {}


def _register_extractor(language: str, node_type: str) -> Callable:
    """
    Decorator to register a custom name extractor for (language, node_type).

    Usage:
        @_register_extractor("python", "decorated_definition")
        def _extract_decorated(node, lines):
            ...
    """

    def decorator(fn: _CustomExtractor) -> _CustomExtractor:
        _CUSTOM_NAME_EXTRACTORS[(language, node_type)] = fn
        return fn

    return decorator


# ── Python: decorated_definition ────────────────────────────────────────────
# A decorated_definition node wraps a function or class with decorators.
# The actual name lives on the inner function_definition or class_definition.


@_register_extractor("python", "decorated_definition")
def _extract_python_decorated(node: Node, lines: list[str]) -> str | None:
    """
    Unwrap a decorated_definition and return the inner symbol's name.

    Args:
        node:  the decorated_definition Node
        lines: source file lines (unused here but required by the signature)

    Returns:
        The name of the inner function or class, or None if not found.
    """
    for child in node.children:
        if child.type in ("function_definition", "class_definition"):
            name_node = child.child_by_field_name("name")
            if name_node:
                return _node_text(name_node) or None
    return None


# ── JavaScript / TypeScript: arrow_function ──────────────────────────────────
# Arrow functions are usually anonymous. The name (if any) comes from
# the variable declaration that holds the arrow function:
#   const myFunc = (x) => x + 1
# The parent of the arrow_function node is a variable_declarator whose
# "name" child is the identifier.


@_register_extractor("javascript", "arrow_function")
@_register_extractor("typescript", "arrow_function")
def _extract_arrow_function(node: Node, lines: list[str]) -> str | None:
    """
    Extract the name of an arrow function from its parent assignment.

    Walks up to the parent variable_declarator or assignment_expression
    and reads the left-hand side identifier.

    Args:
        node:  the arrow_function Node
        lines: source file lines (unused)

    Returns:
        The assigned variable name, e.g. "myFunc", or None if the
        arrow function is not assigned to a named variable.
    """
    parent = node.parent
    if not parent:
        return None

    # Variable declaration: const myFunc = (x) => x + 1
    if parent.type == "variable_declarator":
        name_node = parent.child_by_field_name("name")
        if name_node and name_node.type == "identifier":
            return _node_text(name_node) or None

    # Assignment expression: myFunc = (x) => x + 1
    if parent.type == "assignment_expression":
        left_node = parent.child_by_field_name("left")
        if left_node and left_node.type == "identifier":
            return _node_text(left_node) or None

    return None


# ── JavaScript / TypeScript: export_statement ────────────────────────────────
# export_statement wraps a declaration. The name is on the inner node.
# e.g.: export function foo() {} → inner is function_declaration, name="foo"
#       export class Bar {}      → inner is class_declaration, name="Bar"
#       export default function  → may be anonymous, return None


@_register_extractor("javascript", "export_statement")
@_register_extractor("typescript", "export_statement")
def _extract_export_statement(node: Node, lines: list[str]) -> str | None:
    """
    Unwrap an export_statement and return the inner declaration's name.

    Args:
        node:  the export_statement Node
        lines: source file lines (unused)

    Returns:
        Name of the exported symbol, or None for anonymous/default exports.
    """
    for child in node.children:
        if child.type in (
            "function_declaration",
            "class_declaration",
            "variable_declaration",
        ):
            name_node = child.child_by_field_name("name")
            if name_node:
                return _node_text(name_node) or None
    return None


# ── Go: method_declaration ──────────────────────────────────────────────────
# Go methods have a receiver: func (s *Service) MethodName(...) ...
# The receiver type (Service) becomes the parent field on the NodeRecord.
# The method name is on the "name" field directly.
# This extractor is registered to also populate the receiver property
# during _collect_properties(), not to provide the name — name_field="name"
# works. Registered here as a no-op name extractor so _collect_properties
# can detect that receiver handling is needed.


@_register_extractor("go", "type_declaration")
def _extract_go_type_declaration(node: Node, lines: list[str]) -> str | None:
    """
    Extract the name from a Go type_declaration, which wraps a type_spec.

    A type_declaration node contains one or more type_spec children.
    The type_spec has a "name" field with the type identifier.

    Args:
        node:  the type_declaration Node
        lines: source file lines (unused)

    Returns:
        The type name from the first type_spec child, or None.
    """
    for child in node.children:
        if child.type == "type_spec":
            name_node = child.child_by_field_name("name")
            if name_node:
                return _node_text(name_node) or None
    return None


# ── Rust: impl_item ─────────────────────────────────────────────────────────
# impl blocks have no name of their own. The "type" field holds the
# type being implemented, e.g.:
#   impl PaymentService { ... }    →  name = "PaymentService"
#   impl Display for MyType { ... } → name = "MyType"


@_register_extractor("rust", "impl_item")
def _extract_rust_impl(node: Node, lines: list[str]) -> str | None:
    """
    Extract the type name from a Rust impl_item.

    Reads the "type" field. For trait impls (impl Trait for Type),
    returns the implementing type (the "for" target), not the trait.

    Args:
        node:  the impl_item Node
        lines: source file lines (unused)

    Returns:
        The implemented type name, e.g. "PaymentService", or None.
    """
    type_node = node.child_by_field_name("type")
    if type_node:
        return _node_text(type_node) or None
    return None


# ── C: function_definition ──────────────────────────────────────────────────
# C function definitions have a nested declarator tree:
#   function_definition
#     type: ...
#     declarator: pointer_declarator | function_declarator | identifier
#       declarator: function_declarator
#         declarator: identifier  ← the name is here
# Walk the declarator chain to find the innermost identifier.


@_register_extractor("c", "function_definition")
@_register_extractor("cpp", "function_definition")
def _extract_c_function(node: Node, lines: list[str]) -> str | None:
    """
    Walk the declarator chain of a C/C++ function_definition to find
    the function name identifier.

    Args:
        node:  the function_definition Node
        lines: source file lines (unused)

    Returns:
        The function name, e.g. "process_payment", or None if the
        declarator structure is not recognised.
    """
    declarator_node = node.child_by_field_name("declarator")
    while declarator_node:
        if declarator_node.type == "identifier":
            return _node_text(declarator_node) or None
        declarator_node = declarator_node.child_by_field_name("declarator")
    return None


# ── C++: template_declaration ───────────────────────────────────────────────
# Template declarations wrap a function or class definition.
# The name is on the inner declaration.


@_register_extractor("cpp", "template_declaration")
def _extract_cpp_template(node: Node, lines: list[str]) -> str | None:
    """
    Unwrap a C++ template_declaration and return the inner symbol's name.

    Args:
        node:  the template_declaration Node
        lines: source file lines (unused)

    Returns:
        Name of the templated function or class, or None.
    """
    for child in node.children:
        if child.type in ("function_definition", "class_specifier"):
            name_node = child.child_by_field_name("name")
            if name_node:
                return _node_text(name_node) or None
    return None


# ── Lua: assignment_statement ────────────────────────────────────────────────
# Lua functions defined via table assignment:
#   MyModule.myFunc = function(...)
# The name is constructed from the left-hand side field expression.


@_register_extractor("lua", "assignment_statement")
def _extract_lua_assignment(node: Node, lines: list[str]) -> str | None:
    """
    Extract a function name from a Lua table-field assignment.

    Only matches assignments where the right-hand side is a function
    expression. Returns the left-hand side as the name (e.g.
    "MyModule.myFunc"), or None if the RHS is not a function.

    Args:
        node:  the assignment_statement Node
        lines: source file lines (unused)

    Returns:
        The left-hand side name string, or None.
    """
    # Check if the right-hand side is a function expression
    rhs_node = node.child_by_field_name("value")
    if not rhs_node or rhs_node.type != "function":
        return None

    # Get the left-hand side (the variable being assigned)
    lhs_node = node.child_by_field_name("variable")
    if not lhs_node:
        return None

    name_parts = []
    current_node = lhs_node
    while current_node:
        if current_node.type == "identifier":
            name_parts.append(_node_text(current_node))
        elif current_node.type == "field_expression":
            field_node = current_node.child_by_field_name("field")
            if field_node and field_node.type == "identifier":
                name_parts.append(_node_text(field_node))
        current_node = current_node.child_by_field_name("table")

    return ".".join(reversed(name_parts)) if name_parts else None


# ── Kotlin: secondary_constructor ────────────────────────────────────────────


@_register_extractor("kotlin", "secondary_constructor")
def _extract_kotlin_secondary_constructor(node: Node, lines: list[str]) -> str | None:
    """
    Return a synthetic name for a Kotlin secondary constructor.

    Kotlin secondary constructors have no name field. Returns the
    string "constructor" so the NodeRecord has a non-empty name.

    Args:
        node:  the secondary_constructor Node
        lines: source file lines (unused)

    Returns:
        The string "constructor".
    """
    return "constructor"


# ── Swift: init_declaration ──────────────────────────────────────────────────


@_register_extractor("swift", "init_declaration")
def _extract_swift_init(node: Node, lines: list[str]) -> str | None:
    """
    Return a synthetic name for a Swift init declaration.

    Swift initializers have no name field. Returns "init" so the
    NodeRecord has a non-empty name.

    Args:
        node:  the init_declaration Node
        lines: source file lines (unused)

    Returns:
        The string "init".
    """
    return "init"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(path: str, source: str) -> list[NodeRecord]:
    """
    Extract all symbol definitions from a source file.

    Parses the source using the tree-sitter grammar for the detected
    language, walks the AST, and returns one NodeRecord per extractable
    definition (Function, Class, Method, Interface, Type).

    The qualified_name field on each NodeRecord is left empty (""). The
    caller (pipeline.py) is responsible for setting it via fqn.compute()
    once the project name is known.

    Returns an empty list (not an error) when:
      - The file extension is not in EXTENSION_TO_LANG.
      - The language has no LANG_CONFIG entry.
      - The source is empty.
      - The AST root has no children matching the definition table.

    Does NOT raise on parse errors — tree-sitter produces a partial AST
    for syntactically invalid files. Records extracted from error nodes
    may have incorrect line ranges; the pipeline filters these via
    start_line > 0 and end_line >= start_line.

    Args:
        path:   repo-relative file path. Used for language detection and
                stored on each NodeRecord as file_path.
        source: full UTF-8 source text of the file.

    Returns:
        List of NodeRecord objects in source order (by start_line).

    Examples:
        >>> records = extract("src/payments/service.py", source)
        >>> records[0].name
        'charge'
        >>> records[0].label
        'Function'
        >>> records[0].signature
        'def charge(user: User, amount_cents: int, currency: str) -> Payment:'
        >>> records[0].source[:3]
        'def'
    """
    from .languages import EXTENSION_TO_LANG, LANG_CONFIG

    # Detect language from file extension
    ext = "." + path.rsplit(".", 1)[-1].lower()
    language = EXTENSION_TO_LANG.get(ext)
    if not language:
        return []

    # Get the language config
    lang_config = LANG_CONFIG.get(language)
    if not lang_config:
        return []

    # Skip empty source files
    if not source.strip():
        return []

    # Split source into lines (0-indexed)
    lines = source.splitlines()

    # Create a parser for this language
    parser_name = lang_config["parser"]
    parser = _get_parser(parser_name)

    # Parse the source into a tree-sitter AST
    tree = parser.parse(bytes(source, "utf8"))
    root_node = tree.root_node

    # Walk the AST and collect NodeRecords
    records: list[NodeRecord] = []
    definition_types = set(lang_config["definitions"].keys())
    file_imports = extract_file_imports(root_node, language, lines)
    _walk(
        node=root_node,
        path=path,
        language=language,
        lines=lines,
        defs=lang_config["definitions"],
        definition_types=definition_types,
        file_imports=file_imports,
        records=records,
        parent="",
    )

    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _walk(
    node: Node,
    path: str,
    language: str,
    lines: list[str],
    defs: dict[str, tuple[str, str | None]],
    definition_types: set[str],
    file_imports: list[dict[str, object]],
    records: list[NodeRecord],
    parent: str,
) -> None:
    """
    Recursively walk a tree-sitter AST node and collect NodeRecords.

    Matches each node's type against `defs` (from LANG_CONFIG). On a
    match:
      1. Extracts the symbol name via _extract_name().
      2. Skips the node if the name is None (anonymous / unwanted).
      3. Computes start_line, end_line, signature, source.
      4. Appends a NodeRecord with parent set to the current class context.
      5. Recurses into children with parent=name for class-body nodes,
         or parent=parent for all other nodes.

    For nodes that are "wrappers" (decorated_definition, export_statement,
    template_declaration), the custom extractor may return a name but the
    node's own body is on a child — _walk recurses with the same parent so
    the inner definition is also captured.

    Args:
        node:     current tree-sitter Node being visited
        path:     repo-relative file path (stored on each NodeRecord)
        language: canonical language name (e.g. "python")
        lines:    source split by line (0-indexed)
        defs:     definition table from LANG_CONFIG[language]["definitions"]
        records:  accumulator list — NodeRecords are appended here
        parent:   short name of the enclosing class, or "" for top-level
    """
    for child in node.children:
        if child.type in defs:
            label, name_field = defs[child.type]
            name = _extract_name(child, language, label, name_field, lines)
            if name is None:
                continue  # Skip anonymous or unrecognized nodes

            # Wrapper nodes: override label from the inner definition node
            if child.type in _WRAPPER_NODE_TYPES:
                for inner in child.children:
                    if inner.type in defs:
                        label = defs[inner.type][0]
                        break

            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1
            signature = _extract_signature(child, lines)
            source = _extract_source(child, lines)
            properties = _collect_properties(child, language, lines)
            rel = extract_relationships_for_symbol(
                child,
                language,
                lines,
                definition_types,
            )
            if rel.calls:
                properties["calls"] = rel.calls
            merged_imports = list(file_imports)
            if rel.imports:
                merged_imports.extend(rel.imports)
            if merged_imports:
                properties["imports"] = merged_imports
            if rel.unsupported_calls:
                properties["unsupported_calls"] = rel.unsupported_calls

            # Go method_declaration: derive parent from receiver type
            effective_parent = parent
            if language == "go" and child.type == "method_declaration":
                recv = child.child_by_field_name("receiver")
                if recv:
                    for param in recv.children:
                        if param.type == "parameter_declaration":
                            type_node = param.child_by_field_name("type")
                            if type_node:
                                if type_node.type == "pointer_type":
                                    for tc in type_node.children:
                                        if tc.type == "type_identifier":
                                            effective_parent = _node_text(tc)
                                            break
                                elif type_node.type == "type_identifier":
                                    effective_parent = _node_text(type_node)
                            break

            record = NodeRecord(
                label=label,
                name=name,
                file_path=path,
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                source=source,
                parent=effective_parent,
                language=language,
                properties=properties,
            )
            records.append(record)

            # Wrapper nodes (decorated_definition, export_statement, etc.) already
            # span the full inner definition — recurse into the inner node's children
            # rather than the wrapper itself to avoid duplicating the inner record.
            if child.type in _WRAPPER_NODE_TYPES:
                new_parent = name if label in ("Class", "Interface", "Type") else parent
                for inner in child.children:
                    if inner.type in defs:
                        _walk(
                            inner,
                            path,
                            language,
                            lines,
                            defs,
                            definition_types,
                            file_imports,
                            records,
                            parent=new_parent,
                        )
                        break
            # Recurse into children with updated parent for class-like nodes
            elif label in ("Class", "Interface", "Type"):
                _walk(
                    child,
                    path,
                    language,
                    lines,
                    defs,
                    definition_types,
                    file_imports,
                    records,
                    parent=name,
                )
            else:
                _walk(
                    child,
                    path,
                    language,
                    lines,
                    defs,
                    definition_types,
                    file_imports,
                    records,
                    parent=parent,
                )
        else:
            # Recurse into non-definition nodes without changing parent
            _walk(
                child,
                path,
                language,
                lines,
                defs,
                definition_types,
                file_imports,
                records,
                parent=parent,
            )


def _extract_name(
    node: Node,
    language: str,
    label: str,
    name_field: str | None,
    lines: list[str],
) -> str | None:
    """
    Extract the short symbol name from a definition node.

    Resolution order:
      1. If name_field is not None, call node.child_by_field_name(name_field)
         and return its text.
      2. If name_field is None, look up (language, node.type) in
         _CUSTOM_NAME_EXTRACTORS and call the registered function.
      3. If no custom extractor is registered, return None (node is skipped).

    The returned name is decoded from bytes to str using UTF-8 with
    'replace' error handling to tolerate non-UTF-8 source files.

    Args:
        node:       the definition node (e.g. function_definition)
        language:   canonical language name
        label:      the graph label (unused here, passed for context)
        name_field: field name string or None (from LANG_CONFIG)
        lines:      source file lines (passed to custom extractors)

    Returns:
        The symbol name string, or None if extraction failed / node
        should be skipped.
    """
    if name_field is not None:
        name_node = node.child_by_field_name(name_field)
        if name_node:
            return _node_text(name_node)
        return None

    # Custom extractor path
    extractor_key = (language, node.type)
    extractor_fn = _CUSTOM_NAME_EXTRACTORS.get(extractor_key)
    if extractor_fn:
        return extractor_fn(node, lines)

    return None  # No name_field and no custom extractor


def _extract_signature(
    node: Node,
    lines: list[str],
) -> str:
    """
    Extract the signature of a definition — everything from the start of
    the node up to (but not including) the body opener.

    Finds the body opener line via _find_body_start(). If no body opener
    is found (e.g. a forward declaration or interface method), the entire
    node text is the signature.

    Multi-line signatures (e.g. functions with many parameters spread
    across lines) are joined with a single space and stripped of excess
    whitespace to produce a single readable line.

    Args:
        node:  the definition node
        lines: source file lines (0-indexed)

    Returns:
        A single-line string. Never empty — falls back to the first
        source line of the node.

    Examples:
        A Python function:
            "def charge(user: User, amount_cents: int, currency: str) -> Payment:"
        A Go method:
            "func (s *PaymentService) Charge(user User, cents int) error"
        A TypeScript class:
            "export class UserViewSet extends BaseViewSet"
    """
    if node.start_point[0] >= len(lines):
        return ""
    body_start = _find_body_start(node)
    if body_start is None:
        end = node.end_point[0] + 1
    elif body_start == node.start_point[0]:
        # Body opener is on the same line as the definition (Go, Java, TS, Rust, C)
        end = body_start + 1
    else:
        # Body opener is on a separate line (Python's block, etc.)
        end = body_start
    sig_lines = lines[node.start_point[0] : end]
    return " ".join(part for line in sig_lines for part in line.split() if part)


def _extract_source(
    node: Node,
    lines: list[str],
) -> str:
    """
    Extract the full source text of a node, from start_line to end_line
    inclusive, preserving original indentation.

    Args:
        node:  the definition node
        lines: source file lines (0-indexed)

    Returns:
        Multi-line string. Includes the signature line, body, and closing
        delimiter. Never empty — returns at least the first source line.
    """
    if node.start_point[0] >= len(lines):
        return ""
    return _join_lines(lines, node.start_point[0], node.end_point[0] + 1)


def _find_body_start(node: Node) -> int | None:
    """
    Find the 0-based line index where the body of a definition begins.

    Scans the direct children of `node` for any child whose type is in
    _BODY_OPENERS. Returns that child's start_point[0] (0-based line).

    If no body opener is found among direct children, returns None,
    which signals to _extract_signature() that the whole node is the
    signature (e.g. interface method stubs, forward declarations).

    Args:
        node: the definition node (e.g. function_definition)

    Returns:
        0-based line index of the body opener node's first line,
        or None if no body opener child exists.
    """
    for child in node.children:
        if child.type in _BODY_OPENERS:
            return child.start_point[0]
    # Fall back to grandchildren: handles wrapper nodes like
    # decorated_definition where the block is a child of the inner
    # class_definition / function_definition, not the wrapper itself.
    for child in node.children:
        for grandchild in child.children:
            if grandchild.type in _BODY_OPENERS:
                return grandchild.start_point[0]
    return None


def _collect_properties(
    node: Node,
    language: str,
    lines: list[str],
) -> dict[str, object]:
    """
    Collect language-specific properties for a definition node.

    Inspects the node and its children for extras that are useful in the
    graph but don't belong in the signature or source:
      - async flag (Python, JS/TS, C#, Kotlin)
      - decorators (Python: @login_required, TS: @Injectable, etc.)
      - visibility modifier (Java, C#, Kotlin, PHP: public/private/protected)
      - return type string (extracted from type annotation nodes)
      - receiver type (Go methods: "*PaymentService")
      - abstract / static flags (Java, C#, Python abstractmethod)

    The returned dict is stored as the `properties` field on NodeRecord
    and later serialised to JSON in store.py.

    This function is best-effort — missing properties are silently omitted
    rather than raising. An empty dict is a valid return value.

    Args:
        node:     the definition node
        language: canonical language name
        lines:    source file lines (for text extraction where needed)

    Returns:
        Dict of property name → value. Values are JSON-serialisable
        (str, bool, int, list[str]).

    Examples:
        Python async function with decorator:
            {"async": True, "decorators": ["@login_required"]}
        Go method with receiver:
            {"receiver": "*PaymentService"}
        Java public static method:
            {"visibility": "public", "static": True}
    """
    props: dict[str, object] = {}

    # ── async flag ───────────────────────────────────────────────────────────
    # Python: async is an anonymous keyword child of function_definition
    # JS/TS/C#/Kotlin: same pattern — async keyword child
    if any(c.type == "async" for c in node.children):
        props["async"] = True

    # ── Python ───────────────────────────────────────────────────────────────
    if language == "python" or language in ("javascript", "typescript"):
        decorators = [
            _node_text(c).lstrip("@").strip()
            for c in node.children
            if c.type == "decorator"
        ]
        if decorators:
            props["decorators"] = decorators

    # ── Go ───────────────────────────────────────────────────────────────────
    elif language == "go":
        if node.type == "method_declaration":
            recv = node.child_by_field_name("receiver")
            if recv:
                props["receiver"] = _node_text(recv).strip("()")

    # ── JVM / C# family: visibility + static + abstract ──────────────────────
    elif language in ("java", "c_sharp", "kotlin", "scala", "php"):
        for child in node.children:
            if child.type in ("modifiers", "modifier_list", "visibility_modifier"):
                text = _node_text(child)
                for vis in ("public", "private", "protected", "internal"):
                    if vis in text:
                        props["visibility"] = vis
                        break
                if "static" in text:
                    props["static"] = True
                if "abstract" in text:
                    props["abstract"] = True

    # ── return type (best-effort, multi-language) ─────────────────────────────
    for field_name in ("return_type", "result", "type"):
        ret = node.child_by_field_name(field_name)
        if ret and ret.type not in _BODY_OPENERS:
            text = _node_text(ret).lstrip(":").strip()
            if text:
                props["return_type"] = text
            break

    return props


def _node_text(node: Node) -> str:
    """
    Decode the raw bytes of a tree-sitter node to a UTF-8 string.

    Args:
        node: any tree-sitter Node

    Returns:
        UTF-8 string with 'replace' error handling for non-UTF-8 bytes.
    """
    if node.text:
        return node.text.decode("utf-8", errors="replace")
    return ""


def _join_lines(lines: list[str], start: int, end: int) -> str:
    """
    Join a slice of source lines into a single string, preserving newlines.

    Args:
        lines: source file split by line (0-indexed)
        start: 0-based start index (inclusive)
        end:   0-based end index (exclusive)

    Returns:
        The joined string. Returns "" if start >= end or the slice is
        empty.
    """
    if start >= end or start >= len(lines):
        return ""
    return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------
#
# Maps LANG_CONFIG "parser" names to (module_name, language_fn_name).
# Most packages export a bare language() function; the few exceptions
# (typescript, php) export named variants.

_PARSER_MODULE: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "bash": ("tree_sitter_bash", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "scala": ("tree_sitter_scala", "language"),
    "lua": ("tree_sitter_lua", "language"),
    "elixir": ("tree_sitter_elixir", "language"),
}


def _get_parser(lang: str) -> Parser:
    """
    Create a tree-sitter Parser for the given parser/language name.

    Dynamically imports the corresponding tree_sitter_<lang> package and
    calls its language() (or language_<variant>()) function to build a
    Language object, then wraps it in a fresh Parser.

    Args:
        lang: parser name from LANG_CONFIG[language]["parser"],
              e.g. "python", "typescript", "c_sharp".

    Returns:
        A configured Parser instance ready to call .parse() on.

    Raises:
        KeyError:      if lang is not in _PARSER_MODULE.
        ImportError:   if the corresponding tree_sitter_* package is not
                       installed.
    """
    module_name, fn_name = _PARSER_MODULE[lang]
    mod = importlib.import_module(module_name)
    language = Language(getattr(mod, fn_name)())
    return Parser(language)
