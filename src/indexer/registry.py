"""
registry.py — Symbol registry and call resolution.

The registry is built once after all files have been extracted (pass 4
in the pipeline). It indexes every NodeRecord by qualified name, short
name, and module prefix so the call resolution pass (pass 5) can look
up call sites without re-scanning the graph buffer.

Call resolution turns raw call site strings (e.g. "charge", "service.charge",
"PaymentService.charge") into qualified names that match nodes in the
graph. The result is an edge (source_qn, target_qn, "CALLS", properties)
where properties carries the confidence score and resolution strategy.

Resolution strategy chain (first hit wins):
    1. same_module   — callee name (or dotted suffix) matches a node
                       whose QN starts with the calling file's module
                       prefix. Confidence 0.95.
    2. import_map    — the callee's root name appears in the calling
                       file's import list, and that import resolves to
                       a known module QN prefix. Confidence 0.85.
    3. fuzzy         — bare callee name matches exactly one node across
                       all modules. Confidence 0.40. Skipped when the
                       name is too common (appears in > MAX_FUZZY_MATCHES
                       nodes).
    4. unresolved    — none of the above succeeded. Returns a Resolution
                       with target_qn="" and confidence 0.0. The pipeline
                       emits no edge for unresolved calls.

Public types:
    Registry         — the main symbol index
    CallSite         — a raw call site extracted from source
    Import           — a parsed import statement
    Resolution       — the result of resolving one CallSite

Public API:
    build(records)               — construct a Registry from NodeRecords
    Registry.resolve(call, ctx)  — resolve one CallSite to a Resolution
    Registry.resolve_all(calls, ctx) — resolve a list of CallSites
"""

import logging
from dataclasses import dataclass, field

from .languages import EXTENSION_TO_LANG
from .treesitter import NodeRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence scores for each resolution strategy.
CONFIDENCE_SAME_MODULE: float = 0.95
CONFIDENCE_IMPORT_MAP: float = 0.85
CONFIDENCE_FUZZY: float = 0.40

# Maximum number of nodes that share a short name before fuzzy matching
# is suppressed. A name that matches 10 nodes is ambiguous; picking one
# at random produces noisy edges.
MAX_FUZZY_MATCHES: int = 5

# Minimum length for a callee name to be considered for fuzzy matching.
# Single-character names (e.g. "f", "x") are almost always local
# variables, not callees worth resolving.
MIN_FUZZY_NAME_LEN: int = 2


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CallSite:
    """
    A raw call site extracted from a source file by the extractor.

    Attributes:
        callee:     the raw callee string as it appears in source.
                    May be a bare name ("charge"), a dotted name
                    ("service.charge"), or a method call
                    ("self.service.charge"). Never empty.
        line:       1-based source line where the call appears.
        qualifier:  the dotted prefix before the final name, if any.
                    For "service.charge", qualifier="service" and
                    the callee stored here is still "service.charge".
                    Populated by the extractor; may be "".
        in_function: qualified name of the function/method that contains
                    this call site. Used to set the source_qn on the
                    resulting edge. May be "" for module-level calls.
    """

    callee: str
    line: int
    qualifier: str = ""
    in_function: str = ""


@dataclass
class Import:
    """
    A parsed import statement from a source file.

    Used by the import_map resolution strategy to map short names to
    module QN prefixes.

    Attributes:
        module_path: the imported module path as it appears in source.
                     For "from src.payments import service", this is
                     "src.payments". For "import os.path", this is
                     "os.path".
        names:       list of names imported from the module. For
                     "from src.payments import service, models", this
                     is ["service", "models"]. Empty list for bare
                     "import X" statements.
        alias:       the alias if "as Y" was used, else "". For
                     "import numpy as np", alias="np".
        line:        1-based source line of the import statement.
        in_function: qualified name of the callable where the import is
                     visible. Empty string means file-level visibility.
    """

    module_path: str
    names: list[str] = field(default_factory=list)
    alias: str = ""
    line: int = 0
    in_function: str = ""


@dataclass
class Resolution:
    """
    The result of resolving one CallSite.

    Attributes:
        source_qn:  qualified name of the calling function/method.
                    Copied from CallSite.in_function. May be "" for
                    module-level calls (no edge will be emitted).
        target_qn:  qualified name of the resolved callee node, or ""
                    if resolution failed (strategy="unresolved").
        strategy:   one of: "same_module" | "import_map" | "fuzzy" |
                    "unresolved".
        confidence: float in [0.0, 1.0]. 0.0 means unresolved.
        call_site:  the original CallSite that produced this resolution.
    """

    source_qn: str
    target_qn: str
    strategy: str
    confidence: float
    call_site: CallSite


@dataclass
class ResolutionContext:
    """
    Per-file context passed to Registry.resolve().

    Built by the pipeline for each file before resolving its call sites.
    Avoids re-computing module_qn and the import map for every call site
    in the same file.

    Attributes:
        file_path:  repo-relative file path of the file being resolved.
        module_qn:  dotted module QN for the file, e.g.
                    "src.payments.service". Computed by fqn.module().
        imports:    list of Import objects parsed from the file's import
                    statements.
        project:    project name, used to scope node lookups.
    """

    file_path: str
    module_qn: str
    imports: list[Import] = field(default_factory=list)
    project: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Registry:
    """
    In-memory symbol index for call resolution.

    Built from a flat list of NodeRecord objects by build(). Three
    indexes are maintained:

      _by_qn       dict[qn → NodeRecord]
                   Exact QN lookup. Used by same_module strategy to
                   check if a candidate QN exists.

      _by_name     dict[short_name → list[NodeRecord]]
                   All nodes sharing a short name. Used by fuzzy
                   strategy. A name maps to multiple nodes when the
                   same function name appears in different modules.

      _by_module   dict[module_prefix → list[NodeRecord]]
                   All nodes whose QN starts with a given module prefix.
                   Used by same_module strategy to enumerate candidates
                   in the caller's own module.

    Do not instantiate directly. Use build().
    """

    def __init__(self) -> None:
        """
        Initialise an empty registry.

        All three indexes start empty. Populated by _index_record().
        """
        self._by_qn: dict[str, NodeRecord] = {}
        self._by_name: dict[str, list[NodeRecord]] = {}
        self._by_module: dict[str, list[NodeRecord]] = {}

    # ── Indexing ──────────────────────────────────────────────────────────

    def _index_record(self, record: NodeRecord) -> None:
        """
        Add a single NodeRecord to all three indexes.

        Called once per record by build(). Records with an empty
        qualified_name are silently skipped.

        _by_module is keyed by every prefix of the QN up to (but not
        including) the final component. For a QN like
        "src.payments.service.charge", the prefixes added are:
          "src"
          "src.payments"
          "src.payments.service"
        This allows same_module to find all nodes in a module even when
        the caller's module_qn is a parent package.

        Args:
            record: NodeRecord with qualified_name set.
        """
        qn = record.qualified_name
        if not qn:
            logger.debug("Skipping record with empty qualified_name: %s", record)
            return

        self._by_qn[qn] = record

        self._by_name.setdefault(record.name, []).append(record)

        for prefix in _module_prefixes(qn):
            self._by_module.setdefault(prefix, []).append(record)

    # ── Lookup ────────────────────────────────────────────────────────────

    def get_by_qn(self, qn: str) -> NodeRecord | None:
        """
        Return the NodeRecord for an exact qualified name, or None.

        Args:
            qn: fully qualified name string.

        Returns:
            NodeRecord if found, None otherwise.

        Examples:
            >>> reg.get_by_qn("src.payments.service.charge")
            NodeRecord(name='charge', ...)
            >>> reg.get_by_qn("nonexistent") is None
            True
        """
        return self._by_qn.get(qn)

    def get_by_name(self, name: str) -> list[NodeRecord]:
        """
        Return all NodeRecords with the given short name.

        Args:
            name: short symbol name, e.g. "charge".

        Returns:
            List of matching NodeRecords (may be empty). The list is a
            copy — mutating it does not affect the index.

        Examples:
            >>> reg.get_by_name("charge")
            [NodeRecord(name='charge', qualified_name='src.payments.service.charge',
                ...)]
        """
        return self._by_name.get(name, []).copy()

    def get_by_module(self, module_qn: str) -> list[NodeRecord]:
        """
        Return all NodeRecords whose QN starts with module_qn.

        Used by _resolve_same_module() to find candidates in the
        caller's own module without iterating the full index.

        Args:
            module_qn: dotted module prefix, e.g. "src.payments.service"

        Returns:
            List of matching NodeRecords (may be empty). The list is a
            copy.

        Examples:
            >>> reg.get_by_module("src.payments.service")
            [NodeRecord(name='charge', ...), NodeRecord(name='refund', ...)]
            >>> reg.get_by_module("src.nonexistent")
            []
        """
        return self._by_module.get(module_qn, []).copy()

    def __len__(self) -> int:
        """Return the total number of indexed nodes."""
        return len(self._by_qn)

    def __contains__(self, qn: str) -> bool:
        """
        Return True if a node with the given QN is in the registry.

        Args:
            qn: fully qualified name string.

        Examples:
            >>> "src.payments.service.charge" in reg
            True
            >>> "nonexistent" in reg
            False
        """
        return qn in self._by_qn

    # ── Resolution ────────────────────────────────────────────────────────

    def resolve(
        self,
        call: CallSite,
        ctx: ResolutionContext,
    ) -> Resolution:
        """
        Resolve a single CallSite to a Resolution.

        Tries each strategy in order and returns the first successful
        result. If all strategies fail, returns a Resolution with
        target_qn="" and strategy="unresolved".

        Args:
            call: CallSite extracted from the source file
            ctx:  ResolutionContext for the file containing this call

        Returns:
            Resolution with source_qn, target_qn, strategy, confidence.

        Examples:
            >>> ctx = ResolutionContext(
            ...     file_path="src/payments/views.py",
            ...     module_qn="src.payments.views",
            ...     imports=[Import("src.payments.service", ["charge"])],
            ...     project="my-app",
            ... )
            >>> call = CallSite(callee="charge", line=42,
            ...                 in_function="src.payments.views.checkout")
            >>> res = reg.resolve(call, ctx)
            >>> res.strategy
            'import_map'
            >>> res.target_qn
            'src.payments.service.charge'
        """
        for strategy in (
            self._resolve_same_module,
            self._resolve_import_map,
            self._resolve_fuzzy,
        ):
            result = strategy(call, ctx)
            if result is not None:
                return result
        return self._make_unresolved(call)

    def resolve_all(
        self,
        calls: list[CallSite],
        ctx: ResolutionContext,
    ) -> list[Resolution]:
        """
        Resolve a list of CallSites for a single file.

        Calls resolve() for each CallSite and collects results. Filters
        out resolutions where source_qn is "" (module-level calls with
        no enclosing function) since no edge can be emitted for them.

        Args:
            calls: list of CallSite objects from one source file
            ctx:   ResolutionContext for the same file

        Returns:
            List of Resolution objects. Only includes resolutions where
            source_qn is non-empty. Unresolved calls (target_qn="") are
            included so the pipeline can log statistics.

        Examples:
            >>> resolutions = reg.resolve_all(calls, ctx)
            >>> resolved = [r for r in resolutions if r.target_qn]
            >>> len(resolved) >= 0
            True
        """
        return [self.resolve(call, ctx) for call in calls if call.in_function]

    # ── Private resolution strategies ─────────────────────────────────────

    def _resolve_same_module(
        self,
        call: CallSite,
        ctx: ResolutionContext,
    ) -> Resolution | None:
        """
        Strategy 1: same_module — resolve within the calling file's module.

        Looks for a node in the registry whose QN is
        "<ctx.module_qn>.<callee_tail>" where callee_tail is the last
        component of the (possibly dotted) callee string.

        Also checks parent package prefixes up to two levels up. For a
        caller in "src.payments.views", checks:
          "src.payments.views.<callee>"
          "src.payments.<callee>"
          "src.<callee>"
        This handles calls to sibling modules in the same package.

        For dotted callees like "PaymentService.charge":
          1. Check if the qualifier ("PaymentService") resolves to a
             Class node in the same module.
          2. If so, look for "<class_qn>.charge".

        Args:
            call: CallSite with callee and in_function set
            ctx:  ResolutionContext with module_qn

        Returns:
            Resolution with strategy="same_module" and confidence
            CONFIDENCE_SAME_MODULE, or None if not found.
        """
        callee = _strip_self(call.callee)
        bare = _bare_name(callee)
        qual = _qualifier(callee)

        # Build candidate prefixes: module itself + up to 2 parent packages.
        # For "src.payments.views" → ["src.payments.views", "src.payments", "src"]
        parts = ctx.module_qn.split(".")
        n = len(parts)
        prefixes = [".".join(parts[: n - i]) for i in range(min(n, 3))]

        # Dotted callee like "PaymentService.charge": resolve qualifier as
        # a Class in the same module, then look for <class_qn>.bare.
        if qual:
            qual_bare = _bare_name(qual)
            for prefix in prefixes:
                class_qn = f"{prefix}.{qual_bare}"
                class_node = self._by_qn.get(class_qn)
                if class_node and class_node.label in ("Class", "Interface"):
                    method_qn = f"{class_qn}.{bare}"
                    if method_qn in self._by_qn:
                        return Resolution(
                            source_qn=call.in_function,
                            target_qn=method_qn,
                            strategy="same_module",
                            confidence=CONFIDENCE_SAME_MODULE,
                            call_site=call,
                        )

        # Plain name: check <prefix>.<bare> at each module level.
        for prefix in prefixes:
            candidate = f"{prefix}.{bare}"
            if candidate in self._by_qn:
                return Resolution(
                    source_qn=call.in_function,
                    target_qn=candidate,
                    strategy="same_module",
                    confidence=CONFIDENCE_SAME_MODULE,
                    call_site=call,
                )

        return None

    def _resolve_import_map(
        self,
        call: CallSite,
        ctx: ResolutionContext,
    ) -> Resolution | None:
        """
        Strategy 2: import_map — resolve via the file's import list.

        Builds an import map from ctx.imports: a dict mapping each
        imported name (or alias) to its source module QN.

        For a call site "charge":
          - Find an import where "charge" is in Import.names or
            Import.alias == "charge".
          - If found, the target is "<import.module_path>.charge"
            (converted to dotted QN).
          - Check that this QN exists in the registry.

        For a dotted call "service.charge":
          - Find an import where "service" is in Import.names or
            Import.alias == "service".
          - If found, look up "<import.module_path>.charge" (the
            qualifier "service" resolves to the module, "charge" is
            the symbol within it).
          - Check that this QN exists in the registry.

        Args:
            call: CallSite with callee set
            ctx:  ResolutionContext with imports list

        Returns:
            Resolution with strategy="import_map" and confidence
            CONFIDENCE_IMPORT_MAP, or None if not found.
        """
        callee = _strip_self(call.callee)
        bare = _bare_name(callee)  # final symbol name, e.g. "charge"
        qual = _qualifier(callee)  # prefix before the last dot, e.g. "service"

        for imp in ctx.imports:
            # Function-local imports are visible only in their owning callable.
            if imp.in_function and imp.in_function != call.in_function:
                continue

            mod = _normalise_module_path(imp.module_path)

            if qual:
                # Dotted call: "service.charge" or "svc.charge"
                root = qual.split(".")[0]
                if imp.alias == root:
                    # For import aliases, resolve differently based on import kind:
                    # - import module as m         → m.func()     => <mod>.func
                    # - from pkg import Sym as s   → s.method()   => <mod>.Sym.method
                    if imp.names:
                        candidate = f"{mod}.{imp.names[0]}.{bare}"
                    else:
                        candidate = f"{mod}.{bare}"
                    if candidate in self._by_qn:
                        return Resolution(
                            source_qn=call.in_function,
                            target_qn=candidate,
                            strategy="import_map",
                            confidence=CONFIDENCE_IMPORT_MAP,
                            call_site=call,
                        )
                elif root in imp.names:
                    # "from parent import sub" → sub is a submodule name
                    # → target = <mod>.<root>.<bare>
                    candidate = f"{mod}.{root}.{bare}"
                    if candidate in self._by_qn:
                        return Resolution(
                            source_qn=call.in_function,
                            target_qn=candidate,
                            strategy="import_map",
                            confidence=CONFIDENCE_IMPORT_MAP,
                            call_site=call,
                        )
            else:
                # Bare call: "charge" imported directly from a module.
                if bare in imp.names:
                    candidate = f"{mod}.{bare}"
                    if candidate in self._by_qn:
                        return Resolution(
                            source_qn=call.in_function,
                            target_qn=candidate,
                            strategy="import_map",
                            confidence=CONFIDENCE_IMPORT_MAP,
                            call_site=call,
                        )
                elif imp.alias == bare:
                    # Alias can point either to a module or to a named symbol.
                    candidate = f"{mod}.{imp.names[0]}" if imp.names else mod
                    if candidate in self._by_qn:
                        return Resolution(
                            source_qn=call.in_function,
                            target_qn=candidate,
                            strategy="import_map",
                            confidence=CONFIDENCE_IMPORT_MAP,
                            call_site=call,
                        )

        return None

    def _resolve_fuzzy(
        self,
        call: CallSite,
        ctx: ResolutionContext,
    ) -> Resolution | None:
        """
        Strategy 3: fuzzy — resolve by bare short name across all modules.

        Looks up the callee's final name component in _by_name. Returns
        a resolution only when exactly one node matches and the callee
        name meets the minimum length requirement.

        Suppressed when:
          - len(bare_name) < MIN_FUZZY_NAME_LEN
          - len(matches) == 0 (not found)
          - len(matches) > MAX_FUZZY_MATCHES (too ambiguous)

        Prefers nodes in the same project (ctx.project) over nodes in
        other projects when multiple matches exist within the fuzzy
        threshold.

        Args:
            call: CallSite with callee set
            ctx:  ResolutionContext with project name

        Returns:
            Resolution with strategy="fuzzy" and confidence
            CONFIDENCE_FUZZY, or None if suppressed.
        """
        callee = _strip_self(call.callee)
        bare = _bare_name(callee)

        if len(bare) < MIN_FUZZY_NAME_LEN:
            return None

        matches = self._by_name.get(bare, [])

        if not matches or len(matches) > MAX_FUZZY_MATCHES:
            return None

        # Exactly one match — return it directly.
        if len(matches) == 1:
            return Resolution(
                source_qn=call.in_function,
                target_qn=matches[0].qualified_name,
                strategy="fuzzy",
                confidence=CONFIDENCE_FUZZY,
                call_site=call,
            )

        # Multiple matches within threshold — prefer same-project nodes.
        if ctx.project:
            prefix = ctx.project + "."
            same = [m for m in matches if m.qualified_name.startswith(prefix)]
            if len(same) == 1:
                return Resolution(
                    source_qn=call.in_function,
                    target_qn=same[0].qualified_name,
                    strategy="fuzzy",
                    confidence=CONFIDENCE_FUZZY,
                    call_site=call,
                )

        return None

    def _make_unresolved(
        self,
        call: CallSite,
    ) -> Resolution:
        """
        Return an unresolved Resolution for a call site.

        Called when all three strategies return None. The pipeline uses
        this to count unresolved calls for statistics but emits no edge.

        Args:
            call: the CallSite that could not be resolved

        Returns:
            Resolution(source_qn=call.in_function, target_qn="",
                       strategy="unresolved", confidence=0.0,
                       call_site=call)
        """
        return Resolution(
            source_qn=call.in_function,
            target_qn="",
            strategy="unresolved",
            confidence=0.0,
            call_site=call,
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build(records: list[NodeRecord]) -> Registry:
    """
    Construct a Registry from a flat list of NodeRecord objects.

    Iterates records once, calling _index_record() for each. Records
    with empty qualified_names are skipped with a debug log message.

    Args:
        records: list of NodeRecord objects with qualified_name set.
                 Typically the full output of the extraction pass.

    Returns:
        A fully populated Registry ready for call resolution.

    Examples:
        >>> reg = build(all_node_records)
        >>> len(reg)
        247
        >>> "src.payments.service.charge" in reg
        True
    """
    reg = Registry()
    for record in records:
        if not record.qualified_name:
            logger.debug("Skipping record with empty qualified_name: %s", record)
            continue
        reg._index_record(record)
    return reg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _module_prefixes(qn: str) -> list[str]:
    """
    Return all dotted prefixes of a qualified name, excluding the full
    QN itself and the empty string.

    Used by _index_record() to populate _by_module for every prefix
    of a node's QN.

    Args:
        qn: fully qualified name, e.g. "src.payments.service.charge"

    Returns:
        List of prefix strings in shortest-to-longest order:
          ["src", "src.payments", "src.payments.service"]

    Examples:
        >>> _module_prefixes("src.payments.service.charge")
        ['src', 'src.payments', 'src.payments.service']
        >>> _module_prefixes("foo")
        []
        >>> _module_prefixes("")
        []
    """
    parts = qn.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _bare_name(callee: str) -> str:
    """
    Extract the final component of a (possibly dotted) callee string.

    Used by _resolve_fuzzy() and _resolve_same_module() to isolate the
    symbol name from any qualifier prefix.

    Args:
        callee: raw callee string, e.g. "service.charge" or "charge"

    Returns:
        The last dot-separated component, e.g. "charge".
        Returns the input unchanged if there are no dots.

    Examples:
        >>> _bare_name("charge")
        'charge'
        >>> _bare_name("service.charge")
        'charge'
        >>> _bare_name("self.service.charge")
        'charge'
    """
    return callee.rpartition(".")[2]


def _qualifier(callee: str) -> str:
    """
    Extract everything before the final dot in a callee string.

    Used by _resolve_same_module() and _resolve_import_map() to split
    "service.charge" into qualifier="service" and name="charge".

    Args:
        callee: raw callee string

    Returns:
        Everything before the last dot, or "" if there are no dots.

    Examples:
        >>> _qualifier("service.charge")
        'service'
        >>> _qualifier("self.service.charge")
        'self.service'
        >>> _qualifier("charge")
        ''
    """
    return callee.rpartition(".")[0]


def _normalise_module_path(import_path: str) -> str:
    """
    Convert an import path to a dotted QN-style module prefix.

    Handles two common forms:
      - Relative paths: "src/payments/service" → "src.payments.service"
      - Dotted paths:   "src.payments.service" → "src.payments.service"
      - Paths with extension: "src/payments/service.py" → "src.payments.service"

    Leading dots (Python relative imports like "..utils") are stripped
    — relative imports are resolved by the same_module strategy instead.

    Args:
        import_path: raw import path string from an Import.module_path

    Returns:
        Normalised dotted module prefix string.

    Examples:
        >>> _normalise_module_path("src/payments/service")
        'src.payments.service'
        >>> _normalise_module_path("src.payments.service")
        'src.payments.service'
        >>> _normalise_module_path("src/payments/service.py")
        'src.payments.service'
        >>> _normalise_module_path("..utils")
        'utils'
    """
    path = import_path.lstrip(".")
    for ext in EXTENSION_TO_LANG:
        if path.endswith(ext):
            path = path[: -len(ext)]
            break
    return path.replace("/", ".")


def _strip_self(callee: str) -> str:
    """
    Remove a leading "self." or "cls." prefix from a callee string.

    Python method calls frequently appear as "self.helper()" in source.
    Stripping the self/cls prefix allows the name to be matched against
    the registry without it.

    Args:
        callee: raw callee string from a CallSite

    Returns:
        Callee with leading "self." or "cls." removed, or the input
        unchanged if neither prefix is present.

    Examples:
        >>> _strip_self("self.charge")
        'charge'
        >>> _strip_self("cls.create")
        'create'
        >>> _strip_self("service.charge")
        'service.charge'
        >>> _strip_self("charge")
        'charge'
    """
    return callee.removeprefix("self.").removeprefix("cls.")
