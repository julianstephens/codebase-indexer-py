import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from rich import print
from rich.console import Console
from typer import Argument, Option, Typer

from indexer import context, pipeline
from indexer.pipeline import PipelineConfig
from indexer.store import (
    DEFAULT_CACHE_DIR,
    SearchParams,
    default_db_path,
    open_path_readonly,
)

app = Typer(name="indexer", help="Codebase Indexer CLI")


def _resolve_db(
    project: str,
    db: Optional[str],
    cache_dir: str,
) -> str:
    if db:
        return db
    return default_db_path(project, cache_dir)


@app.command(name="index", help="Index a codebase")
def index(
    repo_path: Annotated[
        str,
        Argument(
            file_okay=False,
            dir_okay=True,
            exists=True,
            help="Path to the repository root",
        ),
    ] = ".",
    project: Annotated[
        str | None,
        Option(
            "--project",
            "-p",
            help="Project name (defaults to repo directory name)",
        ),
    ] = None,
    cache_dir: Annotated[
        str,
        Option(
            "--cache-dir",
            help="Directory for the working .db file",
            dir_okay=True,
            file_okay=False,
            exists=True,
        ),
    ] = DEFAULT_CACHE_DIR,
    workers: Annotated[
        int, Option("--workers", "-w", help="Number of parallel workers (0 = auto)")
    ] = 0,
    incremental: Annotated[
        bool, Option("--incremental/--no-incremental", help="Skip unchanged files")
    ] = True,
    export_artifact: Annotated[
        bool, Option("--export/--no-export", help="Write compressed .zst artifact")
    ] = True,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="Enable debug logging")
    ] = False,
) -> None:
    cfg = PipelineConfig(
        project=project or "",
        cache_dir=cache_dir,
        max_workers=workers,
        incremental=incremental,
        export_artifact=export_artifact,
        verbose=verbose,
    )
    try:
        result = pipeline.run(repo_path, cfg)
    except NotADirectoryError as exc:
        print(f"[red]Error:[/red] {repo_path!r} is not a directory")
        raise typer.Exit(1) from exc

    print(f"[bold]Project:[/bold]    {result.project}")
    print(f"[bold]DB:[/bold]         {result.db_path}")
    if result.artifact_path:
        print(f"[bold]Artifact:[/bold]   {result.artifact_path}")
    extracted = result.files_extracted
    unchanged = result.files_unchanged
    skipped = result.files_skipped
    print(f"Files:    {extracted} extracted, {unchanged} unchanged, {skipped} skipped")
    print(f"Nodes:      {result.nodes_total}")
    print(f"Edges:      {result.edges_total}")
    print(
        "Calls:      "
        f"{result.calls_discovered} discovered, "
        f"{result.calls_resolved} resolved, "
        f"{result.calls_unresolved} unresolved, "
        f"{result.calls_unsupported} unsupported"
    )
    if result.malformed_payloads:
        print(f"Malformed payloads: {result.malformed_payloads}")
    if result.relationship_unavailable_languages:
        unavailable = sorted(
            {
                lang
                for lang in result.relationship_unavailable_languages
                if lang != "unknown"
            }
        )
        if unavailable:
            print("Relationship unavailable: " + ", ".join(unavailable))
        if "unknown" in result.relationship_unavailable_languages:
            print("Relationship unavailable: unrecognized file types")
    print(f"Elapsed:    {result.elapsed_seconds:.2f}s")
    if result.errors:
        print(f"[red]Errors ({len(result.errors)}):[/red]")
        for path, msg in result.errors:
            print(f"  {path}: {msg}")


@app.command(
    name="skeleton",
    help="Print a skeleton of the codebase (file headers, imports, and signatures)",
)
def skeleton(
    project: Annotated[str, Argument(help="Project name")],
    db: Annotated[str | None, Option("--db", help="Path to the .db file")] = None,
    cache_dir: Annotated[
        str,
        Option(
            "--cache-dir",
            help="Directory for .db files",
            dir_okay=True,
            file_okay=False,
        ),
    ] = DEFAULT_CACHE_DIR,
    mode: Annotated[
        Literal["skeleton", "compact", "summary", "deps"] | None,
        Option(
            "--mode",
            "-m",
            help="Rendering mode: skeleton, compact, summary, deps (default: auto)",
        ),
    ] = None,
) -> None:
    db_path = _resolve_db(project, db, cache_dir)
    if not Path(db_path).exists():
        print(f"Error: database not found at {db_path!r}", file=sys.stderr)
        raise typer.Exit(1)
    try:
        text = (
            context.build_skeleton(db_path, project)
            if mode is None
            else context.build_context(db_path, project, mode=mode)
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1) from exc
    console = Console()
    console.print(text, markup=False, highlight=False)


@app.command(
    name="get-source",
    help="Get source code for a symbol",
)
def get_source(
    qualified_name: Annotated[
        str,
        Argument(help="Qualified name of the symbol, e.g. my_app.src.service.charge"),
    ],
    project: Annotated[
        str | None,
        Option("--project", "-p", help="Project name (required when db is not given)"),
    ] = None,
    db: Annotated[
        str | None,
        Option("--db", help="Path to the .db file"),
    ] = None,
    cache_dir: Annotated[
        str,
        Option(
            "--cache-dir",
            help="Directory for .db files",
            dir_okay=True,
            file_okay=False,
        ),
    ] = DEFAULT_CACHE_DIR,
) -> None:
    db_path = _resolve_db(project or "", db, cache_dir)
    if not Path(db_path).exists():
        print(f"Error: database not found at {db_path!r}", file=sys.stderr)
        raise typer.Exit(1)
    st = open_path_readonly(db_path)
    try:
        node = st.get_node_by_qn(qualified_name, project=project)
    finally:
        st.close()
    if node is None:
        print(f"Symbol not found: {qualified_name!r}", file=sys.stderr)
        raise typer.Exit(1)
    location = f"{node.file_path}:{node.start_line}-{node.end_line}"
    print(f"# {node.label}: {node.qualified_name}  ({location})")
    print(node.source)


@app.command(
    name="search",
    help="Search for symbols in the codebase",
)
def search(
    query: Annotated[
        str,
        Argument(help="Full-text search query"),
    ],
    project: Annotated[
        str | None,
        Option("--project", "-p", help="Filter by project name"),
    ] = None,
    label: Annotated[
        str | None,
        Option(
            "--label",
            "-l",
            help="Filter by label: Function, Class, Method, Interface, Type",
        ),
    ] = None,
    file: Annotated[
        str | None,
        Option(
            "--file",
            "-f",
            help="SQL LIKE pattern for file path, e.g. 'src/payments/%'",
        ),
    ] = None,
    limit: Annotated[
        int,
        Option("--limit", "-n", help="Maximum number of results"),
    ] = 20,
    db: Annotated[
        str | None,
        Option("--db", help="Path to the .db file"),
    ] = None,
    cache_dir: Annotated[
        str,
        Option("--cache-dir", help="Directory for .db files"),
    ] = DEFAULT_CACHE_DIR,
) -> None:
    if not db and not project:
        print("Error: provide --project or --db", file=sys.stderr)
        raise typer.Exit(1)
    db_path = _resolve_db(project or "", db, cache_dir)
    if not Path(db_path).exists():
        print(f"Error: database not found at {db_path!r}", file=sys.stderr)
        raise typer.Exit(1)
    params = SearchParams(
        project=project,
        label=label,
        file_pattern=file,
        fts_query=query,
        limit=limit,
    )
    st = open_path_readonly(db_path)
    try:
        result = st.search_nodes(params)
    finally:
        st.close()
    if not result.rows:
        print("No results found.", file=sys.stderr)
        return
    print(f"Found {result.total} result(s) (showing {len(result.rows)}):\n")
    for node in result.rows:
        print(f"[{node.label}] {node.qualified_name}")
        print(f"  {node.file_path}:{node.start_line}  {node.signature}")
        print()
