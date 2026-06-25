from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
import typer


app = typer.Typer()


@app.command("prepare", help="Prepare a repository for benchmarking")
def prepare(
    mainfest: Annotated[
        Path,
        typer.Option(
            help="Path to the benchmark manifest file",
            file_okay=True,
            dir_okay=False,
            exists=True,
            readable=True,
        ),
    ],
): ...


@app.command("run", help="Run benchmarks on a repository")
def run(
    repo: Annotated[
        Path,
        typer.Option(
            help="Path to the repository to benchmark",
            file_okay=False,
            dir_okay=True,
            exists=True,
            readable=True,
        ),
    ],
    task: Annotated[str, typer.Option(help="The benchmark task to run")],
    policy: Annotated[list[str], typer.Option(help="The context policy to use")],
    counter: Annotated[str, typer.Option(help="The token counter to use")],
    repeat: Annotated[
        int, typer.Option(help="The number of times to repeat the benchmark")
    ] = 0,
    fail_fast: Annotated[
        bool, typer.Option(help="Whether to stop on the first failure")
    ] = True,
    json: Annotated[
        bool, typer.Option(help="Whether to output results in JSON format")
    ] = False,
): ...


@app.command("run-all", help="Run all benchmarks on a repository")
def run_all(
    repo: Annotated[
        Path,
        typer.Option(
            help="Path to the repository to benchmark",
            file_okay=False,
            dir_okay=True,
            exists=True,
            readable=True,
        ),
    ],
    policy: Annotated[list[str], typer.Option(help="The context policy to use")],
    counter: Annotated[str, typer.Option(help="The token counter to use")],
    repeat: Annotated[
        int, typer.Option(help="The number of times to repeat the benchmark")
    ] = 0,
    fail_fast: Annotated[
        bool, typer.Option(help="Whether to stop on the first failure")
    ] = True,
    json: Annotated[
        bool, typer.Option(help="Whether to output results in JSON format")
    ] = False,
): ...


if __name__ == "__main__":
    app()
