import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytest

from src.indexer.context import build_context, estimate_tokens
from src.indexer.pipeline import PipelineConfig, run


@dataclass(frozen=True)
class RepoSize:
    name: str
    files: int
    functions_per_file: int


@dataclass(frozen=True)
class LanguageCase:
    name: str
    extension: str


class UnsupportedLanguageCaseError(ValueError):
    def __init__(self, language: str):
        super().__init__(f"Unsupported language case: {language}")


REPEATS = 2
REPO_SIZES = [
    RepoSize(name="small", files=10, functions_per_file=10),
    RepoSize(name="medium", files=40, functions_per_file=14),
    RepoSize(name="large", files=120, functions_per_file=18),
]
LANGUAGE_CASES = [
    LanguageCase(name="python", extension=".py"),
    LanguageCase(name="typescript", extension=".ts"),
    LanguageCase(name="go", extension=".go"),
]


def _python_source(module_idx: int, functions_per_file: int) -> str:
    lines = [
        f"class Service{module_idx}:",
        "    def __init__(self) -> None:",
        "        self.value = 0",
        "",
    ]
    for fn_idx in range(functions_per_file):
        lines.extend(
            [
                f"def function_{module_idx}_{fn_idx}(value: int) -> int:",
                f"    total = value + {fn_idx}",
                "    for i in range(3):",
                "        total += i",
                "    return total",
                "",
            ]
        )
    lines.extend(
        [
            f"def orchestrate_{module_idx}(value: int) -> int:",
            "    return "
            + " + ".join(
                [
                    f"function_{module_idx}_{i}(value)"
                    for i in range(min(4, functions_per_file))
                ]
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _typescript_source(module_idx: int, functions_per_file: int) -> str:
    lines = [
        f"export class Service{module_idx} {{",
        "  value: number;",
        "  constructor() {",
        "    this.value = 0;",
        "  }",
        "}",
        "",
    ]
    for fn_idx in range(functions_per_file):
        lines.extend(
            [
                (
                    "export function "
                    f"function_{module_idx}_{fn_idx}(value: number): number {{"
                ),
                f"  let total = value + {fn_idx};",
                "  for (let i = 0; i < 3; i += 1) {",
                "    total += i;",
                "  }",
                "  return total;",
                "}",
                "",
            ]
        )
    called = " + ".join(
        [f"function_{module_idx}_{i}(value)" for i in range(min(4, functions_per_file))]
    )
    lines.extend(
        [
            f"export function orchestrate_{module_idx}(value: number): number {{",
            f"  return {called};",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _go_source(module_idx: int, functions_per_file: int) -> str:
    lines = [
        "package main",
        "",
        "type Service struct {",
        "    Value int",
        "}",
        "",
    ]
    for fn_idx in range(functions_per_file):
        lines.extend(
            [
                f"func Function{module_idx}_{fn_idx}(value int) int {{",
                f"    total := value + {fn_idx}",
                "    for i := 0; i < 3; i++ {",
                "        total += i",
                "    }",
                "    return total",
                "}",
                "",
            ]
        )
    called = " + ".join(
        [f"Function{module_idx}_{i}(value)" for i in range(min(4, functions_per_file))]
    )
    lines.extend(
        [
            f"func Orchestrate{module_idx}(value int) int {{",
            f"    return {called}",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_source(language: str, module_idx: int, functions_per_file: int) -> str:
    if language == "python":
        return _python_source(module_idx, functions_per_file)
    if language == "typescript":
        return _typescript_source(module_idx, functions_per_file)
    if language == "go":
        return _go_source(module_idx, functions_per_file)
    raise UnsupportedLanguageCaseError(language)


def _write_repo(
    root: Path, language_case: LanguageCase, repo_size: RepoSize
) -> dict[str, str]:
    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for idx in range(repo_size.files):
        rel_path = f"src/module_{idx}{language_case.extension}"
        content = _build_source(language_case.name, idx, repo_size.functions_per_file)
        path = root / rel_path
        path.write_text(content, encoding="utf-8")
        files[rel_path] = content

    return files


def _print_summary(results: list[dict]) -> None:
    rows = sorted(results, key=lambda item: (item["language"], item["size"]))
    table_rows = [
        {
            "language": row["language"],
            "size": row["size"],
            "avg_seconds": f"{row['avg_index_seconds']:.4f}",
            "raw_tokens": f"{row['raw_tokens']:,}",
            "context_tokens": f"{row['context_tokens']:,}",
            "saved_tokens": f"{row['saved_tokens']:,}",
            "saved_percent": f"{row['saved_percent']:.2f}%",
        }
        for row in rows
    ]

    columns = [
        ("language", "Language"),
        ("size", "Size"),
        ("avg_seconds", "Avg Index (s)"),
        ("raw_tokens", "Raw Tokens"),
        ("context_tokens", "Context Tokens"),
        ("saved_tokens", "Saved Tokens"),
        ("saved_percent", "Saved %"),
    ]
    widths = {
        key: max(len(title), *(len(row[key]) for row in table_rows))
        for key, title in columns
    }

    def _line(values: list[str]) -> str:
        padded = [
            f" {value:{widths[key]}} "
            for (key, _), value in zip(columns, values, strict=False)
        ]
        return "|" + "|".join(padded) + "|"

    print("\nindexing benchmark summary")
    print(_line([title for _, title in columns]))
    print("|" + "|".join(["-" * (widths[key] + 2) for key, _ in columns]) + "|")
    for row in table_rows:
        print(_line([row[key] for key, _ in columns]))


@pytest.fixture(scope="session")
def benchmark_results(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[list[dict], None, None]:
    results: list[dict] = []
    yield results

    if not results:
        return

    report_path = tmp_path_factory.getbasetemp() / "indexing-benchmark-report.json"
    report_path.write_text(
        json.dumps(
            {"results": sorted(results, key=lambda x: (x["language"], x["size"]))},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nbenchmark report: {report_path}")
    _print_summary(results)


@pytest.mark.benchmark
@pytest.mark.parametrize("language_case", LANGUAGE_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("repo_size", REPO_SIZES, ids=lambda s: s.name)
def test_indexing_runtime_and_static_context_compression_matrix(
    tmp_path: Path,
    benchmark_results: list[dict],
    language_case: LanguageCase,
    repo_size: RepoSize,
) -> None:
    repo_root = tmp_path / f"repo-{language_case.name}-{repo_size.name}"
    file_map = _write_repo(repo_root, language_case, repo_size)

    raw_tokens = estimate_tokens("\n".join(file_map.values()))
    indexing_times: list[float] = []
    db_path = ""
    project_name = ""

    for repeat_idx in range(REPEATS):
        project_name = f"benchmark-{language_case.name}-{repo_size.name}-r{repeat_idx}"
        result = run(
            str(repo_root),
            PipelineConfig(
                project=project_name,
                cache_dir=str(tmp_path / "cache"),
                incremental=False,
                export_artifact=False,
                max_workers=1,
            ),
        )
        indexing_times.append(result.elapsed_seconds)
        db_path = result.db_path

        assert result.files_extracted == repo_size.files
        assert result.errors == []

    context_text = build_context(db_path, project_name, token_budget=8_000)
    context_tokens = estimate_tokens(context_text)
    avg_index_seconds = statistics.fmean(indexing_times)
    saved_tokens = max(raw_tokens - context_tokens, 0)
    saved_percent = (saved_tokens / raw_tokens) * 100 if raw_tokens else 0.0

    row = {
        "language": language_case.name,
        "size": repo_size.name,
        "files": repo_size.files,
        "functions_per_file": repo_size.functions_per_file,
        "raw_tokens": raw_tokens,
        "context_tokens": context_tokens,
        "avg_raw_tokens_per_file": raw_tokens / repo_size.files,
        "avg_context_tokens_per_file": context_tokens / repo_size.files,
        "saved_tokens": saved_tokens,
        "saved_percent": saved_percent,
        "avg_index_seconds": avg_index_seconds,
        "runs": REPEATS,
    }
    benchmark_results.append(row)

    assert avg_index_seconds > 0
    assert raw_tokens > 0
    assert 0 <= saved_percent <= 100
