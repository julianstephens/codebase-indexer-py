# Supported Languages

The following languages are supported via tree-sitter parsers:

All parser packages for the languages below are included as runtime
dependencies in `codebase-indexer-py`, so standard installs (including pipx)
work without manual parser setup.

| Language | Extensions |
| --- | --- |
| Python | `.py` `.pyi` |
| TypeScript | `.ts` `.tsx` |
| JavaScript | `.js` `.jsx` `.mjs` `.cjs` |
| Go | `.go` |
| Rust | `.rs` |
| Java | `.java` |
| C | `.c` `.h` |
| C++ | `.cpp` `.cc` `.cxx` `.hpp` `.hxx` |
| C# | `.cs` |
| Ruby | `.rb` |
| PHP | `.php` |
| Kotlin | `.kt` `.kts` |
| Swift | `.swift` |
| Scala | `.scala` |
| Lua | `.lua` |
| Elixir | `.ex` `.exs` |
| Bash | `.sh` `.bash` |

## Relationship Extraction State

State meanings:

1. Definitions and relationships supported
2. Definitions supported, relationship extraction unavailable
3. Fallback-only

Current state by language extension mapping:

| Language | State |
| --- | --- |
| Python | 1 |
| TypeScript | 1 |
| JavaScript | 1 |
| Go | 1 |
| Rust | 1 |
| Java | 1 |
| C | 1 |
| C++ | 1 |
| C# | 1 |
| Ruby | 1 |
| PHP | 1 |
| Kotlin | 1 |
| Swift | 1 |
| Scala | 1 |
| Lua | 1 |
| Elixir | 1 |
| Bash | 1 |

Unrecognised file types are state 3 (fallback-only).

## Import Extraction Notes

The relationship extractor currently normalizes these representative import forms:

- Java: `import java.util.List;`, `import static java.util.Collections.*;`
- C#: `using SB = System.Text.StringBuilder;`, `using static System.Math;`
- Rust: `use crate::payments::service::Charge as PayCharge;`, grouped `use` selectors
- Kotlin: alias imports (`import a.b.C as Alias`) and wildcard imports (`import a.b.*`)
- Scala: selector aliases (`import a.b.{C => Alias}`), wildcard (`import a.b._`), and hidden selectors (`C => _`) ignored for binding

Import payloads are attached to symbol records when definitions exist, and to fallback `File` records when a parsed file has no extractable definitions.

## Unrecognised file types

Files with extensions not listed above (YAML, TOML, Dockerfile, SQL, Markdown, etc.) are stored as single `File` nodes so `get_source()` still works on them. To enable this behaviour, set `include_unknown_extensions=True` in `WalkConfig`.
