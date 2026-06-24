# Changelog

## v1.0.4

- Implemented semantic relationship extraction from real source for calls
  and imports, and wired payloads end-to-end through extraction,
  qualification, resolution, and persisted `CALLS` edges
- Added relationship extraction module support across supported languages,
  including improved import parsing for Go, Rust, Java, C#, Kotlin,
  Swift, and Scala (aliases, grouped/wildcard selectors, static imports)
- Hardened call resolution behavior for alias and scope-sensitive imports,
  including callable-local import visibility enforcement
- Expanded pipeline diagnostics and reporting with discovered/resolved/
  unresolved/unsupported call counts, malformed payload tracking, and
  relationship availability reporting
- Added real-source and end-to-end semantic tests covering extraction,
  resolver behavior, persisted call graph traversal, and fallback cases
- Updated language and CLI documentation for relationship extraction state
  and normalized import handling behavior
- Clarified CLI output by reporting unrecognized file types explicitly
  instead of displaying `Relationship unavailable: unknown`

## v1.0.3

- Promoted supported tree-sitter language parser packages to runtime
  dependencies so `indexer index` works across supported languages in
  production installs (including pipx)
- Updated installation and language docs to reflect runtime parser
  packaging and benchmark command output guidance

## v1.0.2

- Added benchmark matrix test infrastructure across language and repo-size
  combinations, including indexing runtime and token-usage metrics
- Improved benchmark output readability with a formatted summary table and
  clearer report-path visibility in `make benchmark`
- Fixed CLI packaging for pipx/PyPI installs by including the `cli` module in
  built artifacts
- Added `rich` as a runtime dependency required by the CLI entrypoint

## v1.0.1

- Implemented pass-6 v2 call resolution from `NodeRecord.properties` payloads
  (`imports`, `calls` / `call_sites`)
- Updated call-resolution pass to return edges plus `calls_resolved` and
  `calls_unresolved` statistics

## v1.0.0

- Initial release
