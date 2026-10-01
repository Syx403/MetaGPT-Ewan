# Runtime and dependency guide

[Back to MAT](../README.md)

## Explore without credentials

The fastest way to understand MAT is to read the checked-in
[KO report](../MAT/report/Strategy/Strategy_report_KO_2022.md) and
[AAPL JSON report](../MAT/report/Strategy/Strategy_report_AAPL_2022.json), then follow
the report fields into [the schemas](../MAT/schemas.py).

This path needs no installation, API credentials, or model calls.

## Dependency inventory

The current source imports the following external packages. This is a source-based
inventory, **not a tested version lock or a claim that arbitrary latest versions
are compatible**.

| Package | Use in MAT |
| --- | --- |
| `metagpt` | Roles, Actions, messages, environment, model access, logging, and search support. |
| `pydantic` | Structured evidence and report contracts. |
| `PyYAML` | Reading the private MAT configuration. |
| `aiohttp` | HTTP/API requests. |
| `pandas`, `pandas-ta`, `yfinance` | Price data and technical indicators. |
| `beautifulsoup4` | Converting financial-source HTML tables to readable evidence. |
| `tavily-python` | The Tavily search client for news and targeted investigation. |

Some of these are also dependencies of MetaGPT. Select compatible versions and a
supported Python interpreter together in an isolated environment. This repository
does not yet contain a validated installation lock; a fresh live installation is
remaining engineering work rather than part of the no-install report tour.

## Live execution in a configured environment

The existing runner accepts a ticker and fiscal year:

```bash
# From the repository root, after establishing a compatible runtime and your own configuration:
python MAT/main.py --ticker KO --fiscal_year 2022
```

Before using it, configure:

1. MetaGPT's model/provider access for its Actions.
2. Your own RAGFlow endpoint and collection containing the relevant financial documents.
3. Your own search-provider credentials and desired data windows.
4. `config/config2.yaml` as described by [MATConfig](../MAT/config_loader.py).

The configuration loader reads the private YAML file; some accessors additionally
support environment-variable overrides. Check each accessor rather than assuming
every field has the same override behavior. Keep private configuration and keys
out of commits.

Live execution contacts external services and may incur model/search charges.
The stored report examples are available without doing this. A fiscal-year argument
does not by itself establish that every external source is point-in-time safe.

## Verification status

- Source, schemas, and stored report artifacts are available for inspection.
- `MAT/tests/verify_*` scripts exercise external-service paths and require setup.
- Some older standalone tests still reference pre-refactor fields and need updating.
- There is no current CI evidence establishing a fresh end-to-end live run.

The next reproducibility milestone is a compatible dependency lock, current offline
contract tests, and a documented live acceptance run with explicit source dates.
