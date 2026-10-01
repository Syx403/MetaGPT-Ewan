# MAT implementation guide

[Project overview, diagram, and report examples](../../README.md) · [Runtime requirements](../../docs/running.md)

The current application is an evidence-oriented, four-role research prototype. `FinancialMetric` and related schemas carry source evidence and qualitative interpretation; older numeric signal examples in this folder predate that refactor.

## Current coordination

1. `ResearchAnalyst`, `TechnicalAnalyst`, and `SentimentAnalyst` publish structured reports.
2. `AlphaStrategist` buffers the reports and calls the `AnalyzeConflict` Action.
3. If a conflict exists and no investigation is stored for the ticker, it requests one SA investigation.
4. With no conflict or with an investigation available, `SynthesizeDecision` creates the final report.

The current request sets `max_retries=1`. The coordinator uses the presence of a stored investigation report to decide when to synthesize; it does not implement the older documentation's revenue-dependent retry policy.

## Authoritative entry points

- [Schemas](../schemas.py)
- [Coordinator](../roles/alpha_strategist.py)
- [Conflict and synthesis Actions](../actions/synthesize_strategy.py)
- [CLI runner](../main.py)

The other documents in this folder are retained as design history and module notes. Check their examples against the current schemas before reuse. The repository's old standalone verification scripts are not presented as a green current test suite.
