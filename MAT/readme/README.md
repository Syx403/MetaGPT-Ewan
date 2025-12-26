# Multi-Agent Trading (MAT) Framework

A MetaGPT-based investment analysis system using publish-subscribe agent coordination.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   InvestmentEnvironment                      │
│  (maintains TradingState, handles message propagation)       │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  StartAnalysis msg  │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┬──────────────┬─────────────┐
        ▼                     ▼              ▼             │
  ┌─────────┐          ┌─────────┐    ┌─────────┐         │
  │   RA    │          │   TA    │    │   SA    │         │
  │ (Fund.) │          │ (Tech.) │    │ (Sent.) │         │
  └────┬────┘          └────┬────┘    └────┬────┘         │
       │                    │              │               │
       ▼                    ▼              ▼               │
  FAReport             TAReport       SAReport             │
       │                    │              │               │
       └────────────────────┴──────────────┘               │
                            │                              │
                            ▼                              │
                    ┌───────────────┐                      │
                    │      AS       │                      │
                    │  (Strategist) │                      │
                    └───────┬───────┘                      │
                            │                              │
                            ▼                              │
                   StrategyDecision ─────────────────────►─┘
                   (updates TradingState)
```

## Components

### 1. Core Schemas (`schemas.py`)
- **Enums**: `SignalIntensity`, `MarketEvent`
- **Reports**: `FAReport`, `TAReport`, `SAReport`, `StrategyDecision`
- **State**: `TradingState` - shared global context

### 2. Environment (`environment.py`)
- **InvestmentEnvironment**: Manages TradingState and message routing
- Methods:
  - `set_ticker()`: Initialize analysis for a ticker
  - `update_*_report()`: Update state with agent reports
  - `is_analysis_complete()`: Check if all reports collected

### 3. Base Agent (`roles/base_agent.py`)
- **BaseInvestmentAgent**: Abstract base for all analysts
- Key methods:
  - `_observe()`: Filter messages by ticker context
  - `publish_message()`: Wrap Pydantic reports into Messages
  - `get_trading_state_context()`: Access shared state
  - `_act()`: Abstract method for agent logic (implemented by subclasses)

### 4. Actions (`actions.py`)
- **StartAnalysis**: Trigger workflow
- **PublishFAReport**: RA publishes fundamental analysis
- **PublishTAReport**: TA publishes technical analysis
- **PublishSAReport**: SA publishes sentiment analysis
- **PublishStrategyDecision**: AS publishes final decision

## Workflow

1. **Initialization**
   ```python
   env = InvestmentEnvironment()
   env.set_ticker("AAPL")
   ```

2. **Start Analysis**
   - Orchestrator publishes `StartAnalysis` message
   - RA, TA, SA observe and react

3. **Parallel Analysis**
   - RA → analyzes fundamentals → publishes `FAReport`
   - TA → analyzes technicals → publishes `TAReport`
   - SA → analyzes sentiment → publishes `SAReport`

4. **Strategy Synthesis**
   - AS observes `FAReport`, `TAReport`, `SAReport`
   - AS synthesizes → publishes `StrategyDecision`

5. **State Updates**
   - Environment updates `TradingState` as reports arrive
   - All agents can access shared state via `env.trading_state`

## Key Design Patterns

### Publish-Subscribe
Agents use MetaGPT's `_watch()` to subscribe to specific Action types:
```python
self._watch([StartAnalysis])  # RA, TA, SA
self._watch([PublishFAReport, PublishTAReport])  # AS
```

### Typed Communication
All inter-agent communication uses Pydantic models for type safety:
```python
report = FAReport(ticker="AAPL", revenue_growth_yoy=0.15, ...)
await self.publish_message(report, cause_by=PublishFAReport)
```

### Shared State
`TradingState` provides a centralized context accessible to all agents:
```python
state = self.env.trading_state
if state.fa_data and state.ta_data:
    # Both reports available, proceed with synthesis
```

## Next Steps

To implement concrete agents (RA, TA, SA, AS):
1. Inherit from `BaseInvestmentAgent`
2. Implement `_act()` method
3. Define actions to watch in `__init__()`
4. Use LLM prompts or computational logic for analysis
5. Publish typed reports using `publish_message()`

## Example Usage

```python
from MAT import InvestmentEnvironment, BaseInvestmentAgent
from MAT.actions import StartAnalysis

# Create environment
env = InvestmentEnvironment()
env.set_ticker("AAPL")

# Create agents (RA, TA, SA, AS implementations)
# ... agent creation code ...

# Start workflow
team = Team(env=env, roles=[ra, ta, sa, as_agent])
await team.run(idea="Analyze AAPL for potential long position")
```

## File Structure

```
MAT/
├── __init__.py           # Package exports
├── schemas.py            # Pydantic data models
├── environment.py        # InvestmentEnvironment
├── actions.py            # Action definitions
├── roles/
│   ├── __init__.py
│   ├── base_agent.py     # BaseInvestmentAgent
│   ├── ra.py             # Research Analyst (to be implemented)
│   ├── ta.py             # Technical Analyst (to be implemented)
│   ├── sa.py             # Sentiment Analyst (to be implemented)
│   └── as_agent.py       # Alpha Strategist (to be implemented)
└── README.md             # This file
```

