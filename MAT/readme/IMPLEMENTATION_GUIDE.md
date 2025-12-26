# MAT Framework Implementation Guide

## 📚 What We've Built

The Multi-Agent Trading (MAT) framework is now ready with the following components:

### ✅ Core Infrastructure

1. **`schemas.py`** - Type-safe data contracts
   - Enums: `SignalIntensity`, `MarketEvent`
   - Reports: `FAReport`, `TAReport`, `SAReport`, `StrategyDecision`
   - State: `TradingState`

2. **`environment.py`** - InvestmentEnvironment
   - Manages shared `TradingState`
   - Handles message propagation
   - Provides state update methods
   - Tracks analysis completion

3. **`roles/base_agent.py`** - BaseInvestmentAgent
   - Abstract base for all analysts
   - Standardized `_observe()` with ticker filtering
   - `publish_message()` helper for Pydantic → Message conversion
   - `get_trading_state_context()` for prompt enrichment
   - Abstract `_act()` method for subclasses

4. **`actions.py`** - Message type definitions
   - `StartAnalysis` - Workflow trigger
   - `PublishFAReport` - RA message type
   - `PublishTAReport` - TA message type
   - `PublishSAReport` - SA message type
   - `PublishStrategyDecision` - AS message type

5. **`example_workflow.py`** - Working demonstration
   - Template implementations of RA, TA, SA, AS
   - Shows the complete workflow execution
   - Demonstrates publish-subscribe pattern

## 🔄 Data Flow Confirmed

```
┌──────────────────────────────────────────────────────────────┐
│                    WORKFLOW SEQUENCE                          │
└──────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ├─ Create InvestmentEnvironment
   ├─ Set ticker (e.g., "AAPL")
   └─ Create agents: RA, TA, SA, AS

2. START TRIGGER
   └─ Publish StartAnalysis message
      ↓
      ├─→ RA observes → _act() → FAReport → env.update_fa_report()
      ├─→ TA observes → _act() → TAReport → env.update_ta_report()
      └─→ SA observes → _act() → SAReport → env.update_sa_report()

3. PARALLEL ANALYSIS (RA, TA, SA run concurrently)
   │
   ├─ RA (Research Analyst)
   │  ├─ Analyzes: Revenue, margins, FCF, guidance, risks
   │  └─ Publishes: FAReport with is_growth_healthy flag
   │
   ├─ TA (Technical Analyst)
   │  ├─ Analyzes: RSI, Bollinger Bands, MA distance, ATR
   │  └─ Publishes: TAReport with technical_signal
   │
   └─ SA (Sentiment Analyst)
      ├─ Analyzes: News sentiment, events, keywords
      └─ Publishes: SAReport with sentiment_score

4. STRATEGY SYNTHESIS
   │
   └─ AS (Alpha Strategist)
      ├─ Observes: FAReport, TAReport, SAReport
      ├─ Waits: env.is_analysis_complete() == True
      ├─ Synthesizes: All three reports
      └─ Publishes: StrategyDecision
         ├─ final_action: STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL
         ├─ confidence_score: 0-100
         ├─ logic_chain: Step-by-step reasoning
         ├─ risk_notes: Risk management guidance
         └─ suggested_module: Execution strategy name

5. FINAL STATE
   └─ TradingState contains all reports + final decision
```

## 🎯 Key Design Decisions Explained

### 1. **Publish-Subscribe Pattern**
- **Why?** Loose coupling between agents
- **How?** Agents use `_watch([ActionType])` to subscribe
- **Benefit?** Agents can be added/removed without modifying others

### 2. **Pydantic Models for Messages**
- **Why?** Type safety and validation
- **How?** `publish_message()` serializes models to JSON
- **Benefit?** Catch data errors early, self-documenting APIs

### 3. **Centralized TradingState**
- **Why?** Single source of truth
- **How?** Environment maintains state, agents read/write
- **Benefit?** Prevents inconsistencies, enables state inspection

### 4. **Ticker-Based Filtering**
- **Why?** Support multi-ticker analysis later
- **How?** `_observe()` filters messages by current_ticker
- **Benefit?** Scalable to parallel ticker analysis

### 5. **Abstract BaseInvestmentAgent**
- **Why?** Code reuse and consistency
- **How?** Concrete agents inherit and implement `_act()`
- **Benefit?** Standardized interface, easier testing

## 📝 Implementation Checklist for Concrete Agents

To implement a real agent (e.g., RA with LLM), follow these steps:

### Research Analyst (RA) Example

```python
class ResearchAnalyst(BaseInvestmentAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Alice",
            profile="Research Analyst",
            goal="Analyze company fundamentals",
            **kwargs
        )
        self._watch([StartAnalysis])  # Subscribe to trigger
        self.set_actions([PublishFAReport])  # Set action type
    
    async def _act(self) -> Message:
        # Step 1: Get ticker from environment
        ticker = self.env.trading_state.current_ticker
        
        # Step 2: Fetch data (API calls)
        financial_data = await self.fetch_financials(ticker)
        
        # Step 3: Analyze with LLM
        prompt = self._build_fa_prompt(financial_data)
        analysis = await self._aask(prompt)
        
        # Step 4: Parse LLM output into FAReport
        report = self._parse_fa_report(analysis, ticker)
        
        # Step 5: Update environment
        self.env.update_fa_report(report)
        
        # Step 6: Publish message
        return await self.publish_message(
            report=report,
            cause_by=PublishFAReport
        )
    
    def _build_fa_prompt(self, data) -> str:
        return f"""
        You are a fundamental analyst. Analyze the following data:
        {data}
        
        Provide:
        1. Revenue growth YoY (as decimal)
        2. Gross margin (as decimal)
        3. FCF growth status
        4. Guidance sentiment (-1 to 1)
        5. Top 3 risk factors
        6. Is growth healthy? (yes/no)
        
        Output as JSON matching this schema:
        {FAReport.model_json_schema()}
        """
```

### Technical Analyst (TA) Implementation Hints

- Use `yfinance` or `pandas-ta` for indicator calculation
- Calculate RSI, Bollinger Bands, MA programmatically
- Use LLM only for interpretation/signal generation
- Return structured TAReport

### Sentiment Analyst (SA) Implementation Hints

- Scrape news with `requests` + `BeautifulSoup`
- Use LLM for sentiment scoring (batch multiple articles)
- Detect events with pattern matching or LLM classification
- Extract keywords with TF-IDF or LLM

### Alpha Strategist (AS) Implementation Hints

- Wait for all reports: `env.is_analysis_complete()`
- Use rich prompt with all three reports
- Ask LLM for structured reasoning
- Parse into StrategyDecision with logic_chain

## 🧪 Testing Strategy

### Unit Tests
```python
# Test individual components
def test_trading_state_initialization():
    state = TradingState(current_ticker="AAPL")
    assert state.fa_data is None
    assert state.final_decision is None

def test_environment_report_updates():
    env = InvestmentEnvironment()
    env.set_ticker("AAPL")
    report = FAReport(ticker="AAPL", ...)
    env.update_fa_report(report)
    assert env.trading_state.fa_data == report
```

### Integration Tests
```python
# Test agent interactions
async def test_workflow():
    env = InvestmentEnvironment()
    env.set_ticker("AAPL")
    
    ra = ResearchAnalyst()
    ra.set_env(env)
    
    # Trigger and verify
    await ra._act()
    assert env.trading_state.fa_data is not None
```

### End-to-End Tests
```python
# Test complete workflow
async def test_full_analysis():
    env = InvestmentEnvironment()
    team = Team(env=env, roles=[ra, ta, sa, as_agent])
    await team.run(idea="Analyze AAPL")
    
    assert env.trading_state.final_decision is not None
    assert env.trading_state.final_decision.final_action in SignalIntensity
```

## 🚀 Next Steps

### Phase 1: Implement Core Agents (Current Priority)
- [ ] Implement real RA with financial API integration
- [ ] Implement real TA with technical indicators
- [ ] Implement real SA with news scraping
- [ ] Implement real AS with multi-report synthesis

### Phase 2: Data Integration
- [ ] Integrate yfinance for market data
- [ ] Add news API (Alpha Vantage, NewsAPI)
- [ ] Add earnings call transcripts (Seeking Alpha)
- [ ] Add social sentiment (Twitter/Reddit via APIs)

### Phase 3: LLM Integration
- [ ] Design prompts for each agent
- [ ] Implement structured output parsing
- [ ] Add retry logic for LLM failures
- [ ] Optimize token usage

### Phase 4: Risk Management
- [ ] Implement RiskManager agent
- [ ] Add position sizing logic
- [ ] Add stop-loss calculation
- [ ] Portfolio constraint checking

### Phase 5: Execution & Backtesting
- [ ] Implement execution modules (Scheme A)
- [ ] Add backtesting framework
- [ ] Performance metrics calculation
- [ ] Paper trading integration

## 📊 Example Usage

```python
from MAT import InvestmentEnvironment
from MAT.roles.ra import ResearchAnalyst
from MAT.roles.ta import TechnicalAnalyst
from MAT.roles.sa import SentimentAnalyst
from MAT.roles.as_agent import AlphaStrategist
from metagpt.team import Team

async def analyze_stock(ticker: str):
    # Setup
    env = InvestmentEnvironment()
    env.set_ticker(ticker)
    
    # Create team
    team = Team(
        env=env,
        roles=[
            ResearchAnalyst(),
            TechnicalAnalyst(),
            SentimentAnalyst(),
            AlphaStrategist()
        ]
    )
    
    # Run analysis
    await team.run(idea=f"Analyze {ticker} for investment opportunity")
    
    # Get result
    decision = env.trading_state.final_decision
    print(f"Action: {decision.final_action.value}")
    print(f"Confidence: {decision.confidence_score}%")
    return decision

# Execute
decision = await analyze_stock("AAPL")
```

## 🔍 Debugging Tips

1. **Enable verbose logging**:
   ```python
   from metagpt.logs import logger
   logger.remove()
   logger.add(sys.stderr, level="DEBUG")
   ```

2. **Inspect trading state**:
   ```python
   print(env.get_state_summary())
   ```

3. **Check message buffer**:
   ```python
   print(f"Agent has {len(agent.rc.msg_buffer)} messages")
   ```

4. **Validate reports manually**:
   ```python
   report = FAReport(ticker="AAPL", ...)  # Will raise ValidationError if invalid
   ```

## 📚 References

- **MetaGPT Documentation**: https://docs.deepwisdom.ai/main/en/
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **Trading Strategy Pattern**: Mean Reversion + Multi-Factor Analysis

---

**Status**: ✅ Framework Complete, Ready for Agent Implementation
**Last Updated**: December 26, 2025
**Author**: Ewan Su

