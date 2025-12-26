# MAT Framework Usage Guide - Scheme C Implementation

## 🎉 Quick Start

### Installation Requirements

```bash
# Install MetaGPT
pip install metagpt

# Install required dependencies
pip install pydantic tavily-python

# Configure Tavily API (for news search)
export TAVILY_API_KEY="your-api-key-here"
```

### Basic Usage

```python
import asyncio
from MAT import (
    InvestmentEnvironment,
    AlphaStrategist,
    SentimentAnalyst,
    FAReport,
    TAReport,
    SAReport,
    SignalIntensity,
    MarketEvent
)
from MAT.actions import PublishFAReport, PublishTAReport, PublishSAReport
from metagpt.schema import Message

async def analyze_stock(ticker: str):
    # 1. Create environment
    env = InvestmentEnvironment()
    env.set_ticker(ticker)
    
    # 2. Create agents
    sa = SentimentAnalyst()
    as_agent = AlphaStrategist()
    
    # 3. Configure agents
    sa.set_ticker(ticker)
    sa.set_env(env)
    as_agent.set_ticker(ticker)
    as_agent.set_env(env)
    
    # 4. Publish analyst reports
    # (In production, RA and TA would generate these)
    fa_report = FAReport(
        ticker=ticker,
        revenue_growth_yoy=0.25,
        gross_margin=0.45,
        fcf_growth=0.15,
        guidance_sentiment=0.6,
        key_risks=["Competition", "Regulation"],
        is_growth_healthy=True
    )
    
    ta_report = TAReport(
        ticker=ticker,
        rsi_14=32.0,
        bb_lower_touch=True,
        price_to_ma200_dist=-0.12,
        volatility_atr=2.5,
        technical_signal=SignalIntensity.BUY
    )
    
    sa_report = SAReport(
        ticker=ticker,
        sentiment_score=-0.3,  # Negative - creates conflict
        impactful_events=[MarketEvent.REGULATORY_ACTION],
        top_keywords=["concern", "risk", "delay"],
        news_summary="Recent regulatory concerns weighing on sentiment"
    )
    
    # Publish reports
    env.update_fa_report(fa_report)
    env.update_ta_report(ta_report)
    env.update_sa_report(sa_report)
    
    # Publish as messages
    env.publish_message(Message(
        content=fa_report.model_dump_json(),
        role="RA",
        cause_by=PublishFAReport
    ))
    env.publish_message(Message(
        content=ta_report.model_dump_json(),
        role="TA",
        cause_by=PublishTAReport
    ))
    env.publish_message(Message(
        content=sa_report.model_dump_json(),
        role="SA",
        cause_by=PublishSAReport
    ))
    
    # 5. Trigger Alpha Strategist analysis
    await as_agent._observe()
    result = await as_agent._act()
    
    # 6. If investigation requested, SA responds
    if result:
        await sa._observe()
        await sa._act()
        
        # 7. AS makes final decision
        await as_agent._observe()
        await as_agent._act()
    
    # 8. Get final decision
    decision = env.trading_state.final_decision
    return decision

# Run
decision = asyncio.run(analyze_stock("AAPL"))
print(f"Action: {decision.final_action.value}")
print(f"Confidence: {decision.confidence_score}%")
```

## 🔧 Component Overview

### 1. Alpha Strategist (AS)

**Purpose**: Synthesize multi-dimensional analysis and detect conflicts

**Key Methods**:
- `_detect_conflict()`: Identify signal misalignments
- `_calculate_importance_level()`: Determine investigation priority
- `_request_investigation()`: Trigger deep dive
- `_finalize_decision()`: Make final trading decision

**Watches**:
- `PublishFAReport`
- `PublishTAReport`
- `PublishSAReport`
- `PublishInvestigationReport`

**Publishes**:
- `RequestInvestigation` (when conflict detected)
- `PublishStrategyDecision` (final decision)

### 2. Sentiment Analyst (SA)

**Purpose**: Analyze news sentiment with normal and deep dive modes

**Key Methods**:
- `_normal_analysis()`: Regular sentiment scoring
- `_deep_dive_investigation()`: Targeted conflict investigation
- `_search_news()`: Use Tavily for news gathering
- `_analyze_sentiment_with_llm()`: LLM-based sentiment analysis

**Watches**:
- `StartAnalysis` (normal trigger)
- `RequestInvestigation` (deep dive trigger)

**Publishes**:
- `PublishSAReport` (normal mode)
- `PublishInvestigationReport` (deep dive mode)

## 📊 Workflow Patterns

### Pattern 1: No Conflict (Fast Path)

```
FA Report (bullish) ──┐
TA Report (bullish) ──┼──> AS Synthesis ──> StrategyDecision (BUY)
SA Report (bullish) ──┘
                            ⏱️ Fast: ~2-3 seconds
```

### Pattern 2: Conflict with Resolution

```
FA Report (bullish) ──┐
TA Report (bullish) ──┼──> AS Detect Conflict ──> InvestigationRequest
SA Report (negative) ─┘           ↓
                                  ↓
                         SA Deep Dive Search
                                  ↓
                         InvestigationReport (revised: positive)
                                  ↓
                         AS Final Decision (BUY with context)
                            ⏱️ Medium: ~5-8 seconds
```

### Pattern 3: Conflict Unresolved (Safe-First)

```
FA Report (bullish) ──┐
TA Report (bullish) ──┼──> AS Detect Conflict ──> InvestigationRequest
SA Report (negative) ─┘           ↓
                                  ↓
                         SA Deep Dive Search
                                  ↓
                         InvestigationReport (revised: still negative)
                                  ↓
                         AS Retry? (if retries < max_retries)
                            ├─ Yes ──> Loop back
                            └─ No ───> Safe-First Decision (NEUTRAL)
                            ⏱️ Slow: ~10-15 seconds
```

## 🎯 Configuration Examples

### Conservative Settings (Minimize Risk)

```python
# In alpha_strategist.py
def _calculate_importance_level(self, state):
    # Lower threshold for investigations
    if revenue_growth > 0.15:  # Was 0.3
        return (2, 3)  # More retries
    else:
        return (1, 2)

# Higher confidence requirement
if conflict_unresolved:
    confidence = 30.0  # Was 40.0, more conservative
```

### Aggressive Settings (Maximize Opportunity)

```python
# In alpha_strategist.py
def _detect_conflict(self, state):
    # Higher tolerance for negative sentiment
    is_sentiment_negative = sa.sentiment_score < -0.2  # Was 0.1
    
# In alpha_strategist.py
if conflict_unresolved:
    # Still take position if FA/TA very strong
    if fa.revenue_growth_yoy > 0.4:
        final_action = SignalIntensity.BUY
        confidence = 55.0
    else:
        final_action = SignalIntensity.NEUTRAL
        confidence = 40.0
```

## 🧪 Testing

### Unit Tests

```python
import pytest
from MAT import InvestmentEnvironment, AlphaStrategist, TradingState
from MAT.schemas import FAReport, TAReport, SAReport, SignalIntensity

def test_conflict_detection():
    env = InvestmentEnvironment()
    env.set_ticker("TEST")
    as_agent = AlphaStrategist()
    as_agent.set_env(env)
    
    # Create conflicting state
    state = TradingState(
        current_ticker="TEST",
        fa_data=FAReport(
            ticker="TEST",
            revenue_growth_yoy=0.35,
            is_growth_healthy=True,
            ...
        ),
        ta_data=TAReport(
            ticker="TEST",
            technical_signal=SignalIntensity.BUY,
            ...
        ),
        sa_data=SAReport(
            ticker="TEST",
            sentiment_score=-0.4,
            ...
        )
    )
    
    # Test conflict detection
    conflict = as_agent._detect_conflict(state)
    assert conflict is not None
    assert "CONFLICT" in conflict
    
    # Test importance calculation
    importance, max_retries = as_agent._calculate_importance_level(state)
    assert importance == 2
    assert max_retries == 2

@pytest.mark.asyncio
async def test_investigation_workflow():
    env = InvestmentEnvironment()
    env.set_ticker("TEST")
    
    sa = SentimentAnalyst()
    as_agent = AlphaStrategist()
    
    # Setup and test full workflow
    # ... (see scheme_c_demo.py for complete example)
```

### Integration Tests

Run the demo:

```bash
cd /Users/richsion/Desktop/MetaGPT/MetaGPT-Ewan
python -m MAT.scheme_c_demo
```

Expected output:
- Conflict detection log
- Investigation request
- Deep dive analysis
- Final decision with logic chain

## 📈 Monitoring & Logging

### Enable Detailed Logging

```python
from metagpt.logs import logger
import sys

# Set to DEBUG for detailed logs
logger.remove()
logger.add(sys.stderr, level="DEBUG")
```

### Key Log Markers

Look for these in logs:
- `⚠️ CONFLICT DETECTED` - AS found signal mismatch
- `🔍 DEEP DIVE MODE activated` - SA starting investigation
- `📊 Deep Dive Results` - Investigation findings
- `🛡️ Safe-First Decision` - Conflict unresolved, using conservative approach
- `✅ Aligned Decision` - No conflict, normal synthesis

### Metrics to Track

```python
# Track in production
metrics = {
    "conflict_rate": conflicts / total_analyses,
    "investigation_trigger_rate": investigations / conflicts,
    "resolution_rate": resolved / investigations,
    "safe_first_rate": safe_decisions / total_decisions,
    "avg_processing_time_seconds": sum(times) / len(times)
}
```

## 🚨 Common Issues & Solutions

### Issue 1: SA Not Finding News

**Problem**: `_search_news()` returns empty list

**Solutions**:
1. Check Tavily API key is configured
2. Verify internet connection
3. Check ticker symbol is valid
4. Add fallback to mock data for testing

```python
# In sentiment_analyst.py
if not news_data:
    logger.warning("Using mock data for testing")
    news_data = self._get_mock_news(ticker)
```

### Issue 2: Investigation Loop Not Terminating

**Problem**: AS keeps requesting investigations

**Solution**: Verify retry tracking

```python
# Debug retry counts
logger.info(f"Current retry: {self._retry_counts.get(ticker, 0)}")
logger.info(f"Max retries: {max_retries}")
```

### Issue 3: LLM Response Parsing Fails

**Problem**: `_parse_llm_sentiment_response()` raises exception

**Solution**: Add robust error handling

```python
try:
    data = json.loads(response)
except json.JSONDecodeError:
    # Extract JSON from markdown code blocks
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1))
```

## 🎓 Best Practices

### 1. Importance Threshold Tuning

Start conservative, adjust based on backtesting:

```python
# Week 1: Conservative (many investigations)
if revenue_growth > 0.15:
    importance = 2

# Week 2-4: Monitor resolution rates
# Adjust thresholds based on how often investigations help

# Production: Optimized
if revenue_growth > 0.25:
    importance = 2
```

### 2. Sentiment Threshold Calibration

Calibrate based on your sentiment model:

```python
# If your sentiment scores are typically mild (-0.3 to 0.3)
is_sentiment_negative = sa.sentiment_score < 0.0

# If your sentiment scores are more extreme (-0.8 to 0.8)
is_sentiment_negative = sa.sentiment_score < 0.15
```

### 3. LLM Prompt Engineering

Optimize prompts for your LLM:

```python
# For GPT-4
prompt = "You are an expert analyst. Provide precise JSON..."

# For smaller models (Llama, Claude)
prompt = "Analyze sentiment. Output format:\n{example}\nYour analysis:"
```

### 4. Caching News Searches

Avoid redundant API calls:

```python
# In sentiment_analyst.py
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._news_cache = {}  # ticker -> (timestamp, news_data)
    self._cache_ttl = 300  # 5 minutes

async def _search_news(self, ticker, ...):
    # Check cache first
    if ticker in self._news_cache:
        cached_time, cached_data = self._news_cache[ticker]
        if time.time() - cached_time < self._cache_ttl:
            return cached_data
    
    # Fetch fresh data
    news_data = await self._search_engine.run(...)
    self._news_cache[ticker] = (time.time(), news_data)
    return news_data
```

## 📚 Next Steps

1. **Implement RA and TA**: Create Research Analyst and Technical Analyst
2. **Backtesting**: Test Scheme C on historical data
3. **Production Integration**: Connect to real trading systems
4. **Monitoring Dashboard**: Build visualization for conflict resolution metrics
5. **Multi-Ticker**: Extend to analyze multiple stocks in parallel

## 🤝 Support

- **Documentation**: See `SCHEME_C_README.md` for architecture details
- **Examples**: Check `scheme_c_demo.py` for working examples
- **Issues**: Report bugs or questions via GitHub issues

---

**Version**: 1.0.0
**Last Updated**: December 27, 2025
**Author**: Ewan Su

