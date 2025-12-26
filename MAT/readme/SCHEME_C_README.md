# Scheme C: Active Inquiry - Implementation Guide

## 🎯 Overview

**Scheme C (Active Inquiry)** is an advanced agent coordination pattern that enables dynamic conflict resolution through targeted investigations. When the Alpha Strategist (AS) detects conflicts between fundamental/technical signals and sentiment analysis, it can request deep dive investigations from the Sentiment Analyst (SA) to resolve ambiguities.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCHEME C WORKFLOW                             │
└─────────────────────────────────────────────────────────────────┘

1. INITIAL ANALYSIS
   ├─ FA Report → Bullish (growth_healthy=True, revenue_growth>20%)
   ├─ TA Report → Bullish (signal=BUY, RSI oversold)
   └─ SA Report → Negative (sentiment_score<0)
                  ↓
            ⚠️ CONFLICT!
                  ↓
2. CONFLICT DETECTION (Alpha Strategist)
   ├─ Detect: FA/TA bullish BUT SA negative
   ├─ Calculate importance_level:
   │  ├─ revenue_growth > 30% → importance=2, max_retries=2
   │  └─ revenue_growth ≤ 30% → importance=1, max_retries=1
   └─ Check retry_count < max_retries
                  ↓
3a. REQUEST INVESTIGATION (if retries available)
    ├─ AS publishes InvestigationRequest
    │  ├─ ticker: "NVDA"
    │  ├─ context_issue: "Conflict description"
    │  ├─ importance_level: 1 or 2
    │  └─ current_retry: 0, 1, ...
    └─ SA observes request
                  ↓
4. DEEP DIVE INVESTIGATION (Sentiment Analyst)
   ├─ Targeted news search (focus on conflict)
   ├─ LLM analysis with conflict context
   ├─ Generate InvestigationReport:
   │  ├─ detailed_findings: "Explanation..."
   │  ├─ revised_sentiment_score: -1 to +1
   │  └─ is_ambiguity_resolved: true/false
   └─ Publish InvestigationReport
                  ↓
5. UPDATE & RE-EVALUATE (Alpha Strategist)
   ├─ Receive InvestigationReport
   ├─ Update SA data with revised sentiment
   └─ Re-check for conflicts
                  ↓
3b. LOOP OR FINALIZE
    ├─ If conflict persists AND retries available → GO TO 3a
    └─ If conflict resolved OR retries exhausted → GO TO 6
                  ↓
6. FINAL DECISION
   ├─ If signals aligned → Weighted synthesis
   └─ If conflict unresolved → 🛡️ SAFE-FIRST (NEUTRAL)
```

## 🔑 Key Components

### 1. InvestigationRequest Schema

```python
class InvestigationRequest(BaseModel):
    ticker: str
    target_agent: str = "SA"  # Target for investigation
    context_issue: str  # Specific conflict description
    current_retry: int = 0
    max_retries: int = 1  # Based on importance
    importance_level: int = 1  # 1=Normal, 2=High
```

### 2. InvestigationReport Schema

```python
class InvestigationReport(BaseModel):
    ticker: str
    detailed_findings: str  # Deep dive results
    revised_sentiment_score: float  # Updated sentiment
    is_ambiguity_resolved: bool  # Conflict cleared?
```

### 3. Alpha Strategist Logic

**Conflict Detection:**
```python
def _detect_conflict(self, state: TradingState) -> Optional[str]:
    is_fa_bullish = state.fa_data.is_growth_healthy
    is_ta_bullish = state.ta_data.technical_signal in [BUY, STRONG_BUY]
    is_sentiment_negative = state.sa_data.sentiment_score < 0.1
    
    if is_fa_bullish and is_ta_bullish and is_sentiment_negative:
        return "CONFLICT: Bullish fundamentals/technicals vs negative sentiment"
    return None
```

**Importance Calculation:**
```python
def _calculate_importance_level(self, state: TradingState) -> tuple[int, int]:
    revenue_growth = state.fa_data.revenue_growth_yoy
    
    if revenue_growth > 0.3:  # 30% threshold
        return (2, 2)  # importance=2, max_retries=2
    else:
        return (1, 1)  # importance=1, max_retries=1
```

**Decision Logic:**
```python
async def _make_decision(self, state: TradingState) -> Message:
    conflict = self._detect_conflict(state)
    
    if conflict:
        importance, max_retries = self._calculate_importance_level(state)
        
        if current_retry < max_retries:
            # Request investigation
            return await self._request_investigation(...)
        else:
            # Retries exhausted: Safe-First decision
            return await self._finalize_decision(conflict_unresolved=True)
    else:
        # No conflict: Normal synthesis
        return await self._finalize_decision(conflict_unresolved=False)
```

### 4. Sentiment Analyst Dual-Mode

**Mode Detection:**
```python
async def _act(self) -> Message:
    latest_msg = self.get_memories()[-1]
    
    if latest_msg.cause_by == RequestInvestigation:
        # Deep dive mode
        return await self._deep_dive_investigation(latest_msg)
    elif latest_msg.cause_by == StartAnalysis:
        # Normal mode
        return await self._normal_analysis()
```

**Deep Dive Investigation:**
```python
async def _deep_dive_investigation(self, request_msg: Message) -> Message:
    investigation_req = InvestigationRequest(**json.loads(request_msg.content))
    
    # Targeted search focusing on conflict
    search_query = self._build_investigation_query(
        investigation_req.ticker,
        investigation_req.context_issue
    )
    news_data = await self._search_news(custom_query=search_query, max_results=10)
    
    # LLM analysis with conflict context
    report = await self._analyze_investigation_with_llm(
        ticker=investigation_req.ticker,
        news_data=news_data,
        context_issue=investigation_req.context_issue,
        importance_level=investigation_req.importance_level
    )
    
    return await self.publish_message(report, PublishInvestigationReport)
```

## 📊 Usage Example

### Scenario: High-Growth Company with Sentiment Conflict

```python
from MAT import (
    InvestmentEnvironment,
    AlphaStrategist,
    SentimentAnalyst,
    FAReport,
    TAReport,
    SAReport,
    SignalIntensity
)

# Setup
env = InvestmentEnvironment()
env.set_ticker("NVDA")

as_agent = AlphaStrategist()
sa = SentimentAnalyst()

# Configure agents
as_agent.set_env(env)
sa.set_env(env)

# Publish conflicting reports
fa_report = FAReport(
    ticker="NVDA",
    revenue_growth_yoy=0.35,  # 35% growth → importance_level=2
    is_growth_healthy=True,   # Bullish
    ...
)

ta_report = TAReport(
    ticker="NVDA",
    technical_signal=SignalIntensity.BUY,  # Bullish
    ...
)

sa_report = SAReport(
    ticker="NVDA",
    sentiment_score=-0.4,  # Negative → CONFLICT!
    ...
)

# Trigger workflow
# AS will detect conflict and request investigation
await as_agent._act()  # → InvestigationRequest

# SA performs deep dive
await sa._act()  # → InvestigationReport

# AS makes final decision
await as_agent._act()  # → StrategyDecision
```

## 🎛️ Configuration Options

### Importance Thresholds

Adjust in `alpha_strategist.py`:

```python
def _calculate_importance_level(self, state: TradingState):
    revenue_growth = state.fa_data.revenue_growth_yoy
    
    # Customize thresholds
    if revenue_growth > 0.3:  # High growth threshold
        return (2, 2)  # High importance, 2 retries
    elif revenue_growth > 0.15:  # Medium growth
        return (1, 1)  # Normal importance, 1 retry
    else:
        return (0, 0)  # Low growth, no retries
```

### Conflict Detection Sensitivity

Adjust sentiment threshold:

```python
def _detect_conflict(self, state: TradingState):
    # More sensitive (detect conflicts earlier)
    is_sentiment_negative = state.sa_data.sentiment_score < 0.2
    
    # Less sensitive (only clear negative sentiment)
    is_sentiment_negative = state.sa_data.sentiment_score < -0.1
```

### Safe-First Confidence

Adjust confidence levels for unresolved conflicts:

```python
if conflict_unresolved:
    final_action = SignalIntensity.NEUTRAL
    confidence = 40.0  # Low confidence (adjust as needed)
```

## 🧪 Testing Scenarios

### Test 1: Conflict with Resolution

```python
# Input: FA/TA bullish, SA negative
# Investigation: Finds positive hidden catalyst
# Expected: revised_sentiment_score > 0, is_ambiguity_resolved=True
# Output: BUY decision with good confidence
```

### Test 2: Conflict without Resolution

```python
# Input: FA/TA bullish, SA negative
# Investigation: Confirms negative sentiment is valid
# Expected: revised_sentiment_score still < 0, is_ambiguity_resolved=False
# Output: NEUTRAL decision (Safe-First)
```

### Test 3: High Importance (2 retries)

```python
# Input: revenue_growth > 30%, persistent conflict
# Expected: Two investigation attempts before Safe-First decision
```

### Test 4: No Conflict

```python
# Input: FA/TA/SA all aligned (bullish or bearish)
# Expected: No investigation, direct synthesis decision
```

## 📈 Performance Metrics

Track these metrics for Scheme C effectiveness:

1. **Investigation Trigger Rate**: % of analyses requiring investigation
2. **Resolution Rate**: % of conflicts resolved by investigation
3. **Average Retries**: Mean number of investigation cycles
4. **Safe-First Rate**: % of decisions using Safe-First approach
5. **Sentiment Revision Magnitude**: |original_sentiment - revised_sentiment|

## 🔧 Troubleshooting

### Issue: Infinite Investigation Loops

**Solution**: The framework prevents this with:
```python
self._retry_counts[ticker] += 1
if current_retry < max_retries:
    # Request investigation
else:
    # Force final decision
```

### Issue: No Investigation Triggered

**Check:**
1. Conflict detection logic: Are thresholds correct?
2. Reports present: FA, TA, SA all received?
3. Retry count: Not already exhausted?

### Issue: SA Not Responding to Investigation

**Check:**
1. SA is watching `RequestInvestigation` action
2. Message routing is correct
3. Tavily API credentials configured

## 🚀 Advanced Features

### Custom Conflict Types

Extend conflict detection for other patterns:

```python
def _detect_valuation_conflict(self, state):
    # High valuation + negative sentiment
    if state.fa_data.pe_ratio > 50 and state.sa_data.sentiment_score < 0:
        return "Valuation concern conflict"

def _detect_momentum_conflict(self, state):
    # Strong momentum + negative news
    if state.ta_data.rsi_14 > 70 and state.sa_data.sentiment_score < -0.3:
        return "Overbought with negative news conflict"
```

### Multi-Agent Investigation

Request investigations from multiple agents:

```python
investigation_request = InvestigationRequest(
    ticker=ticker,
    target_agent="SA,FA",  # Multiple targets
    context_issue=conflict
)
```

### Adaptive Importance

Dynamic importance based on multiple factors:

```python
def _calculate_adaptive_importance(self, state):
    score = 0
    if state.fa_data.revenue_growth_yoy > 0.3:
        score += 2
    if state.ta_data.rsi_14 < 30:  # Deep oversold
        score += 1
    if len(state.sa_data.impactful_events) > 2:
        score += 1
    
    importance = min(3, score)
    max_retries = importance
    return importance, max_retries
```

## 📚 References

- **Paper**: "Active Inquiry in Multi-Agent Systems" (fictional reference for demo)
- **Pattern**: Dynamic Agent Coordination with Feedback Loops
- **Related**: Scheme A (Voting), Scheme B (Sequential), Scheme C (Active Inquiry)

---

**Implementation Status**: ✅ Complete
**Last Updated**: December 27, 2025
**Author**: Ewan Su

