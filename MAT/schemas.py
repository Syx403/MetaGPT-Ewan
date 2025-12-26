from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# --- 1. Signal & Event Enums ---

class SignalIntensity(Enum):
    """Standardized signals for trading decisions."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class MarketEvent(Enum):
    """Categories of market-moving events."""
    EARNINGS_CALL = "EARNINGS_CALL"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    REGULATORY_ACTION = "REGULATORY_ACTION"
    ANALYST_UPGRADE = "ANALYST_UPGRADE"
    MACRO_ECONOMIC = "MACRO_ECONOMIC"
    NONE = "NONE"

# --- 2. Agent Output Protocols ---

class FAReport(BaseModel):
    """Protocol for Research Analyst (RA) focusing on Fundamental Analysis."""
    ticker: str
    revenue_growth_yoy: float = Field(description="Year-over-Year Revenue Growth Rate")
    gross_margin: float = Field(description="Gross Profit Margin")
    fcf_growth: float = Field(description="Free Cash Flow growth status")
    guidance_sentiment: float = Field(ge=-1, le=1, description="Management guidance sentiment score from -1 to 1")
    key_risks: List[str] = Field(default_factory=list, description="Top 3 risk factors extracted via RAG")
    is_growth_healthy: bool = Field(description="Whether the company meets the fundamental threshold")

class TAReport(BaseModel):
    """Protocol for Technical Analyst (TA) focusing on Mean Reversion."""
    ticker: str
    rsi_14: float = Field(description="Relative Strength Index over 14 periods")
    bb_lower_touch: bool = Field(description="Whether price touched/broke the lower Bollinger Band")
    price_to_ma200_dist: float = Field(description="Distance percentage from the 200-day Moving Average")
    volatility_atr: float = Field(description="Average True Range for dynamic stop-loss setting")
    technical_signal: SignalIntensity

class SAReport(BaseModel):
    """Protocol for Sentiment Analyst (SA) focusing on News & Catalysts."""
    ticker: str
    sentiment_score: float = Field(ge=-1, le=1, description="Aggregated sentiment score from news")
    impactful_events: List[MarketEvent] = Field(description="List of detected major market events")
    top_keywords: List[str]
    news_summary: str

# --- 3. Final Strategy & Decision ---

class StrategyDecision(BaseModel):
    """Protocol for Alpha Strategist (AS) final output."""
    ticker: str
    final_action: SignalIntensity
    confidence_score: float = Field(ge=0, le=100)
    logic_chain: List[str] = Field(description="Step-by-step reasoning for the decision")
    risk_notes: str = Field(description="Risk management constraints for RiskMgr")
    suggested_module: str = Field(description="The name of the execution module from Scheme A")

# --- 4. Shared Global State ---

class TradingState(BaseModel):
    """Global context maintained in the Environment."""
    current_ticker: str
    fa_data: Optional[FAReport] = None
    ta_data: Optional[TAReport] = None
    sa_data: Optional[SAReport] = None
    final_decision: Optional[StrategyDecision] = None

# --- Updated Schemas for Scheme C (Dynamic Inquiry) ---

class InvestigationRequest(BaseModel):
    """Protocol for AS to request a deep dive investigation."""
    ticker: str
    target_agent: str = "SA"
    context_issue: str = Field(description="The specific conflicting issue to investigate")
    current_retry: int = Field(default=0, description="Current number of retries")
    max_retries: int = Field(default=1, description="Max retries allowed based on importance")
    importance_level: int = Field(default=1, description="1 for Normal, 2 for High Importance")

class InvestigationReport(BaseModel):
    """Protocol for SA to provide deep dive results."""
    ticker: str
    detailed_findings: str = Field(description="Results of the deep search")
    revised_sentiment_score: float = Field(ge=-1, le=1)
    is_ambiguity_resolved: bool = Field(description="Whether the conflict is cleared")