"""
Filename: MetaGPT-Ewan/MAT/example_workflow.py
Created Date: Friday, December 26th 2025
Author: Ewan Su
Description: Example demonstrating the MAT framework workflow pattern.

This is a template showing how concrete agents will be implemented.
The actual RA, TA, SA, and AS agents will follow this pattern.
"""

import asyncio
from typing import Optional
from metagpt.schema import Message
from metagpt.logs import logger

from MAT import (
    InvestmentEnvironment,
    BaseInvestmentAgent,
    FAReport,
    TAReport,
    SAReport,
    StrategyDecision,
    SignalIntensity,
    MarketEvent
)
from MAT.actions import (
    StartAnalysis,
    PublishFAReport,
    PublishTAReport,
    PublishSAReport,
    PublishStrategyDecision
)


# ============================================================================
# Example Agent Implementations (Simplified for demonstration)
# ============================================================================

class ResearchAnalyst(BaseInvestmentAgent):
    """
    Research Analyst (RA) - Fundamental Analysis.
    
    Watches for: StartAnalysis
    Publishes: FAReport
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Alice",
            profile="Research Analyst",
            goal="Analyze company fundamentals and financial health",
            constraints="Focus on revenue growth, margins, and cash flow",
            **kwargs
        )
        # Subscribe to StartAnalysis messages
        self._watch([StartAnalysis])
        self.set_actions([PublishFAReport])
    
    async def _act(self) -> Message:
        """
        Perform fundamental analysis and publish FAReport.
        """
        logger.info(f"💼 {self.profile} analyzing fundamentals for {self._current_ticker}")
        
        # In a real implementation, this would:
        # 1. Fetch financial data from APIs (yfinance, Bloomberg, etc.)
        # 2. Use LLM to analyze earnings calls, guidance, risks
        # 3. Calculate metrics and generate insights
        
        # For this example, we'll create a mock report
        report = FAReport(
            ticker=self._current_ticker,
            revenue_growth_yoy=0.15,  # 15% YoY growth
            gross_margin=0.42,  # 42% margin
            fcf_growth=0.08,  # 8% FCF growth
            guidance_sentiment=0.6,  # Positive guidance
            key_risks=[
                "Supply chain disruptions",
                "Increased competition",
                "Regulatory pressure"
            ],
            is_growth_healthy=True
        )
        
        # Update environment state
        self.env.update_fa_report(report)
        
        # Publish the report as a message
        return await self.publish_message(
            report=report,
            cause_by=PublishFAReport
        )


class TechnicalAnalyst(BaseInvestmentAgent):
    """
    Technical Analyst (TA) - Mean Reversion Analysis.
    
    Watches for: StartAnalysis
    Publishes: TAReport
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Bob",
            profile="Technical Analyst",
            goal="Identify mean reversion opportunities using technical indicators",
            constraints="Focus on RSI, Bollinger Bands, and moving averages",
            **kwargs
        )
        self._watch([StartAnalysis])
        self.set_actions([PublishTAReport])
    
    async def _act(self) -> Message:
        """
        Perform technical analysis and publish TAReport.
        """
        logger.info(f"📊 {self.profile} analyzing technicals for {self._current_ticker}")
        
        # In a real implementation:
        # 1. Fetch historical price data
        # 2. Calculate technical indicators (RSI, BB, MA)
        # 3. Identify mean reversion signals
        
        report = TAReport(
            ticker=self._current_ticker,
            rsi_14=32.5,  # Oversold territory
            bb_lower_touch=True,  # Price touched lower band
            price_to_ma200_dist=-0.12,  # 12% below 200-day MA
            volatility_atr=2.8,  # ATR for stop-loss
            technical_signal=SignalIntensity.BUY
        )
        
        # Update environment state
        self.env.update_ta_report(report)
        
        # Publish the report
        return await self.publish_message(
            report=report,
            cause_by=PublishTAReport
        )


class SentimentAnalyst(BaseInvestmentAgent):
    """
    Sentiment Analyst (SA) - News & Catalyst Analysis.
    
    Watches for: StartAnalysis
    Publishes: SAReport
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Charlie",
            profile="Sentiment Analyst",
            goal="Analyze news sentiment and identify market-moving catalysts",
            constraints="Focus on recent news, social media, and major events",
            **kwargs
        )
        self._watch([StartAnalysis])
        self.set_actions([PublishSAReport])
    
    async def _act(self) -> Message:
        """
        Perform sentiment analysis and publish SAReport.
        """
        logger.info(f"📰 {self.profile} analyzing sentiment for {self._current_ticker}")
        
        # In a real implementation:
        # 1. Scrape news from multiple sources
        # 2. Use NLP/LLM for sentiment scoring
        # 3. Identify major events and catalysts
        
        report = SAReport(
            ticker=self._current_ticker,
            sentiment_score=0.4,  # Moderately positive
            impactful_events=[
                MarketEvent.PRODUCT_LAUNCH,
                MarketEvent.ANALYST_UPGRADE
            ],
            top_keywords=["innovation", "growth", "expansion", "partnership"],
            news_summary="Recent product launch received positive analyst coverage. Market sentiment improving."
        )
        
        # Update environment state
        self.env.update_sa_report(report)
        
        # Publish the report
        return await self.publish_message(
            report=report,
            cause_by=PublishSAReport
        )


class AlphaStrategist(BaseInvestmentAgent):
    """
    Alpha Strategist (AS) - Final Decision Synthesis.
    
    Watches for: PublishFAReport, PublishTAReport, PublishSAReport
    Publishes: StrategyDecision
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Diana",
            profile="Alpha Strategist",
            goal="Synthesize all analyst reports into actionable trading decisions",
            constraints="Balance risk-reward, consider all three analysis dimensions",
            **kwargs
        )
        # Watch for all three analyst reports
        self._watch([PublishFAReport, PublishTAReport, PublishSAReport])
        self.set_actions([PublishStrategyDecision])
    
    async def _act(self) -> Message:
        """
        Synthesize all reports and publish final StrategyDecision.
        """
        # Wait until all reports are available
        if not self.env.is_analysis_complete():
            logger.warning("⏳ AS waiting for all reports to complete...")
            return None
        
        logger.info(f"🎯 {self.profile} synthesizing final decision for {self._current_ticker}")
        
        # Get all reports from trading state
        state = self.env.trading_state
        fa = state.fa_data
        ta = state.ta_data
        sa = state.sa_data
        
        # In a real implementation:
        # 1. Use LLM with structured reasoning
        # 2. Apply decision logic/framework
        # 3. Calculate confidence scores
        
        # Example synthesis logic
        logic_chain = [
            f"FA: Growth healthy ({fa.is_growth_healthy}), revenue +{fa.revenue_growth_yoy:.1%} YoY",
            f"TA: RSI oversold ({ta.rsi_14:.1f}), touched BB lower, signal={ta.technical_signal.value}",
            f"SA: Positive sentiment ({sa.sentiment_score:.2f}), catalysts: {[e.value for e in sa.impactful_events]}",
            "CONCLUSION: Strong fundamental health + technical oversold + positive catalysts = BUY opportunity"
        ]
        
        decision = StrategyDecision(
            ticker=self._current_ticker,
            final_action=SignalIntensity.BUY,
            confidence_score=75.0,
            logic_chain=logic_chain,
            risk_notes="Set stop-loss at -8% using ATR. Monitor for earnings surprise. Position size: 3% of portfolio.",
            suggested_module="MeanReversionLong"
        )
        
        # Update environment state
        self.env.update_final_decision(decision)
        
        # Publish the decision
        return await self.publish_message(
            report=decision,
            cause_by=PublishStrategyDecision
        )


# ============================================================================
# Example Workflow Execution
# ============================================================================

async def run_analysis(ticker: str = "AAPL"):
    """
    Demonstrate the complete MAT framework workflow.
    
    Args:
        ticker: Stock ticker symbol to analyze
    """
    logger.info(f"🚀 Starting MAT Framework Analysis for {ticker}")
    
    # 1. Create the investment environment
    env = InvestmentEnvironment(desc="Multi-Agent Investment Analysis")
    env.set_ticker(ticker)
    
    # 2. Create all agents
    ra = ResearchAnalyst()
    ta = TechnicalAnalyst()
    sa = SentimentAnalyst()
    as_agent = AlphaStrategist()
    
    # 3. Set ticker for all agents
    for agent in [ra, ta, sa, as_agent]:
        agent.set_ticker(ticker)
        agent.set_env(env)
    
    # 4. Publish StartAnalysis message to trigger workflow
    logger.info("📣 Publishing StartAnalysis message...")
    start_action = StartAnalysis()
    start_message = Message(
        content=await start_action.run(ticker=ticker, context="Mean reversion opportunity screening"),
        role="Orchestrator",
        cause_by=StartAnalysis
    )
    env.publish_message(start_message)
    
    # 5. Run agents in sequence (in real Team, this would be automatic)
    # RA, TA, SA run in parallel
    logger.info("\n" + "="*60)
    logger.info("Phase 1: Parallel Analysis by RA, TA, SA")
    logger.info("="*60)
    
    await ra._observe()
    await ra._act()
    
    await ta._observe()
    await ta._act()
    
    await sa._observe()
    await sa._act()
    
    # AS runs after all reports are in
    logger.info("\n" + "="*60)
    logger.info("Phase 2: Strategy Synthesis by AS")
    logger.info("="*60)
    
    await as_agent._observe()
    await as_agent._act()
    
    # 6. Display final results
    logger.info("\n" + "="*60)
    logger.info("Final Trading State Summary")
    logger.info("="*60)
    logger.info(env.get_state_summary())
    
    if env.trading_state.final_decision:
        decision = env.trading_state.final_decision
        logger.info(f"\n🎯 FINAL DECISION: {decision.final_action.value}")
        logger.info(f"📊 Confidence: {decision.confidence_score}%")
        logger.info(f"🧠 Logic Chain:")
        for i, step in enumerate(decision.logic_chain, 1):
            logger.info(f"   {i}. {step}")
        logger.info(f"⚠️  Risk Notes: {decision.risk_notes}")
        logger.info(f"🔧 Suggested Module: {decision.suggested_module}")
    
    logger.info("\n✅ MAT Framework Analysis Complete!")


if __name__ == "__main__":
    # Run the example workflow
    asyncio.run(run_analysis(ticker="AAPL"))

