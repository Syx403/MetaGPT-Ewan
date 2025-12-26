"""
Filename: MetaGPT-Ewan/MAT/scheme_c_demo.py
Created Date: Saturday, December 27th 2025
Author: Ewan Su
Description: Demonstration of Scheme C (Active Inquiry) with conflict detection and dynamic investigation.

This demo shows:
1. How Alpha Strategist detects conflicts between FA/TA and SA
2. How investigation requests are triggered based on importance levels
3. How Sentiment Analyst performs deep dive investigations
4. How final decisions are made with Safe-First approach
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from metagpt.logs import logger
from metagpt.schema import Message

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
from MAT.actions import StartAnalysis, PublishFAReport, PublishTAReport, PublishSAReport

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Create output directory for search results
SEARCH_OUTPUT_DIR = Path("MAT/data/search_results")
SEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_search_results(ticker: str, round_num: int, articles: list):
    """
    Save search results to a file for verification.
    
    Args:
        ticker: Stock ticker symbol
        round_num: Investigation round number
        articles: List of news articles from search
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = SEARCH_OUTPUT_DIR / f"{ticker}_round{round_num}_{timestamp}.json"
    
    # Also create a markdown version for easy reading
    md_filename = SEARCH_OUTPUT_DIR / f"{ticker}_round{round_num}_{timestamp}.md"
    
    # Save JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    # Save Markdown
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(f"# DuckDuckGo Search Results for {ticker}\n")
        f.write(f"**Investigation Round:** {round_num}\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Total Articles:** {len(articles)}\n\n")
        f.write("---\n\n")
        
        for i, article in enumerate(articles, 1):
            f.write(f"## Article {i}: {article.get('title', 'No Title')}\n\n")
            f.write(f"**URL:** {article.get('url', 'N/A')}\n\n")
            f.write(f"**Content:**\n{article.get('content', 'No content')}\n\n")
            f.write(f"**Relevance Score:** {article.get('score', 'N/A')}\n\n")
            f.write("---\n\n")
    
    logger.info(f"📁 Search results saved to:")
    logger.info(f"   JSON: {filename}")
    logger.info(f"   MD:   {md_filename}")


async def demo_conflict_scenario():
    """
    Demonstrate Scheme C with a conflict scenario.
    
    Scenario: NVDA stock
    - Fundamentals: Excellent (30% revenue growth, healthy margins)
    - Technicals: Bullish (RSI oversold, touched lower BB)
    - Sentiment: Negative (-0.4) due to regulatory concerns
    
    Expected Behavior:
    1. AS detects conflict (bullish FA/TA vs negative SA)
    2. AS calculates importance_level=2 (revenue growth > 30%)
    3. AS requests investigation (max_retries=2)
    4. SA performs deep dive with targeted search
    5. SA publishes InvestigationReport
    6. AS makes final decision based on revised findings
    """
    logger.info("🚀 Starting Scheme C Demonstration: Conflict Resolution")
    logger.info("="*80)
    logger.info(f"📁 Search results will be saved to: {SEARCH_OUTPUT_DIR.absolute()}")
    logger.info("="*80)
    
    # Setup environment
    env = InvestmentEnvironment(desc="Scheme C Active Inquiry Demo")
    ticker = "NVDA"
    env.set_ticker(ticker)
    
    # Create agents
    sa = SentimentAnalyst()
    as_agent = AlphaStrategist()
    
    # Configure agents
    sa.set_ticker(ticker)
    sa.set_env(env)
    as_agent.set_ticker(ticker)
    as_agent.set_env(env)
    
    logger.info(f"✅ Environment and agents initialized for {ticker}")
    logger.info("="*80 + "\n")
    
    # ========================================================================
    # Phase 1: Publish Initial Reports (Conflict Scenario)
    # ========================================================================
    
    logger.info("📊 PHASE 1: Publishing Initial Analyst Reports")
    logger.info("-"*80)
    
    # FA Report: Strong fundamentals (bullish)
    fa_report = FAReport(
        ticker=ticker,
        revenue_growth_yoy=0.35,  # 35% growth - triggers importance_level=2
        gross_margin=0.68,  # Excellent 68% margin
        fcf_growth=0.25,  # Strong cash flow
        guidance_sentiment=0.8,  # Very positive guidance
        key_risks=[
            "AI chip competition intensifying",
            "Export restrictions to China",
            "Valuation concerns at current levels"
        ],
        is_growth_healthy=True  # Bullish signal
    )
    
    logger.info(f"📈 FA: Revenue growth {fa_report.revenue_growth_yoy:.1%}, Healthy={fa_report.is_growth_healthy}")
    fa_message = Message(
        content=fa_report.model_dump_json(),
        role="ResearchAnalyst",
        cause_by=PublishFAReport
    )
    env.publish_message(fa_message)
    env.update_fa_report(fa_report)
    
    # TA Report: Bullish technicals
    ta_report = TAReport(
        ticker=ticker,
        rsi_14=28.5,  # Oversold
        bb_lower_touch=True,  # Touched lower band
        price_to_ma200_dist=-0.15,  # 15% below 200-day MA
        volatility_atr=3.2,
        technical_signal=SignalIntensity.BUY  # Bullish signal
    )
    
    logger.info(f"📊 TA: RSI={ta_report.rsi_14:.1f}, Signal={ta_report.technical_signal.value}")
    ta_message = Message(
        content=ta_report.model_dump_json(),
        role="TechnicalAnalyst",
        cause_by=PublishTAReport
    )
    env.publish_message(ta_message)
    env.update_ta_report(ta_report)
    
    # SA Report: Negative sentiment (creates conflict!)
    sa_report = SAReport(
        ticker=ticker,
        sentiment_score=-0.4,  # Negative - conflicts with FA/TA!
        impactful_events=[
            MarketEvent.REGULATORY_ACTION,
            MarketEvent.MACRO_ECONOMIC
        ],
        top_keywords=[
            "regulatory concerns",
            "export restrictions",
            "AI oversight",
            "valuation worries"
        ],
        news_summary="Recent regulatory scrutiny and export restrictions to China have dampened market sentiment despite strong earnings."
    )
    
    logger.info(f"📰 SA: Sentiment={sa_report.sentiment_score:.2f} (NEGATIVE)")
    logger.info(f"⚠️  Events: {[e.value for e in sa_report.impactful_events]}")
    sa_message = Message(
        content=sa_report.model_dump_json(),
        role="SentimentAnalyst",
        cause_by=PublishSAReport
    )
    env.publish_message(sa_message)
    env.update_sa_report(sa_report)
    
    logger.info("\n" + "="*80)
    logger.info("💥 CONFLICT SETUP COMPLETE!")
    logger.info("   FA/TA = BULLISH | SA = NEGATIVE")
    logger.info("="*80 + "\n")
    
    # ========================================================================
    # Phase 2-N: Investigation Loop (up to max_retries)
    # ========================================================================
    
    logger.info("📊 PHASE 2+: Multi-Round Investigation Loop")
    logger.info("="*80)
    
    max_rounds = 5  # Safety limit to prevent infinite loops
    round_num = 1
    
    while round_num <= max_rounds:
        logger.info(f"\n🔄 ROUND {round_num}:")
        logger.info("-"*80)
        
        # Alpha Strategist observes and acts
        logger.info(f"📊 Round {round_num}.1: Alpha Strategist Analysis")
        await as_agent._observe()
        as_msg = await as_agent._act()
        
        if not as_msg:
            logger.info("⏹️  No message from Alpha Strategist - workflow complete")
            break
        
        # Check if it's a final decision or investigation request
        if "RequestInvestigation" in str(as_msg.cause_by):
            logger.info(f"✅ Alpha Strategist requested Investigation (Attempt {round_num})")
            
            # Sentiment Analyst performs deep dive
            logger.info(f"📊 Round {round_num}.2: Sentiment Analyst Deep Dive")
            await sa._observe()
            
            # Hook: Save search results before acting
            # We'll need to temporarily store the search results from SA
            sa_msg = await sa._act()
            
            # Try to extract and save search results if available
            if hasattr(sa, '_last_search_results') and sa._last_search_results:
                save_search_results(ticker, round_num, sa._last_search_results)
            
            if sa_msg:
                logger.info(f"✅ Sentiment Analyst completed Investigation")
            else:
                logger.warning(f"⚠️ Investigation failed")
                break
            
            round_num += 1
            
        elif "PublishStrategyDecision" in str(as_msg.cause_by):
            logger.info(f"🎯 Alpha Strategist made FINAL DECISION!")
            break
        else:
            logger.warning(f"⚠️ Unexpected message type: {as_msg.cause_by}")
            break
    
    if round_num > max_rounds:
        logger.warning(f"⚠️ Reached maximum rounds ({max_rounds}) - stopping")
    
    # ========================================================================
    # Final: Display Results
    # ========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL RESULTS")
    logger.info("="*80)
    
    logger.info(env.get_state_summary())
    
    if env.trading_state.final_decision:
        decision = env.trading_state.final_decision
        logger.info(f"\n🎯 FINAL ACTION: {decision.final_action.value}")
        logger.info(f"📊 Confidence: {decision.confidence_score}%")
        logger.info(f"🔧 Module: {decision.suggested_module}")
        logger.info(f"\n🧠 Logic Chain:")
        for i, step in enumerate(decision.logic_chain, 1):
            logger.info(f"   {i}. {step}")
        logger.info(f"\n⚠️  Risk Notes: {decision.risk_notes}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Scheme C Demonstration Complete!")
    logger.info("="*80)


async def demo_high_growth_scenario():
    """
    Demonstrate high importance scenario with max_retries=2.
    
    Scenario: High-growth company (>30% revenue) with sentiment conflict
    Shows that AS will retry investigation twice before making safe decision.
    """
    logger.info("\n\n")
    logger.info("🚀 Starting High-Growth Scenario: Multiple Investigation Retries")
    logger.info("="*80)
    
    # This is a simplified version - in real implementation would show
    # multiple investigation cycles
    logger.info("📝 This scenario demonstrates:")
    logger.info("   - Revenue growth > 30% → importance_level=2")
    logger.info("   - max_retries=2 (two investigation attempts)")
    logger.info("   - Safe-First decision if conflict persists after retries")
    logger.info("="*80)


async def demo_normal_importance_scenario():
    """
    Demonstrate normal importance scenario with max_retries=1.
    
    Scenario: Normal growth company (<30% revenue) with sentiment conflict
    Shows single investigation before decision.
    """
    logger.info("\n\n")
    logger.info("🚀 Starting Normal Importance Scenario: Single Investigation")
    logger.info("="*80)
    
    logger.info("📝 This scenario demonstrates:")
    logger.info("   - Revenue growth < 30% → importance_level=1")
    logger.info("   - max_retries=1 (single investigation)")
    logger.info("   - Faster decision cycle for lower importance conflicts")
    logger.info("="*80)


if __name__ == "__main__":
    # Run the main conflict resolution demo
    asyncio.run(demo_conflict_scenario())
    
    # Show additional scenario descriptions
    asyncio.run(demo_high_growth_scenario())
    asyncio.run(demo_normal_importance_scenario())

