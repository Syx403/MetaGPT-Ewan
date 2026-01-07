"""
Filename: MetaGPT-Ewan/MAT/main.py
Created Date: Wednesday, January 8th 2026
Author: Claude Code (Anthropic)
Description: Pure Dynamic Orchestrator for Multi-Agent Trading System (MAT).

This script orchestrates the full "Scheme C" investment analysis workflow using MetaGPT.
Features:
- CLI interface for ticker and fiscal year configuration
- Real-time API calls (RAGFlow, yfinance, Tavily)
- Enhanced debugging and message tracing
- Error handling with fail-safe logging
- Result persistence (JSON + Markdown)
"""

import argparse
import asyncio
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from metagpt.team import Team
from metagpt.logs import logger
from metagpt.schema import Message

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MAT.environment import InvestmentEnvironment
from MAT.roles import (
    ResearchAnalyst,
    TechnicalAnalyst,
    SentimentAnalyst,
    AlphaStrategist,
)
from MAT.actions import StartAnalysis
from MAT.schemas import StrategyDecision


# ============================================================================
# CLI Configuration
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments for the MAT orchestrator.

    Returns:
        argparse.Namespace with ticker, fiscal_year, and debug flags
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Trading System (MAT) - Dynamic Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis for Coca-Cola 2022
  python MAT/main.py --ticker KO --fiscal_year 2022

  # With debug logging enabled
  python MAT/main.py --ticker AAPL --fiscal_year 2023 --debug

  # Analyze Tesla with verbose output
  python MAT/main.py --ticker TSLA --fiscal_year 2021 --debug --verbose
        """
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., KO, AAPL, TSLA)"
    )

    parser.add_argument(
        "--fiscal_year",
        type=int,
        required=True,
        help="Target fiscal year for analysis (e.g., 2021, 2022)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debugging logs and metric snapshots"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable extra verbose logging (includes all message traces)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="MAT/report/Strategy",
        help="Output directory for strategy reports (default: MAT/report/Strategy)"
    )

    parser.add_argument(
        "--max_rounds",
        type=int,
        default=10,
        help="Maximum number of agent execution rounds (default: 10)"
    )

    return parser.parse_args()


# ============================================================================
# Message Tracing Hook
# ============================================================================

class MessageTracer:
    """
    Real-time message tracer for debugging agent communication.
    """

    def __init__(self, enabled: bool = False, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.message_count = 0
        self.phase_tracker = {
            "perception": {"RA": False, "TA": False, "SA": False},
            "audit": False,
            "investigation": False,
            "synthesis": False
        }

    def trace_message(self, message: Message, phase: str = "unknown"):
        """
        Log a traced message with phase information.

        Args:
            message: The message being published
            phase: The workflow phase (perception, audit, investigation, synthesis)
        """
        if not self.enabled:
            return

        self.message_count += 1

        # Extract cause_by string
        cause_by_str = str(message.cause_by) if not isinstance(message.cause_by, str) else message.cause_by

        if self.verbose:
            logger.info(f"[TRACE #{self.message_count}] Phase={phase} | Action={cause_by_str}")

        # Track phase progression
        if "PublishFAReport" in cause_by_str:
            self.phase_tracker["perception"]["RA"] = True
            logger.info("[TRACE] Perception Phase: RA published FAReport ✅")
        elif "PublishTAReport" in cause_by_str:
            self.phase_tracker["perception"]["TA"] = True
            logger.info("[TRACE] Perception Phase: TA published TAReport ✅")
        elif "PublishSAReport" in cause_by_str:
            self.phase_tracker["perception"]["SA"] = True
            logger.info("[TRACE] Perception Phase: SA published SAReport ✅")
        elif "RequestInvestigation" in cause_by_str:
            self.phase_tracker["investigation"] = True
            logger.info("[TRACE] Audit Phase: AS triggered Investigation Request 🔍")
        elif "PublishInvestigationReport" in cause_by_str:
            logger.info("[TRACE] Investigation Phase: SA published InvestigationReport ✅")
        elif "PublishStrategyDecision" in cause_by_str:
            self.phase_tracker["synthesis"] = True
            logger.info("[TRACE] Synthesis Phase: AS published StrategyDecision 🎯")

    def check_perception_complete(self) -> bool:
        """Check if all perception agents have published reports."""
        return all(self.phase_tracker["perception"].values())

    def print_summary(self):
        """Print a summary of traced messages."""
        if not self.enabled:
            return

        logger.info("\n" + "="*80)
        logger.info("MESSAGE TRACE SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Messages Traced: {self.message_count}")
        logger.info(f"Perception Complete: {self.check_perception_complete()}")
        logger.info(f"Investigation Triggered: {self.phase_tracker['investigation']}")
        logger.info(f"Synthesis Complete: {self.phase_tracker['synthesis']}")
        logger.info("="*80 + "\n")


# ============================================================================
# Metric Snapshot Printer
# ============================================================================

def print_metric_snapshot(env: InvestmentEnvironment, debug: bool = False):
    """
    Print a snapshot of key metrics before final decision synthesis.

    Args:
        env: InvestmentEnvironment with trading state
        debug: Whether debug mode is enabled
    """
    if not debug:
        return

    state = env.trading_state
    if not state:
        logger.warning("[DEBUG] No trading state available for metric snapshot")
        return

    logger.info("\n" + "="*80)
    logger.info("METRIC SNAPSHOT (Pre-Decision)")
    logger.info("="*80)

    # FA Metrics
    if state.fa_data:
        logger.info(f"📊 RA Metrics:")
        logger.info(f"   - Revenue: {state.fa_data.revenue_performance.value}")
        logger.info(f"   - Profitability: {state.fa_data.profitability_audit.value}")
        logger.info(f"   - Cash Flow: {state.fa_data.cash_flow_stability.value}")
        logger.info(f"   - Guidance: {state.fa_data.management_guidance_audit[:60]}...")
        logger.info(f"   - Key Risks: {len(state.fa_data.key_risks_evidence)} identified")
    else:
        logger.warning("   - FA data not available ❌")

    # TA Metrics
    if state.ta_data:
        logger.info(f"📈 TA Metrics:")
        logger.info(f"   - RSI(14): {state.ta_data.rsi_14:.2f}")
        logger.info(f"   - BB Lower Touch: {state.ta_data.bb_lower_touch}")
        logger.info(f"   - MA Distances: MA20={state.ta_data.price_to_ma20_dist:+.2%}, "
                   f"MA50={state.ta_data.price_to_ma50_dist:+.2%}, "
                   f"MA200={state.ta_data.price_to_ma200_dist:+.2%}")
        logger.info(f"   - Market Regime: {state.ta_data.market_regime[:60]}...")
    else:
        logger.warning("   - TA data not available ❌")

    # SA Metrics
    if state.sa_data:
        logger.info(f"📰 SA Metrics:")
        logger.info(f"   - Sentiment: {state.sa_data.qualitative_sentiment_assessment[:60]}...")
        logger.info(f"   - Events: {[e.value for e in state.sa_data.impactful_events]}")
        logger.info(f"   - Expectation Gap: {state.sa_data.expectation_gap[:60]}...")
    else:
        logger.warning("   - SA data not available ❌")

    logger.info("="*80 + "\n")


# ============================================================================
# Result Persistence
# ============================================================================

def save_strategy_decision(
    decision: StrategyDecision,
    output_dir: str,
    ticker: str,
    fiscal_year: int
):
    """
    Save StrategyDecision to JSON and Markdown formats.

    Args:
        decision: StrategyDecision object to save
        output_dir: Output directory path
        ticker: Stock ticker symbol
        fiscal_year: Target fiscal year
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filenames
    json_filename = f"Strategy_report_{ticker}_{fiscal_year}.json"
    md_filename = f"Strategy_report_{ticker}_{fiscal_year}.md"

    json_path = output_path / json_filename
    md_path = output_path / md_filename

    # Save JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            # Use mode='json' to properly serialize Enums
            f.write(decision.model_dump_json(indent=2))
        logger.info(f"✅ Strategy decision saved to JSON: {json_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save JSON: {e}")

    # Save Markdown
    try:
        md_content = generate_markdown_report(decision, ticker, fiscal_year)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logger.info(f"✅ Strategy decision saved to Markdown: {md_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save Markdown: {e}")


def generate_markdown_report(
    decision: StrategyDecision,
    ticker: str,
    fiscal_year: int
) -> str:
    """
    Generate a formatted Markdown report for the strategy decision.

    Args:
        decision: StrategyDecision object
        ticker: Stock ticker symbol
        fiscal_year: Target fiscal year

    Returns:
        Markdown-formatted string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Strategy Decision Report

**Ticker:** {ticker}
**Fiscal Year:** {fiscal_year}
**Generated:** {timestamp}

---

## Executive Summary

**Final Action:** `{decision.final_action.value}`
**Confidence Score:** {decision.confidence_score}%
**Suggested Module:** {decision.suggested_module}

### Decision Summary
{decision.decision_summary}

---

## Decision Logic Chain

"""

    for i, step in enumerate(decision.logic_chain, 1):
        md += f"{i}. {step}\n"

    md += f"""
---

## Risk Management Notes

{decision.risk_notes}

---

## Conflict Resolution Report

{decision.conflict_report}

---

## Metadata

- **Report Type:** Strategy Decision
- **Generated By:** Multi-Agent Trading System (MAT)
- **Framework:** Scheme C (Active Inquiry with Conflict Resolution)
- **Analyst Roles:** Research Analyst (RA), Technical Analyst (TA), Sentiment Analyst (SA), Alpha Strategist (AS)

---

*This report was generated using the MAT framework powered by MetaGPT and Claude Sonnet 4.5.*
"""

    return md


# ============================================================================
# Console Final Report
# ============================================================================

def print_final_report(decision: StrategyDecision, ticker: str, fiscal_year: int):
    """
    Print a high-level summary of the final decision to console.

    Args:
        decision: StrategyDecision object
        ticker: Stock ticker symbol
        fiscal_year: Target fiscal year
    """
    logger.info("\n" + "="*80)
    logger.info("FINAL STRATEGY DECISION")
    logger.info("="*80)
    logger.info(f"Ticker: {ticker} | Fiscal Year: {fiscal_year}")
    logger.info(f"")
    logger.info(f"🎯 Final Action: {decision.final_action.value}")
    logger.info(f"📊 Confidence Score: {decision.confidence_score}%")
    logger.info(f"⚙️  Suggested Module: {decision.suggested_module}")
    logger.info(f"")
    logger.info(f"📝 Decision Summary:")
    logger.info(f"   {decision.decision_summary}")
    logger.info(f"")
    logger.info(f"⚠️  Risk Notes:")
    logger.info(f"   {decision.risk_notes[:200]}...")
    logger.info(f"")
    logger.info(f"🔍 Conflict Report:")
    logger.info(f"   {decision.conflict_report[:200]}...")
    logger.info("="*80 + "\n")


# ============================================================================
# Main Orchestrator
# ============================================================================

async def run_analysis(
    ticker: str,
    fiscal_year: int,
    debug: bool = False,
    verbose: bool = False,
    output_dir: str = "MAT/report/Strategy",
    max_rounds: int = 10
):
    """
    Execute the full Scheme C investment analysis workflow.

    Args:
        ticker: Stock ticker symbol
        fiscal_year: Target fiscal year
        debug: Enable debug logging
        verbose: Enable verbose message tracing
        output_dir: Output directory for reports
        max_rounds: Maximum agent execution rounds

    Returns:
        StrategyDecision if successful, None if failed
    """
    logger.info("\n" + "="*80)
    logger.info("MULTI-AGENT TRADING SYSTEM (MAT) - SCHEME C")
    logger.info("="*80)
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Fiscal Year: {fiscal_year}")
    logger.info(f"Debug Mode: {debug}")
    logger.info(f"Max Rounds: {max_rounds}")
    logger.info("="*80 + "\n")

    # Initialize message tracer
    tracer = MessageTracer(enabled=debug, verbose=verbose)

    try:
        # Step 1: Initialize Environment
        logger.info("[SYSTEM] Initializing InvestmentEnvironment...")
        env = InvestmentEnvironment(desc=f"MAT Analysis Environment - {ticker} {fiscal_year}")
        env.set_ticker(ticker)

        # Step 2: Initialize Roles with system_reference_date for time-aligned analysis
        logger.info("[SYSTEM] Initializing analyst roles...")

        # Calculate reference date (end of fiscal year)
        reference_date = f"{fiscal_year}-12-31"

        ra = ResearchAnalyst(system_reference_date=reference_date)
        ta = TechnicalAnalyst(system_reference_date=reference_date)
        sa = SentimentAnalyst(use_tavily=True, system_reference_date=reference_date)
        as_role = AlphaStrategist()

        # Step 3: Add roles to environment
        logger.info("[SYSTEM] Adding roles to environment...")
        env.add_roles([ra, ta, sa, as_role])

        # Confirmation message
        logger.info("[SYSTEM] ✅ 4 Roles (RA, TA, SA, AS) successfully integrated into the Environment")
        logger.info("")

        # Step 4: Set environment for all roles (skip Team, use env directly)
        logger.info("[SYSTEM] Setting environment for all roles...")
        for role in [ra, ta, sa, as_role]:
            role.set_env(env)

        logger.info("[SYSTEM] ✅ All roles configured with InvestmentEnvironment")
        logger.info("")

        # Step 5: Publish StartAnalysis message
        logger.info(f"[SYSTEM] 🚀 Triggering analysis for {ticker} (Fiscal Year {fiscal_year})...")
        start_message = Message(
            content=json.dumps({"ticker": ticker, "fiscal_year": fiscal_year}),
            cause_by=StartAnalysis,
            role="Orchestrator"
        )
        env.publish_message(start_message)

        logger.info("[SYSTEM] StartAnalysis message published")
        logger.info("")

        # Step 6: Execute Environment (Asynchronous)
        logger.info(f"[SYSTEM] Running environment for maximum {max_rounds} rounds...")
        logger.info("[SYSTEM] ⏳ Please wait for agents to complete their analysis...")
        logger.info("")

        # Run environment directly with message tracing
        round_count = 0
        for round_num in range(max_rounds):
            round_count += 1

            if debug:
                logger.info(f"[DEBUG] Round {round_num + 1}/{max_rounds} starting...")

            # Run one round
            await env.run(k=1)

            # Check if analysis is complete (final decision published)
            if env.trading_state and env.trading_state.final_decision:
                logger.info(f"[SYSTEM] ✅ Analysis complete after {round_count} rounds")
                break

            # Check for timeout warning
            if round_num == max_rounds - 2:
                logger.warning(f"[WARNING] Approaching maximum rounds ({max_rounds}), analysis may be incomplete")

        # Step 7: Print metric snapshot (debug mode)
        if debug:
            logger.info("[DEBUG] Generating metric snapshot...")
            print_metric_snapshot(env, debug=debug)

        # Step 8: Retrieve final decision
        if not env.trading_state or not env.trading_state.final_decision:
            logger.error("[ERROR] ❌ Analysis failed: No final decision produced")
            logger.error("[ERROR] This could indicate:")
            logger.error("   - One or more agents failed to publish reports")
            logger.error("   - AlphaStrategist did not synthesize a decision")
            logger.error("   - Maximum rounds reached before completion")
            return None

        decision = env.trading_state.final_decision
        logger.info("[SYSTEM] ✅ Final decision retrieved successfully")
        logger.info("")

        # Step 9: Print message trace summary (debug mode)
        if debug:
            tracer.print_summary()

        # Step 10: Print final report to console
        print_final_report(decision, ticker, fiscal_year)

        # Step 11: Save results
        logger.info("[SYSTEM] Saving strategy decision...")
        save_strategy_decision(decision, output_dir, ticker, fiscal_year)
        logger.info("")

        logger.info("[SYSTEM] ✅ Analysis workflow complete!")
        return decision

    except KeyboardInterrupt:
        logger.warning("\n[SYSTEM] ⚠️ Analysis interrupted by user (Ctrl+C)")
        return None

    except Exception as e:
        logger.error(f"\n[ERROR] ❌ Analysis failed with exception: {e}")
        logger.error(f"[ERROR] Traceback:")
        logger.error(traceback.format_exc())

        # Fail-safe logging for debugging
        if debug:
            logger.error("[DEBUG] Environment state at failure:")
            if env and env.trading_state:
                logger.error(f"   - FA data: {'✅' if env.trading_state.fa_data else '❌'}")
                logger.error(f"   - TA data: {'✅' if env.trading_state.ta_data else '❌'}")
                logger.error(f"   - SA data: {'✅' if env.trading_state.sa_data else '❌'}")
                logger.error(f"   - Final decision: {'✅' if env.trading_state.final_decision else '❌'}")

        return None


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """
    Main entry point for the MAT orchestrator.
    """
    # Parse CLI arguments
    args = parse_arguments()

    # Run analysis
    decision = asyncio.run(run_analysis(
        ticker=args.ticker,
        fiscal_year=args.fiscal_year,
        debug=args.debug,
        verbose=args.verbose,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds
    ))

    # Exit with appropriate code
    if decision:
        logger.info("[SYSTEM] 🎉 MAT analysis completed successfully!")
        sys.exit(0)
    else:
        logger.error("[SYSTEM] ❌ MAT analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
