"""
Filename: MetaGPT-Ewan/MAT/roles/alpha_strategist.py
Created Date: Saturday, December 27th 2025
Updated Date: Tuesday, January 7th 2026
Author: Ewan Su
Description: Alpha Strategist implementing Scheme C (Active Inquiry) using AnalyzeConflict and SynthesizeDecision Actions.

REFACTORED: Now delegates all conflict detection and decision synthesis to native Actions.
"""

from typing import Dict, Optional
from metagpt.schema import Message
from metagpt.logs import logger
import json

from ..roles.base_agent import BaseInvestmentAgent
from ..schemas import (
    FAReport,
    TAReport,
    SAReport,
    InvestigationReport,
    InvestigationRequest,
    StrategyDecision,
    TradingState
)
from ..actions import (
    PublishFAReport,
    PublishTAReport,
    PublishSAReport,
    PublishInvestigationReport,
    RequestInvestigation,
    PublishStrategyDecision,
    AnalyzeConflict,
    SynthesizeDecision
)


class AlphaStrategist(BaseInvestmentAgent):
    """
    Alpha Strategist (AS) implementing Scheme C: Active Inquiry with conflict resolution.

    REFACTORED DESIGN:
    - All conflict detection logic delegated to AnalyzeConflict Action
    - All decision synthesis logic delegated to SynthesizeDecision Action
    - AS Role only handles: report buffering, workflow orchestration, message publishing

    Workflow (Scheme C):
    Step 1 (Wait): Collect FA, TA, SA reports
    Step 2 (Inquiry): If conflict detected → Request SA investigation
    Step 3 (Decision): Call SynthesizeDecision with all reports (including SA-Advanced if available)
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="AlphaStrategist",
            profile="Alpha Strategist",
            goal="Synthesize multi-dimensional analysis into actionable trading decisions",
            constraints="Use Safe-First approach; delegate reasoning to Actions",
            **kwargs
        )

        # Subscribe to all analyst reports and investigation reports
        self._watch([
            PublishFAReport,
            PublishTAReport,
            PublishSAReport,
            PublishInvestigationReport
        ])

        # Set our action types
        self.set_actions([RequestInvestigation, PublishStrategyDecision])

        # Initialize Actions for conflict detection and decision synthesis
        self._analyze_conflict_action = AnalyzeConflict()
        self._synthesize_decision_action = SynthesizeDecision()

        # Internal state tracking per ticker
        self._ticker_states: Dict[str, TradingState] = {}
        self._pending_investigations: Dict[str, bool] = {}  # Track if investigation is pending
        self._investigation_reports: Dict[str, InvestigationReport] = {}  # Store investigation results

        logger.info("🧠 Alpha Strategist initialized with Scheme C (delegated to Actions)")

    def _get_or_create_state(self, ticker: str) -> TradingState:
        """
        Get or create internal trading state for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            TradingState for the ticker
        """
        if ticker not in self._ticker_states:
            self._ticker_states[ticker] = TradingState(current_ticker=ticker)
            self._pending_investigations[ticker] = False
            logger.debug(f"📊 Created new state for ticker: {ticker}")

        return self._ticker_states[ticker]

    async def _act(self) -> Message:
        """
        Main action logic: process reports, orchestrate workflow.

        Returns:
            Message containing either InvestigationRequest or StrategyDecision
        """
        # Get the newly observed messages (from _observe)
        news = self.rc.news
        if not news:
            logger.debug("⏳ AlphaStrategist: No new messages to process")
            return None

        # Debug: log what messages we're processing
        logger.info(f"📬 AlphaStrategist received {len(news)} message(s):")
        for i, msg in enumerate(news, 1):
            logger.info(f"   {i}. {msg.cause_by}")

        # Process all recent messages to update internal state
        for msg in news:
            await self._process_message(msg)

        # Check if we have a ticker to work with
        if not self._current_ticker:
            logger.warning("⚠️ No current ticker set")
            return None

        state = self._get_or_create_state(self._current_ticker)

        # Check if we're waiting for an investigation response
        if self._pending_investigations.get(self._current_ticker, False):
            logger.debug(f"⏳ Waiting for investigation response for {self._current_ticker}")
            return None

        # Check if we have all required reports (FA, TA, SA)
        if not self._has_minimum_reports(state):
            logger.debug(f"⏳ Waiting for more reports for {self._current_ticker}")
            return None

        # Perform Scheme C workflow
        return await self._execute_scheme_c_workflow(state)

    async def _process_message(self, message: Message):
        """
        Process incoming messages and update internal state.

        Args:
            message: Message from other agents
        """
        try:
            content = json.loads(message.content)
            ticker = content.get("ticker")

            if not ticker:
                return

            # Update current ticker if not set
            if not self._current_ticker:
                self._current_ticker = ticker

            state = self._get_or_create_state(ticker)

            # Get the cause_by string for comparison
            cause_by_str = str(message.cause_by) if not isinstance(message.cause_by, str) else message.cause_by

            # Update state based on message type
            if message.cause_by == PublishFAReport or cause_by_str.endswith("PublishFAReport"):
                state.fa_data = FAReport(**content)
                logger.info(f"📈 Received FA Report for {ticker}")
                logger.info(f"   Revenue: {state.fa_data.revenue_performance.value}")

            elif message.cause_by == PublishTAReport or cause_by_str.endswith("PublishTAReport"):
                state.ta_data = TAReport(**content)
                logger.info(f"📊 Received TA Report for {ticker}")
                logger.info(f"   Market Regime: {state.ta_data.market_regime[:60]}...")

            elif message.cause_by == PublishSAReport or cause_by_str.endswith("PublishSAReport"):
                state.sa_data = SAReport(**content)
                logger.info(f"📰 Received SA Report for {ticker}")
                logger.info(f"   Assessment: {state.sa_data.qualitative_sentiment_assessment[:60]}...")

            elif message.cause_by == PublishInvestigationReport or cause_by_str.endswith("PublishInvestigationReport"):
                # Store investigation report separately
                inv_report = InvestigationReport(**content)
                self._investigation_reports[ticker] = inv_report
                self._pending_investigations[ticker] = False

                logger.info(f"🔍 Received Investigation Report for {ticker}")
                logger.info(f"   Risk Classification: {inv_report.risk_classification}")
                logger.info(f"   Ambiguity Resolved: {inv_report.is_ambiguity_resolved}")

        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")

    def _has_minimum_reports(self, state: TradingState) -> bool:
        """
        Check if we have minimum required reports to make a decision.

        Args:
            state: TradingState to check

        Returns:
            True if FA, TA, and SA reports are all available
        """
        return (
            state.fa_data is not None and
            state.ta_data is not None and
            state.sa_data is not None
        )

    async def _execute_scheme_c_workflow(self, state: TradingState) -> Optional[Message]:
        """
        Execute Scheme C workflow using AnalyzeConflict and SynthesizeDecision Actions.

        Workflow Steps:
        1. Call AnalyzeConflict.run(ra, ta, sa) to detect conflicts
        2. If conflict detected and no investigation done yet → Request investigation
        3. If no conflict OR investigation complete → Call SynthesizeDecision.run()

        Args:
            state: TradingState with all reports

        Returns:
            Message with InvestigationRequest or StrategyDecision
        """
        ticker = state.current_ticker

        logger.info(f"\n{'='*70}")
        logger.info(f"🧠 SCHEME C WORKFLOW for {ticker}")
        logger.info(f"{'='*70}")

        # STEP 1: Conflict Detection (delegated to AnalyzeConflict Action)
        logger.info(f"📋 Step 1: Analyzing conflicts using AnalyzeConflict Action...")

        conflict_result = await self._analyze_conflict_action.run(
            ticker=ticker,
            ra=state.fa_data,
            ta=state.ta_data,
            sa=state.sa_data
        )

        has_conflict = conflict_result["has_conflict"]
        context_issue = conflict_result["context_issue"]

        logger.info(f"   Conflict Detected: {has_conflict}")
        if has_conflict:
            logger.info(f"   Issue: {context_issue[:100]}...")

        # STEP 2: Inquiry Decision
        # Check if we already have an investigation report for this ticker
        has_investigation = ticker in self._investigation_reports

        if has_conflict and not has_investigation:
            # Request investigation
            logger.info(f"🔍 Step 2: CONFLICT DETECTED - Requesting SA investigation")
            return await self._request_investigation(state, context_issue)
        else:
            # Proceed to decision synthesis
            if has_conflict and has_investigation:
                logger.info(f"✅ Step 2: Conflict detected BUT investigation complete - proceeding to synthesis")
            else:
                logger.info(f"✅ Step 2: No conflicts detected - proceeding to synthesis")

            # STEP 3: Decision Synthesis (delegated to SynthesizeDecision Action)
            return await self._finalize_decision(state, context_issue if has_conflict else None)

    async def _request_investigation(
        self,
        state: TradingState,
        conflict_issue: str
    ) -> Message:
        """
        Request a deep dive investigation from the Sentiment Analyst.

        Args:
            state: TradingState with all reports
            conflict_issue: Description of the detected conflict

        Returns:
            Message with InvestigationRequest
        """
        ticker = state.current_ticker

        # Create investigation request
        # Note: We use importance_level=2 for all conflicts (can be made dynamic later)
        request = InvestigationRequest(
            ticker=ticker,
            target_agent="SA",
            context_issue=conflict_issue,
            current_retry=0,
            max_retries=1,
            importance_level=2  # High importance by default
        )

        # Mark investigation as pending
        self._pending_investigations[ticker] = True

        logger.info(f"📤 Publishing InvestigationRequest for {ticker}")
        logger.info(f"   Context: {conflict_issue[:100]}...")

        # Publish investigation request
        message = self.publish_message(
            report=request,
            cause_by=RequestInvestigation
        )

        return message

    async def _finalize_decision(
        self,
        state: TradingState,
        conflict_issue: Optional[str] = None
    ) -> Message:
        """
        Make the final trading decision using SynthesizeDecision Action.

        This method delegates ALL decision logic to the SynthesizeDecision Action,
        which contains the LLM prompts and reasoning logic.

        Args:
            state: TradingState with all reports
            conflict_issue: Optional conflict description (if detected)

        Returns:
            Message with StrategyDecision
        """
        ticker = state.current_ticker

        logger.info(f"🎯 Step 3: DECISION SYNTHESIS using SynthesizeDecision Action")

        # Get SA-Advanced report if available
        sa_adv_report = self._investigation_reports.get(ticker, None)

        if sa_adv_report:
            logger.info(f"   Using SA-Advanced report for conflict resolution")
        else:
            logger.info(f"   No SA-Advanced report - using basic reports only")

        # Call SynthesizeDecision Action (delegated reasoning)
        strategy_decision = await self._synthesize_decision_action.run(
            ticker=ticker,
            ra=state.fa_data,
            ta=state.ta_data,
            sa=state.sa_data,
            sa_adv=sa_adv_report,
            conflict_issue=conflict_issue
        )

        # Update environment
        self.env.update_final_decision(strategy_decision)

        # Log final decision
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 FINAL DECISION for {ticker}")
        logger.info(f"{'='*70}")
        logger.info(f"Action: {strategy_decision.final_action.value}")
        logger.info(f"Confidence: {strategy_decision.confidence_score}%")
        logger.info(f"Module: {strategy_decision.suggested_module}")
        logger.info(f"Summary: {strategy_decision.decision_summary}")
        logger.info(f"Conflict Report: {strategy_decision.conflict_report}")
        logger.info(f"{'='*70}\n")

        # Publish decision
        return self.publish_message(
            report=strategy_decision,
            cause_by=PublishStrategyDecision
        )
