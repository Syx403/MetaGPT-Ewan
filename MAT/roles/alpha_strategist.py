"""
Filename: MetaGPT-Ewan/MAT/roles/alpha_strategist.py
Created Date: Saturday, December 27th 2025
Author: Ewan Su
Description: Alpha Strategist implementing Scheme C (Active Inquiry) with dynamic conflict resolution.
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
    SignalIntensity,
    TradingState
)
from ..actions import (
    PublishFAReport,
    PublishTAReport,
    PublishSAReport,
    PublishInvestigationReport,
    RequestInvestigation,
    PublishStrategyDecision
)


class AlphaStrategist(BaseInvestmentAgent):
    """
    Alpha Strategist (AS) implementing Scheme C: Active Inquiry with dynamic conflict resolution.
    
    Key Features:
    1. Observes all analyst reports (FA, TA, SA) and investigation reports
    2. Detects conflicts between fundamental/technical signals and sentiment
    3. Dynamically requests deep dive investigations when conflicts arise
    4. Tracks retry counts per ticker to prevent infinite loops
    5. Makes final "Safe-First" decisions when conflicts remain unresolved
    
    Workflow:
    - Collect FA, TA, SA reports
    - Detect signal conflicts (bullish fundamentals/technicals vs negative sentiment)
    - Calculate importance level based on growth metrics
    - Request investigation if conflict exists and retries available
    - Make final decision when signals align or retries exhausted
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="AlphaStrategist",
            profile="Alpha Strategist",
            goal="Synthesize multi-dimensional analysis into actionable trading decisions",
            constraints="Use Safe-First approach when conflicts remain; prioritize capital preservation",
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
        
        # Internal state tracking per ticker
        self._ticker_states: Dict[str, TradingState] = {}
        self._retry_counts: Dict[str, int] = {}  # Track investigation retries per ticker
        self._pending_investigations: Dict[str, InvestigationRequest] = {}  # Track active investigations
        
        logger.info("🧠 Alpha Strategist initialized with Scheme C (Active Inquiry)")
    
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
            self._retry_counts[ticker] = 0
            logger.debug(f"📊 Created new state for ticker: {ticker}")
        
        return self._ticker_states[ticker]
    
    async def _act(self) -> Message:
        """
        Main action logic: process reports, detect conflicts, and make decisions.
        
        Returns:
            Message containing either InvestigationRequest or StrategyDecision
        """
        # Get the newly observed messages (from _observe)
        news = self.rc.news
        if not news:
            logger.debug("⏳ AlphaStrategist: No new messages to process")
            return None
        
        # Process all recent messages to update internal state
        for msg in news:
            await self._process_message(msg)
        
        # Check if we have a ticker to work with
        if not self._current_ticker:
            logger.warning("⚠️ No current ticker set")
            return None
        
        state = self._get_or_create_state(self._current_ticker)
        
        # Check if we're waiting for an investigation response
        if self._current_ticker in self._pending_investigations:
            if state.sa_data is None:
                logger.debug(f"⏳ Waiting for investigation response for {self._current_ticker}")
                return None
        
        # Check if we have all required reports (FA, TA, SA)
        if not self._has_minimum_reports(state):
            logger.debug(f"⏳ Waiting for more reports for {self._current_ticker}")
            return None
        
        # Perform conflict detection and decision logic
        return await self._make_decision(state)
    
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
            
            state = self._get_or_create_state(ticker)
            
            # Get the cause_by string for comparison
            cause_by_str = str(message.cause_by) if not isinstance(message.cause_by, str) else message.cause_by
            
            # Update state based on message type
            # Note: cause_by might be a string like "MAT.actions.PublishFAReport" or the class itself
            if message.cause_by == PublishFAReport or cause_by_str.endswith("PublishFAReport"):
                state.fa_data = FAReport(**content)
                logger.info(f"📈 Received FA Report for {ticker}")
                
            elif message.cause_by == PublishTAReport or cause_by_str.endswith("PublishTAReport"):
                state.ta_data = TAReport(**content)
                logger.info(f"📊 Received TA Report for {ticker}")
                
            elif message.cause_by == PublishSAReport or cause_by_str.endswith("PublishSAReport"):
                state.sa_data = SAReport(**content)
                logger.info(f"📰 Received SA Report for {ticker}")
                # Clear pending investigation if this was a response
                if ticker in self._pending_investigations:
                    logger.info(f"✅ Investigation response received for {ticker}")
                    del self._pending_investigations[ticker]
                    
            elif message.cause_by == PublishInvestigationReport:
                # Process investigation report - update SA data with revised findings
                inv_report = InvestigationReport(**content)
                logger.info(f"🔍 Received Investigation Report for {ticker}")
                
                # Update sentiment with revised findings
                if state.sa_data:
                    state.sa_data.sentiment_score = inv_report.revised_sentiment_score
                    state.sa_data.news_summary = f"[REVISED] {inv_report.detailed_findings}"
                
                # Clear pending investigation
                if ticker in self._pending_investigations:
                    del self._pending_investigations[ticker]
                    
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def _has_minimum_reports(self, state: TradingState) -> bool:
        """
        Check if we have minimum required reports to make a decision.
        
        Args:
            state: TradingState to check
            
        Returns:
            True if FA, TA, and SA reports are available
        """
        return (
            state.fa_data is not None and
            state.ta_data is not None and
            state.sa_data is not None
        )
    
    def _detect_conflict(self, state: TradingState) -> Optional[str]:
        """
        Detect conflicts between fundamental/technical signals and sentiment.
        
        Conflict exists when:
        - FA shows growth_healthy = True (bullish fundamentals)
        - TA shows BUY or STRONG_BUY signal (bullish technicals)
        - SA shows negative or unclear sentiment (< 0 or close to 0)
        
        Args:
            state: TradingState with all reports
            
        Returns:
            Conflict description if detected, None otherwise
        """
        fa = state.fa_data
        ta = state.ta_data
        sa = state.sa_data
        
        # Check if fundamentals and technicals are bullish
        is_fa_bullish = fa.is_growth_healthy
        is_ta_bullish = ta.technical_signal in [SignalIntensity.BUY, SignalIntensity.STRONG_BUY]
        
        # Check if sentiment is negative or unclear
        is_sentiment_negative = sa.sentiment_score < 0.1  # Below 0.1 considered negative/unclear
        
        if is_fa_bullish and is_ta_bullish and is_sentiment_negative:
            conflict_desc = (
                f"CONFLICT DETECTED: Fundamentals (healthy={fa.is_growth_healthy}, "
                f"revenue_growth={fa.revenue_growth_yoy:.2%}) and Technicals (signal={ta.technical_signal.value}, "
                f"RSI={ta.rsi_14:.1f}) are BULLISH, but Sentiment is NEGATIVE/UNCLEAR "
                f"(score={sa.sentiment_score:.2f})"
            )
            logger.warning(f"⚠️ {conflict_desc}")
            return conflict_desc
        
        return None
    
    def _calculate_importance_level(self, state: TradingState) -> tuple[int, int]:
        """
        Calculate importance level and max retries based on fundamental metrics.
        
        Logic:
        - If revenue_growth_yoy > 0.3 (30%), importance_level=2, max_retries=2
        - Otherwise, importance_level=1, max_retries=1
        
        Args:
            state: TradingState with FA report
            
        Returns:
            Tuple of (importance_level, max_retries)
        """
        revenue_growth = state.fa_data.revenue_growth_yoy
        
        if revenue_growth > 0.3:
            importance_level = 2
            max_retries = 2
            logger.info(f"💎 High importance detected: revenue_growth={revenue_growth:.1%} > 30%")
        else:
            importance_level = 1
            max_retries = 1
            logger.info(f"📊 Normal importance: revenue_growth={revenue_growth:.1%}")
        
        return importance_level, max_retries
    
    async def _make_decision(self, state: TradingState) -> Optional[Message]:
        """
        Make final decision or request investigation based on conflict detection.
        
        Decision Logic:
        1. Detect conflicts between signals
        2. If conflict exists and retries available, request investigation
        3. If no conflict or retries exhausted, make final decision
        4. Use "Safe-First" approach if unresolved conflicts remain
        
        Args:
            state: TradingState with all reports
            
        Returns:
            Message with InvestigationRequest or StrategyDecision
        """
        ticker = state.current_ticker
        current_retry = self._retry_counts[ticker]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🧠 ALPHA STRATEGIST THINKING PROCESS for {ticker}")
        logger.info(f"{'='*70}")
        
        # Step 1: Detect conflicts
        conflict = self._detect_conflict(state)
        
        if conflict:
            # Step 2: Calculate importance and max retries
            importance_level, max_retries = self._calculate_importance_level(state)
            
            logger.info(f"📋 Conflict Analysis:")
            logger.info(f"   - Conflict: {conflict}")
            logger.info(f"   - Current Retry: {current_retry}")
            logger.info(f"   - Max Retries: {max_retries}")
            logger.info(f"   - Importance Level: {importance_level}")
            
            # Step 3: Decide whether to investigate or finalize
            if current_retry < max_retries:
                # Request investigation
                logger.info(f"🔍 Decision: INITIATE DEEP DIVE (Attempt {current_retry + 1}/{max_retries})")
                return await self._request_investigation(state, conflict, importance_level, max_retries)
            else:
                # Retries exhausted, make safe decision
                logger.warning(f"⚠️ Retries exhausted ({current_retry}/{max_retries}), proceeding with SAFE-FIRST decision")
                return await self._finalize_decision(state, conflict_unresolved=True)
        else:
            # No conflict, make decision
            logger.info("✅ No conflicts detected, signals are aligned")
            return await self._finalize_decision(state, conflict_unresolved=False)
    
    async def _request_investigation(
        self,
        state: TradingState,
        conflict: str,
        importance_level: int,
        max_retries: int
    ) -> Message:
        """
        Request a deep dive investigation from the Sentiment Analyst.
        
        Args:
            state: TradingState with all reports
            conflict: Description of the detected conflict
            importance_level: 1 for normal, 2 for high
            max_retries: Maximum number of retries allowed
            
        Returns:
            Message with InvestigationRequest
        """
        ticker = state.current_ticker
        current_retry = self._retry_counts[ticker]
        
        # Create investigation request
        request = InvestigationRequest(
            ticker=ticker,
            target_agent="SA",
            context_issue=conflict,
            current_retry=current_retry,
            max_retries=max_retries,
            importance_level=importance_level
        )
        
        # Update retry count
        self._retry_counts[ticker] += 1
        self._pending_investigations[ticker] = request
        
        logger.info(f"📤 Publishing InvestigationRequest for {ticker}")
        logger.info(f"   Context: {conflict[:100]}...")
        
        # Publish investigation request
        message = await self.publish_message(
            report=request,
            cause_by=RequestInvestigation
        )
        
        # Update environment
        self.env.publish_message(message)
        
        return message
    
    async def _finalize_decision(
        self,
        state: TradingState,
        conflict_unresolved: bool = False
    ) -> Message:
        """
        Make the final trading decision and publish StrategyDecision.
        
        Uses "Safe-First" approach:
        - If conflict unresolved: NEUTRAL (preserve capital)
        - If signals aligned: Use weighted synthesis of FA, TA, SA
        
        Args:
            state: TradingState with all reports
            conflict_unresolved: Whether an unresolved conflict remains
            
        Returns:
            Message with StrategyDecision
        """
        ticker = state.current_ticker
        fa = state.fa_data
        ta = state.ta_data
        sa = state.sa_data
        
        logger.info(f"🎯 FINALIZING DECISION for {ticker}")
        
        # Build logic chain
        logic_chain = []
        logic_chain.append(f"FA: Growth {'HEALTHY' if fa.is_growth_healthy else 'WEAK'} (Revenue YoY: {fa.revenue_growth_yoy:.1%}, Margin: {fa.gross_margin:.1%})")
        logic_chain.append(f"TA: Signal={ta.technical_signal.value}, RSI={ta.rsi_14:.1f}, BB_Touch={ta.bb_lower_touch}, MA200_Dist={ta.price_to_ma200_dist:.1%}")
        logic_chain.append(f"SA: Sentiment={sa.sentiment_score:.2f}, Events={[e.value for e in sa.impactful_events]}")
        
        if conflict_unresolved:
            # Safe-First: NEUTRAL when conflict remains
            final_action = SignalIntensity.NEUTRAL
            confidence = 40.0  # Low confidence due to unresolved conflict
            logic_chain.append("⚠️ CONFLICT UNRESOLVED: Fundamentals/Technicals bullish but Sentiment negative")
            logic_chain.append("🛡️ SAFE-FIRST APPROACH: Position=NEUTRAL to preserve capital")
            risk_notes = "Unresolved conflict between signals. Monitor closely for sentiment shift. No position recommended."
            suggested_module = "NoAction"
            
            logger.warning(f"🛡️ Safe-First Decision: NEUTRAL (confidence={confidence}%)")
            
        else:
            # Signals aligned: Synthesize decision
            final_action, confidence, synthesis_logic = self._synthesize_aligned_signals(state)
            logic_chain.extend(synthesis_logic)
            
            # Generate risk notes
            risk_notes = self._generate_risk_notes(state, final_action)
            
            # Suggest execution module
            suggested_module = self._suggest_execution_module(final_action, ta)
            
            logger.info(f"✅ Aligned Decision: {final_action.value} (confidence={confidence}%)")
        
        # Create strategy decision
        decision = StrategyDecision(
            ticker=ticker,
            final_action=final_action,
            confidence_score=confidence,
            logic_chain=logic_chain,
            risk_notes=risk_notes,
            suggested_module=suggested_module
        )
        
        # Update environment
        self.env.update_final_decision(decision)
        
        # Log final decision
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 FINAL DECISION for {ticker}")
        logger.info(f"{'='*70}")
        logger.info(f"Action: {final_action.value}")
        logger.info(f"Confidence: {confidence}%")
        logger.info(f"Module: {suggested_module}")
        for i, step in enumerate(logic_chain, 1):
            logger.info(f"{i}. {step}")
        logger.info(f"Risk Notes: {risk_notes}")
        logger.info(f"{'='*70}\n")
        
        # Publish decision
        return await self.publish_message(
            report=decision,
            cause_by=PublishStrategyDecision
        )
    
    def _synthesize_aligned_signals(self, state: TradingState) -> tuple[SignalIntensity, float, list[str]]:
        """
        Synthesize final action when all signals are aligned.
        
        Weighting:
        - FA: 40% (long-term health)
        - TA: 30% (timing)
        - SA: 30% (catalysts)
        
        Args:
            state: TradingState with all reports
            
        Returns:
            Tuple of (final_action, confidence_score, logic_steps)
        """
        fa = state.fa_data
        ta = state.ta_data
        sa = state.sa_data
        
        logic = []
        
        # Calculate component scores (-2 to +2 scale)
        fa_score = 2 if fa.is_growth_healthy else -1
        if fa.revenue_growth_yoy > 0.2:
            fa_score += 1
        
        ta_signal_map = {
            SignalIntensity.STRONG_BUY: 2,
            SignalIntensity.BUY: 1,
            SignalIntensity.NEUTRAL: 0,
            SignalIntensity.SELL: -1,
            SignalIntensity.STRONG_SELL: -2
        }
        ta_score = ta_signal_map[ta.technical_signal]
        
        sa_score = sa.sentiment_score * 2  # Convert -1 to 1 → -2 to 2
        
        # Weighted synthesis
        weighted_score = (fa_score * 0.4) + (ta_score * 0.3) + (sa_score * 0.3)
        
        logic.append(f"Weighted Synthesis: FA_Score={fa_score:.1f}*0.4 + TA_Score={ta_score:.1f}*0.3 + SA_Score={sa_score:.1f}*0.3 = {weighted_score:.2f}")
        
        # Map to signal intensity
        if weighted_score >= 1.5:
            final_action = SignalIntensity.STRONG_BUY
            confidence = min(95, 70 + weighted_score * 10)
        elif weighted_score >= 0.5:
            final_action = SignalIntensity.BUY
            confidence = min(85, 60 + weighted_score * 10)
        elif weighted_score >= -0.5:
            final_action = SignalIntensity.NEUTRAL
            confidence = 55
        elif weighted_score >= -1.5:
            final_action = SignalIntensity.SELL
            confidence = min(85, 60 + abs(weighted_score) * 10)
        else:
            final_action = SignalIntensity.STRONG_SELL
            confidence = min(95, 70 + abs(weighted_score) * 10)
        
        logic.append(f"Final Synthesis: weighted_score={weighted_score:.2f} → Action={final_action.value}")
        
        return final_action, confidence, logic
    
    def _generate_risk_notes(self, state: TradingState, action: SignalIntensity) -> str:
        """
        Generate risk management notes based on the decision.
        
        Args:
            state: TradingState with all reports
            action: Final action decided
            
        Returns:
            Risk management guidance string
        """
        ta = state.ta_data
        fa = state.fa_data
        
        notes = []
        
        # Stop-loss based on ATR
        if action in [SignalIntensity.BUY, SignalIntensity.STRONG_BUY]:
            stop_loss_pct = ta.volatility_atr * 2
            notes.append(f"Set stop-loss at -{stop_loss_pct:.1f}% (2x ATR)")
            notes.append(f"Position size: {'5%' if action == SignalIntensity.STRONG_BUY else '3%'} of portfolio")
        
        # Key risks from FA
        if fa.key_risks:
            notes.append(f"Monitor risks: {', '.join(fa.key_risks[:2])}")
        
        # Volatility warning
        if ta.volatility_atr > 3.0:
            notes.append("⚠️ High volatility - consider smaller position size")
        
        return " | ".join(notes) if notes else "Standard risk management applies"
    
    def _suggest_execution_module(self, action: SignalIntensity, ta_report: TAReport) -> str:
        """
        Suggest execution module based on action and technical conditions.
        
        Args:
            action: Final trading action
            ta_report: Technical analysis report
            
        Returns:
            Suggested execution module name
        """
        if action in [SignalIntensity.BUY, SignalIntensity.STRONG_BUY]:
            if ta_report.bb_lower_touch:
                return "MeanReversionLong"
            else:
                return "TrendFollowingLong"
        elif action in [SignalIntensity.SELL, SignalIntensity.STRONG_SELL]:
            return "ExitPosition"
        else:
            return "NoAction"

