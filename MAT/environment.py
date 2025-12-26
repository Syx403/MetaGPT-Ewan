"""
Filename: MetaGPT-Ewan/MAT/environment.py
Created Date: Friday, December 26th 2025
Author: Ewan Su
Description: Investment Environment for managing shared trading state and agent communication.
"""

from typing import Optional
from metagpt.environment import Environment
from metagpt.schema import Message
from metagpt.logs import logger

from .schemas import TradingState, FAReport, TAReport, SAReport, StrategyDecision


class InvestmentEnvironment(Environment):
    """
    Custom environment for investment agents that maintains a shared TradingState.
    
    This environment extends MetaGPT's base Environment to:
    1. Store and update the global TradingState as agents publish reports
    2. Handle message propagation using the publish-subscribe pattern
    3. Provide state access to all agents in the system
    """
    
    def __init__(self, desc: str = "Investment Analysis Environment", **kwargs):
        """
        Initialize the investment environment.
        
        Args:
            desc: Description of the environment
            **kwargs: Additional arguments passed to base Environment
        """
        super().__init__(desc=desc, **kwargs)
        self._trading_state: Optional[TradingState] = None
        logger.info(f"📊 {desc} initialized")
    
    def set_ticker(self, ticker: str):
        """
        Initialize or reset the trading state for a new ticker symbol.
        
        Args:
            ticker: The stock ticker symbol to analyze (e.g., "AAPL", "TSLA")
        """
        self._trading_state = TradingState(current_ticker=ticker)
        logger.info(f"🎯 Trading state initialized for ticker: {ticker}")
    
    @property
    def trading_state(self) -> Optional[TradingState]:
        """
        Get the current trading state.
        
        Returns:
            The current TradingState or None if not initialized
        """
        return self._trading_state
    
    def update_fa_report(self, report: FAReport):
        """
        Update the trading state with a new fundamental analysis report.
        
        Args:
            report: FAReport from the Research Analyst (RA)
        """
        if self._trading_state is None:
            logger.warning("⚠️ Cannot update FA report: trading state not initialized")
            return
        
        self._trading_state.fa_data = report
        logger.info(f"✅ FA Report updated for {report.ticker}")
    
    def update_ta_report(self, report: TAReport):
        """
        Update the trading state with a new technical analysis report.
        
        Args:
            report: TAReport from the Technical Analyst (TA)
        """
        if self._trading_state is None:
            logger.warning("⚠️ Cannot update TA report: trading state not initialized")
            return
        
        self._trading_state.ta_data = report
        logger.info(f"✅ TA Report updated for {report.ticker}")
    
    def update_sa_report(self, report: SAReport):
        """
        Update the trading state with a new sentiment analysis report.
        
        Args:
            report: SAReport from the Sentiment Analyst (SA)
        """
        if self._trading_state is None:
            logger.warning("⚠️ Cannot update SA report: trading state not initialized")
            return
        
        self._trading_state.sa_data = report
        logger.info(f"✅ SA Report updated for {report.ticker}")
    
    def update_final_decision(self, decision: StrategyDecision):
        """
        Update the trading state with the final strategy decision.
        
        Args:
            decision: StrategyDecision from the Alpha Strategist (AS)
        """
        if self._trading_state is None:
            logger.warning("⚠️ Cannot update decision: trading state not initialized")
            return
        
        self._trading_state.final_decision = decision
        logger.info(f"🎯 Final Decision updated for {decision.ticker}: {decision.final_action.value}")
    
    def is_analysis_complete(self) -> bool:
        """
        Check if all required reports have been collected.
        
        Returns:
            True if FA, TA, and SA reports are all available
        """
        if self._trading_state is None:
            return False
        
        return (
            self._trading_state.fa_data is not None and
            self._trading_state.ta_data is not None and
            self._trading_state.sa_data is not None
        )
    
    def publish_message(self, message: Message):
        """
        Publish a message to the environment for all subscribed agents.
        
        This method wraps the base environment's publish mechanism and
        adds logging for better observability.
        
        Args:
            message: The Message to broadcast to all agents
        """
        logger.info(f"📤 Publishing message: {message.cause_by} for ticker {self._trading_state.current_ticker if self._trading_state else 'N/A'}")
        super().publish_message(message)
    
    def get_state_summary(self) -> str:
        """
        Get a human-readable summary of the current trading state.
        
        Returns:
            A formatted string summarizing the state of all reports
        """
        if self._trading_state is None:
            return "❌ No trading state initialized"
        
        summary = f"📊 Trading State for {self._trading_state.current_ticker}:\n"
        summary += f"  - FA Report: {'✅' if self._trading_state.fa_data else '❌'}\n"
        summary += f"  - TA Report: {'✅' if self._trading_state.ta_data else '❌'}\n"
        summary += f"  - SA Report: {'✅' if self._trading_state.sa_data else '❌'}\n"
        summary += f"  - Final Decision: {'✅' if self._trading_state.final_decision else '❌'}\n"
        
        return summary

