"""
Filename: MetaGPT-Ewan/MAT/roles/base_agent.py
Created Date: Friday, December 26th 2025
Author: Ewan Su
Description: Base agent class for all investment analysts using MetaGPT's pub-sub pattern.
"""

from typing import Type, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel

from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.actions import Action
from metagpt.logs import logger

from ..environment import InvestmentEnvironment


class BaseInvestmentAgent(Role, ABC):
    """
    Abstract base class for all investment agents (RA, TA, SA, AS).
    
    This class provides:
    1. Standardized message observation filtering by ticker
    2. A publish_message() helper to wrap Pydantic models into MetaGPT Messages
    3. Template methods for subscribing to and reacting to specific message types
    
    All concrete agents (RA, TA, SA, AS) should inherit from this class.
    """
    
    def __init__(
        self,
        name: str,
        profile: str,
        goal: str,
        constraints: str = "",
        **kwargs
    ):
        """
        Initialize a base investment agent.
        
        Args:
            name: Agent's name (e.g., "ResearchAnalyst")
            profile: Agent's profile/role description
            goal: Agent's primary objective
            constraints: Optional constraints for the agent's behavior
            **kwargs: Additional arguments passed to the Role base class
        """
        super().__init__(name=name, profile=profile, goal=goal, constraints=constraints, **kwargs)
        self._current_ticker: Optional[str] = None
        logger.info(f"🤖 {profile} '{name}' initialized")
    
    @property
    def env(self) -> InvestmentEnvironment:
        """
        Get the investment environment with proper type hint.
        
        Returns:
            The InvestmentEnvironment this agent operates in
        """
        return self.rc.env
    
    def set_ticker(self, ticker: str):
        """
        Set the ticker symbol this agent should focus on.
        
        Args:
            ticker: The stock ticker symbol (e.g., "AAPL", "TSLA")
        """
        self._current_ticker = ticker
        logger.debug(f"🎯 {self.profile} now tracking: {ticker}")
    
    async def _observe(self) -> int:
        """
        Observe messages from the environment and filter by ticker context.
        
        This method:
        1. Calls the parent's _observe() to get new messages into rc.news
        2. Filters messages to only keep those relevant to the current ticker
        3. Returns the count of relevant messages
        
        Returns:
            Number of messages observed that are relevant to this agent
        """
        # Call parent's observe method to populate rc.news from msg_buffer
        count_before = await super()._observe()
        logger.debug(f"🔍 {self.profile}._observe() - After super(), rc.news has {len(self.rc.news)} messages")
        
        # Filter rc.news by ticker if we have a current ticker set
        if self._current_ticker and self.env.trading_state:
            # Only keep messages for the current ticker
            relevant_messages = []
            for msg in self.rc.news:
                # Check if message is for the current ticker
                is_relevant = self._is_message_relevant(msg)
                if is_relevant:
                    relevant_messages.append(msg)
            
            # Replace rc.news with filtered messages
            self.rc.news = relevant_messages
            
            if relevant_messages:
                logger.debug(f"🔍 {self.profile} filtered to {len(relevant_messages)} relevant messages for {self._current_ticker}")
            
            return len(relevant_messages)
        
        # No filter applied
        return len(self.rc.news)
    
    def _is_message_relevant(self, message: Message) -> bool:
        """
        Determine if a message is relevant to this agent based on ticker context.
        
        This can be overridden by subclasses for custom filtering logic.
        
        Args:
            message: The message to check for relevance
            
        Returns:
            True if the message is relevant to the current ticker, False otherwise
        """
        # By default, check if the current ticker matches the environment's ticker
        if not self.env.trading_state:
            return True  # No state to filter by, process all messages
        
        # Check if message content contains the ticker (simple heuristic)
        # Subclasses can override this for more sophisticated filtering
        return (
            self._current_ticker == self.env.trading_state.current_ticker or
            self._current_ticker in message.content
        )
    
    async def publish_message(
        self,
        report: BaseModel,
        cause_by: Type[Action],
        send_to: str = ""
    ) -> Message:
        """
        Wrap a Pydantic report model into a MetaGPT Message and publish it.
        
        This is the standard way for agents to publish their reports to the environment.
        The report will be:
        1. Serialized to JSON
        2. Wrapped in a Message object with proper metadata
        3. Published to the environment for other agents to observe
        
        Args:
            report: A Pydantic model (FAReport, TAReport, SAReport, or StrategyDecision)
            cause_by: The Action class that generated this report
            send_to: Optional specific recipient role name
            
        Returns:
            The Message object that was published
        """
        # Serialize the Pydantic model to JSON for message content
        message_content = report.model_dump_json(indent=2)
        
        # Create the MetaGPT Message
        message = Message(
            content=message_content,
            role=self.profile,
            cause_by=cause_by,
            send_to=send_to,
            instruct_content=None
        )
        
        logger.info(f"📤 {self.profile} publishing {cause_by.__name__} for {report.ticker}")
        
        # Publish to the environment
        self.env.publish_message(message)
        
        return message
    
    @abstractmethod
    async def _act(self) -> Message:
        """
        Abstract method for agent's action logic.
        
        Each concrete agent (RA, TA, SA, AS) must implement this method to:
        1. Retrieve relevant data from the environment or messages
        2. Perform analysis (via LLM or computation)
        3. Generate a typed report (FAReport, TAReport, etc.)
        4. Publish the report using publish_message()
        
        Returns:
            The Message containing the agent's report
        """
        pass
    
    def get_trading_state_context(self) -> str:
        """
        Get a formatted string of the current trading state for use in prompts.
        
        This helper method is useful for agents that need context from other
        agents' reports when making their own analysis.
        
        Returns:
            A formatted string summarizing the trading state
        """
        if not self.env.trading_state:
            return "No trading state available."
        
        state = self.env.trading_state
        context = f"Current Ticker: {state.current_ticker}\n\n"
        
        if state.fa_data:
            context += f"Fundamental Analysis:\n"
            context += f"  - Revenue Growth YoY: {state.fa_data.revenue_growth_yoy:.2%}\n"
            context += f"  - Gross Margin: {state.fa_data.gross_margin:.2%}\n"
            context += f"  - Growth Healthy: {state.fa_data.is_growth_healthy}\n\n"
        
        if state.ta_data:
            context += f"Technical Analysis:\n"
            context += f"  - RSI(14): {state.ta_data.rsi_14:.2f}\n"
            context += f"  - BB Lower Touch: {state.ta_data.bb_lower_touch}\n"
            context += f"  - Signal: {state.ta_data.technical_signal.value}\n\n"
        
        if state.sa_data:
            context += f"Sentiment Analysis:\n"
            context += f"  - Sentiment Score: {state.sa_data.sentiment_score:.2f}\n"
            context += f"  - Events: {', '.join([e.value for e in state.sa_data.impactful_events])}\n\n"
        
        return context

