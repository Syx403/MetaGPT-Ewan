"""
Filename: MetaGPT-Ewan/MAT/__init__.py
Created Date: Friday, December 26th 2025
Author: Ewan Su
Description: Multi-Agent Trading (MAT) Framework - Core package initialization.
"""

from .schemas import (
    SignalIntensity,
    MarketEvent,
    FAReport,
    TAReport,
    SAReport,
    StrategyDecision,
    TradingState,
    InvestigationRequest,
    InvestigationReport
)
from .environment import InvestmentEnvironment
from .roles import BaseInvestmentAgent, AlphaStrategist, SentimentAnalyst
from .actions import (
    StartAnalysis,
    PublishFAReport,
    PublishTAReport,
    PublishSAReport,
    PublishStrategyDecision,
    RequestInvestigation,
    PublishInvestigationReport
)

__all__ = [
    # Enums
    "SignalIntensity",
    "MarketEvent",
    # Report Schemas
    "FAReport",
    "TAReport",
    "SAReport",
    "StrategyDecision",
    "TradingState",
    "InvestigationRequest",
    "InvestigationReport",
    # Core Framework
    "InvestmentEnvironment",
    "BaseInvestmentAgent",
    # Concrete Agents (Scheme C)
    "AlphaStrategist",
    "SentimentAnalyst",
    # Actions
    "StartAnalysis",
    "PublishFAReport",
    "PublishTAReport",
    "PublishSAReport",
    "PublishStrategyDecision",
    "RequestInvestigation",
    "PublishInvestigationReport",
]

__version__ = "0.1.0"

