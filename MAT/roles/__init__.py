"""
Filename: MetaGPT-Ewan/MAT/roles/__init__.py
Created Date: Friday, December 26th 2025
Author: Ewan Su
Description: Package initialization for investment agent roles.
"""

from .base_agent import BaseInvestmentAgent
from .alpha_strategist import AlphaStrategist
from .sentiment_analyst import SentimentAnalyst

__all__ = [
    "BaseInvestmentAgent",
    "AlphaStrategist",
    "SentimentAnalyst"
]

