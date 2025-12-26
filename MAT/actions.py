"""
Filename: MetaGPT-Ewan/MAT/actions.py
Created Date: Friday, December 26th 2025
Author: Ewan Su
Description: Action definitions for the investment agent workflow.
"""

from metagpt.actions import Action


class StartAnalysis(Action):
    """
    Trigger action to start the investment analysis workflow.
    
    This action is published by the orchestrator/Team to signal that
    all analysts (RA, TA, SA) should begin their analysis for a given ticker.
    """
    name: str = "StartAnalysis"
    
    async def run(self, ticker: str, context: str = "") -> str:
        """
        Initiate the analysis process.
        
        Args:
            ticker: The stock ticker symbol to analyze
            context: Optional additional context or requirements
            
        Returns:
            A message indicating the analysis has started
        """
        return f"Starting investment analysis for {ticker}. {context}"


class PublishFAReport(Action):
    """
    Action for publishing Fundamental Analysis reports.
    
    The Research Analyst (RA) uses this action to publish FAReport.
    """
    name: str = "PublishFAReport"


class PublishTAReport(Action):
    """
    Action for publishing Technical Analysis reports.
    
    The Technical Analyst (TA) uses this action to publish TAReport.
    """
    name: str = "PublishTAReport"


class PublishSAReport(Action):
    """
    Action for publishing Sentiment Analysis reports.
    
    The Sentiment Analyst (SA) uses this action to publish SAReport.
    """
    name: str = "PublishSAReport"


class PublishStrategyDecision(Action):
    """
    Action for publishing the final strategy decision.
    
    The Alpha Strategist (AS) uses this action to publish StrategyDecision
    after synthesizing all analyst reports.
    """
    name: str = "PublishStrategyDecision"


class RequestInvestigation(Action):
    """
    Action for Alpha Strategist to request a deep dive investigation.
    
    Used in Scheme C (Active Inquiry) when conflicts are detected between
    fundamental/technical signals and sentiment analysis.
    """
    name: str = "RequestInvestigation"
    
    async def run(self, ticker: str, context_issue: str, importance_level: int = 1) -> str:
        """
        Request a deep dive investigation.
        
        Args:
            ticker: The stock ticker symbol
            context_issue: Description of the conflict or issue to investigate
            importance_level: 1 for normal, 2 for high importance
            
        Returns:
            A message describing the investigation request
        """
        return f"Requesting deep dive investigation for {ticker}: {context_issue} (Level {importance_level})"


class PublishInvestigationReport(Action):
    """
    Action for publishing deep dive investigation results.
    
    The Sentiment Analyst (SA) uses this action to publish InvestigationReport
    after performing a targeted deep dive analysis.
    """
    name: str = "PublishInvestigationReport"

