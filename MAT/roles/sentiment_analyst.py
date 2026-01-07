"""
Filename: MetaGPT-Ewan/MAT/roles/sentiment_analyst.py
Created Date: Saturday, December 27th 2025
Author: Ewan Su
Description: Sentiment Analyst with dual-mode operation: normal analysis and deep dive investigation.

Updated: December 29th 2025
- Integrated SearchDeepDive action for Tavily-powered investigations
- Uses centralized config from config/config2.yaml
"""

from typing import List, Optional
from metagpt.schema import Message
from metagpt.logs import logger
from metagpt.tools.search_engine import SearchEngine
from metagpt.tools import SearchEngineType
import json
import re

from ..roles.base_agent import BaseInvestmentAgent
from ..schemas import (
    SAReport,
    InvestigationReport,
    InvestigationRequest,
    MarketEvent
)
from ..actions import (
    StartAnalysis,
    PublishSAReport,
    RequestInvestigation,
    PublishInvestigationReport
)
from ..actions.search_deep_dive import SearchDeepDive
from ..config_loader import get_config


class SentimentAnalyst(BaseInvestmentAgent):
    """
    Sentiment Analyst (SA) with dual-mode operation for Scheme C (Active Inquiry).
    
    Modes:
    1. Normal Mode: Regular news sentiment analysis (triggered by StartAnalysis)
       - Searches general news about the ticker
       - Analyzes sentiment and detects events
       - Publishes SAReport
    
    2. Deep Dive Mode: Targeted investigation (triggered by InvestigationRequest)
       - Performs focused search on specific conflict/issue
       - Provides more detailed findings
       - Publishes InvestigationReport with revised sentiment
    
    Features:
    - Uses Tavily search engine for news gathering
    - LLM-based sentiment scoring and event detection
    - Keyword extraction for context
    - Dynamic response to AS inquiries
    """
    
    def __init__(self, use_tavily: bool = True, **kwargs):
        """
        Initialize Sentiment Analyst with dual-mode operation.
        
        Args:
            use_tavily: If True and Tavily API is configured in config/config2.yaml,
                       use Tavily for deep dive. Otherwise, falls back to DuckDuckGo.
            **kwargs: Additional arguments for BaseInvestmentAgent.
        """
        super().__init__(
            name="SentimentAnalyst",
            profile="Sentiment Analyst",
            goal="Analyze news sentiment and identify market-moving catalysts",
            constraints="Focus on recent news (last 7 days), verify sources, detect major events",
            **kwargs
        )
        
        # Subscribe to both normal trigger and investigation requests
        self._watch([StartAnalysis, RequestInvestigation])
        
        # Set action types
        self.set_actions([PublishSAReport, PublishInvestigationReport])
        
        # Initialize search engine (using DuckDuckGo for normal analysis)
        self._search_engine = SearchEngine(engine=SearchEngineType.DUCK_DUCK_GO)
        
        # Load config and check Tavily availability
        config = get_config()
        self._use_tavily = use_tavily and config.use_tavily()
        self._deep_dive_action: Optional[SearchDeepDive] = None
        
        if self._use_tavily:
            self._deep_dive_action = SearchDeepDive()
            logger.info("📰 Sentiment Analyst initialized with Tavily deep dive")
        else:
            logger.info("📰 Sentiment Analyst initialized (DuckDuckGo fallback for deep dive)")
            if use_tavily and not config.is_tavily_configured():
                logger.warning("⚠️ Tavily API key not configured in config/config2.yaml")
                logger.warning("   Using DuckDuckGo for investigations")
        
        # Track pending investigations
        self._investigation_mode = False
        self._current_investigation: Optional[InvestigationRequest] = None
        
        # Store last search results for demo/debugging
        self._last_search_results: List[dict] = []
    
    async def _act(self) -> Message:
        """
        Main action logic: perform normal or deep dive analysis based on trigger.
        
        Returns:
            Message with SAReport (normal mode) or InvestigationReport (deep dive mode)
        """
        # Get newly observed messages (from _observe)
        news = self.rc.news
        if not news:
            logger.debug("⏳ SentimentAnalyst: No messages to process")
            return None
        
        # Check the most recent message type to determine mode
        latest_msg = news[-1]
        cause_by_str = str(latest_msg.cause_by) if not isinstance(latest_msg.cause_by, str) else latest_msg.cause_by
        
        if latest_msg.cause_by == RequestInvestigation or cause_by_str.endswith("RequestInvestigation"):
            # Deep dive mode
            logger.info("🔍 DEEP DIVE MODE activated")
            return await self._deep_dive_investigation(latest_msg)
        
        elif latest_msg.cause_by == StartAnalysis or cause_by_str.endswith("StartAnalysis"):
            # Normal mode
            logger.info("📊 NORMAL MODE: Regular sentiment analysis")
            return await self._normal_analysis()
        
        logger.warning(f"⚠️ Unrecognized message type: {cause_by_str}")
        return None
    
    async def _normal_analysis(self) -> Message:
        """
        Perform normal sentiment analysis using SearchDeepDive in basic mode.

        Workflow:
        1. Use SearchDeepDive with mode="basic" (uses Tavily with search_depth="basic")
        2. Receives SAReport from the action
        3. Publishes SAReport

        Returns:
            Message with SAReport
        """
        ticker = self._current_ticker or self.env.trading_state.current_ticker
        if not ticker:
            logger.error("❌ No ticker specified for sentiment analysis")
            return self.publish_message(
                report=self._create_default_report("UNKNOWN"),
                cause_by=PublishSAReport
            )

        logger.info(f"📰 Starting normal sentiment analysis for {ticker}")

        # Use Tavily SearchDeepDive if available, otherwise fallback to DuckDuckGo
        if self._use_tavily and self._deep_dive_action:
            logger.info(f"🌐 Using Tavily API (mode=basic, search_depth=basic)")
            if self.system_reference_date:
                logger.info(f"🕰️  Time-Travel Mode: Reference date = {self.system_reference_date}")

            # Execute SearchDeepDive in basic mode
            report = await self._deep_dive_action.run(
                ticker=ticker,
                mode="basic",
                llm_callback=self.llm.aask,
                reference_date=self.system_reference_date
            )
        else:
            # Fallback: Use DuckDuckGo search
            logger.info(f"🦆 Using DuckDuckGo fallback")

            # Step 1: Search for news
            news_data = await self._search_news(ticker, query_type="general")

            if not news_data:
                logger.warning(f"⚠️ No news found for {ticker}, using neutral defaults")
                report = self._create_default_report(ticker)
            else:
                # Step 2: Analyze with LLM
                report = await self._analyze_sentiment_with_llm(ticker, news_data, mode="normal")

        # Update environment
        self.env.update_sa_report(report)

        logger.info(f"✅ SA Report completed for {ticker}")
        logger.info(f"   Qualitative Assessment: {report.qualitative_sentiment_assessment[:80]}...")
        logger.info(f"   Events: {[e.value for e in report.impactful_events]}")

        # Publish report
        return self.publish_message(
            report=report,
            cause_by=PublishSAReport
        )
    
    async def _deep_dive_investigation(self, request_msg: Message) -> Message:
        """
        Perform deep dive investigation using SearchDeepDive in advanced mode.

        This mode is triggered when the Alpha Strategist detects conflicts
        and needs more detailed sentiment analysis.

        Uses SearchDeepDive action with mode="advanced" (search_depth="advanced").
        If Tavily is not available, falls back to DuckDuckGo search.

        Args:
            request_msg: Message containing InvestigationRequest

        Returns:
            Message with InvestigationReport
        """
        try:
            request_data = json.loads(request_msg.content)
            investigation_req = InvestigationRequest(**request_data)

            ticker = investigation_req.ticker
            context_issue = investigation_req.context_issue
            importance_level = investigation_req.importance_level
            current_retry = investigation_req.current_retry

            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 DEEP DIVE INVESTIGATION for {ticker}")
            logger.info(f"{'='*70}")
            logger.info(f"Issue: {context_issue[:100]}...")
            logger.info(f"Importance Level: {importance_level}")
            logger.info(f"Retry: {current_retry + 1}/{investigation_req.max_retries}")

            # Use Tavily SearchDeepDive if available
            if self._use_tavily and self._deep_dive_action:
                logger.info(f"🌐 Using Tavily API (mode=advanced, search_depth=advanced)")
                if self.system_reference_date:
                    logger.info(f"🕰️  Time-Travel Mode: Reference date = {self.system_reference_date}")
                logger.info(f"{'='*70}\n")

                # Execute SearchDeepDive action in advanced mode with LLM callback
                report = await self._deep_dive_action.run(
                    investigation_request=investigation_req,
                    mode="advanced",
                    llm_callback=self.llm.aask,  # Pass LLM method for analysis
                    reference_date=self.system_reference_date
                )

                # Store search results for demo/debugging
                if hasattr(self._deep_dive_action, '_last_search_results'):
                    self._last_search_results = self._deep_dive_action._last_search_results

            else:
                # Fallback: Use DuckDuckGo search
                logger.info(f"🦆 Using DuckDuckGo fallback")
                logger.info(f"{'='*70}\n")

                # Step 1: Perform targeted news search
                search_query = self._build_investigation_query(ticker, context_issue)
                news_data = await self._search_news(
                    ticker,
                    query_type="deep_dive",
                    custom_query=search_query,
                    max_results=10 if importance_level == 2 else 5
                )

                if not news_data:
                    logger.warning("⚠️ Deep dive found no additional news")
                    report = InvestigationReport(
                        ticker=ticker,
                        detailed_findings="No significant new information found in deep dive search. Original sentiment analysis stands.",
                        qualitative_sentiment_revision="No significant new information found in deep dive search",
                        is_ambiguity_resolved=False,
                        risk_classification="INSUFFICIENT_DATA"
                    )
                else:
                    # Step 2: Analyze with LLM in investigation mode
                    report = await self._analyze_investigation_with_llm(
                        ticker,
                        news_data,
                        context_issue,
                        importance_level
                    )

            logger.info(f"✅ Investigation completed:")
            logger.info(f"   - Ambiguity Resolved: {report.is_ambiguity_resolved}")
            logger.info(f"   - Risk Classification: {report.risk_classification}")
            logger.info(f"   - Sentiment Revision: {report.qualitative_sentiment_revision[:80]}...")

            # Publish investigation report
            return self.publish_message(
                report=report,
                cause_by=PublishInvestigationReport
            )

        except Exception as e:
            logger.error(f"❌ Error in deep dive investigation: {e}")
            # Return default investigation report
            report = InvestigationReport(
                ticker=self._current_ticker or "UNKNOWN",
                detailed_findings=f"Investigation failed due to error: {str(e)}",
                qualitative_sentiment_revision="Investigation failed - unable to revise sentiment",
                is_ambiguity_resolved=False,
                risk_classification="INSUFFICIENT_DATA"
            )
            return await self.publish_message(
                report=report,
                cause_by=PublishInvestigationReport
            )
    
    async def _search_news(
        self,
        ticker: str,
        query_type: str = "general",
        custom_query: Optional[str] = None,
        max_results: int = 5
    ) -> List[dict]:
        """
        Search for news using Tavily search engine.
        
        Args:
            ticker: Stock ticker symbol
            query_type: "general" or "deep_dive"
            custom_query: Custom search query (for deep dive)
            max_results: Maximum number of results to return
            
        Returns:
            List of news articles with title, content, url, published_date
        """
        try:
            # Build search query
            if custom_query:
                query = custom_query
            elif query_type == "general":
                query = f"{ticker} stock news sentiment analysis latest"
            else:
                query = f"{ticker} stock news"
            
            logger.info(f"🔎 Searching news: '{query}'")
            
            # Use DuckDuckGo search (returns list of dicts when as_string=False)
            search_results = await self._search_engine.run(
                query,
                max_results=max_results,
                as_string=False  # Return list of dicts instead of string
            )
            
            if not search_results:
                logger.warning(f"No search results found for: {query}")
                return []
            
            # Parse results from DuckDuckGo format
            news_articles = []
            for result in search_results[:max_results]:
                # DuckDuckGo returns: {'title': ..., 'href': ..., 'body': ...}
                article = {
                    "title": result.get("title", ""),
                    "content": result.get("body", "")[:500],  # DuckDuckGo uses 'body' not 'content'
                    "url": result.get("href", ""),  # DuckDuckGo uses 'href' not 'url'
                    "score": 1.0  # DuckDuckGo doesn't provide relevance scores
                }
                news_articles.append(article)
            
            # Store for demo/debugging purposes
            self._last_search_results = news_articles
            
            logger.info(f"✅ Found {len(news_articles)} news articles")
            return news_articles
            
        except Exception as e:
            logger.error(f"❌ News search failed: {e}")
            return []
    
    def _build_investigation_query(self, ticker: str, context_issue: str) -> str:
        """
        Build a targeted search query for deep dive investigation.
        
        Args:
            ticker: Stock ticker symbol
            context_issue: The conflict/issue to investigate
            
        Returns:
            Optimized search query string
        """
        # Extract key terms from context issue
        keywords = ["sentiment", "news", "controversy", "concerns"]
        
        # Build query focusing on the conflict
        query = f"{ticker} stock {' '.join(keywords)} recent developments"
        
        logger.debug(f"Built investigation query: {query}")
        return query
    
    async def _analyze_sentiment_with_llm(
        self,
        ticker: str,
        news_data: List[dict],
        mode: str = "normal"
    ) -> SAReport:
        """
        Analyze sentiment using LLM in normal mode.
        
        Args:
            ticker: Stock ticker symbol
            news_data: List of news articles
            mode: Analysis mode
            
        Returns:
            SAReport with sentiment analysis
        """
        # Prepare news summary for LLM
        news_summary = "\n\n".join([
            f"Title: {article['title']}\nContent: {article['content']}"
            for article in news_data[:5]
        ])
        
        prompt = f"""
You are a professional sentiment analyst. Analyze the following news articles about {ticker} stock.

NEWS ARTICLES:
{news_summary}

TASK:
1. Provide an overall sentiment score from -1 (very negative) to +1 (very positive)
2. Identify any major market-moving events from these categories:
   - EARNINGS_CALL
   - PRODUCT_LAUNCH
   - REGULATORY_ACTION
   - ANALYST_UPGRADE
   - MACRO_ECONOMIC
   - NONE
3. Extract 3-5 key keywords that characterize the news
4. Write a brief summary (2-3 sentences)

OUTPUT FORMAT (JSON):
{{
    "sentiment_score": <float between -1 and 1>,
    "events": [<list of event types>],
    "keywords": [<list of 3-5 keywords>],
    "summary": "<brief summary>"
}}

Provide ONLY the JSON output, no additional text.
"""
        
        try:
            # Call LLM
            response = await self._aask(prompt)
            
            # Parse LLM response
            parsed = self._parse_llm_sentiment_response(response)
            
            # Create SAReport
            report = SAReport(
                ticker=ticker,
                sentiment_score=parsed["sentiment_score"],
                impactful_events=parsed["events"],
                top_keywords=parsed["keywords"],
                news_summary=parsed["summary"]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"❌ LLM sentiment analysis failed: {e}")
            return self._create_default_report(ticker)
    
    async def _analyze_investigation_with_llm(
        self,
        ticker: str,
        news_data: List[dict],
        context_issue: str,
        importance_level: int
    ) -> InvestigationReport:
        """
        Analyze news in deep dive investigation mode.
        
        Args:
            ticker: Stock ticker symbol
            news_data: List of news articles
            context_issue: The specific conflict to investigate
            importance_level: 1 or 2
            
        Returns:
            InvestigationReport with revised findings
        """
        news_summary = "\n\n".join([
            f"Title: {article['title']}\nContent: {article['content']}"
            for article in news_data[:10]
        ])
        
        prompt = f"""
You are a senior sentiment analyst conducting a DEEP DIVE investigation for {ticker} stock.

CONTEXT - DETECTED CONFLICT:
{context_issue}

INVESTIGATION IMPORTANCE LEVEL: {importance_level} (1=Normal, 2=High)

RECENT NEWS ARTICLES:
{news_summary}

INVESTIGATION TASK:
Your goal is to RESOLVE THE CONFLICT by finding additional context or information.

1. Analyze the news with focus on the conflict described above
2. Provide detailed findings that explain or clarify the sentiment discrepancy
3. Give a REVISED sentiment score (-1 to +1) based on deeper analysis
4. Determine if the ambiguity/conflict is now RESOLVED (true/false)

Consider:
- Are there hidden positive/negative factors not initially obvious?
- Is the negative sentiment temporary or structural?
- Are there upcoming catalysts that might shift sentiment?

OUTPUT FORMAT (JSON):
{{
    "detailed_findings": "<comprehensive explanation of findings, 3-4 sentences>",
    "revised_sentiment_score": <float between -1 and 1>,
    "is_ambiguity_resolved": <true or false>,
    "key_insights": [<list of 2-3 key insights>]
}}

Provide ONLY the JSON output, no additional text.
"""
        
        try:
            # Call LLM
            response = await self._aask(prompt)
            
            # Parse LLM response
            parsed = self._parse_llm_investigation_response(response)
            
            # Create InvestigationReport
            report = InvestigationReport(
                ticker=ticker,
                detailed_findings=parsed["detailed_findings"],
                revised_sentiment_score=parsed["revised_sentiment_score"],
                is_ambiguity_resolved=parsed["is_ambiguity_resolved"]
            )
            
            logger.info(f"📊 Deep Dive Results:")
            logger.info(f"   - Revised Sentiment: {report.revised_sentiment_score:.2f}")
            logger.info(f"   - Ambiguity Resolved: {report.is_ambiguity_resolved}")
            logger.info(f"   - Findings: {report.detailed_findings[:100]}...")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Investigation LLM analysis failed: {e}")
            return InvestigationReport(
                ticker=ticker,
                detailed_findings=f"Investigation analysis failed: {str(e)}",
                revised_sentiment_score=0.0,
                is_ambiguity_resolved=False
            )
    
    def _parse_llm_sentiment_response(self, response: str) -> dict:
        """
        Parse LLM response for normal sentiment analysis.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed dictionary with sentiment data
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Convert event strings to MarketEvent enums
            events = []
            for event_str in data.get("events", []):
                try:
                    events.append(MarketEvent[event_str])
                except KeyError:
                    logger.warning(f"Unknown event type: {event_str}")
            
            if not events:
                events = [MarketEvent.NONE]
            
            return {
                "sentiment_score": float(data.get("sentiment_score", 0.0)),
                "events": events,
                "keywords": data.get("keywords", []),
                "summary": data.get("summary", "No summary available")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM sentiment response: {e}")
            return {
                "sentiment_score": 0.0,
                "events": [MarketEvent.NONE],
                "keywords": [],
                "summary": "Parse error"
            }
    
    def _parse_llm_investigation_response(self, response: str) -> dict:
        """
        Parse LLM response for investigation analysis.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed dictionary with investigation data
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return {
                "detailed_findings": data.get("detailed_findings", "No findings available"),
                "revised_sentiment_score": float(data.get("revised_sentiment_score", 0.0)),
                "is_ambiguity_resolved": bool(data.get("is_ambiguity_resolved", False))
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM investigation response: {e}")
            return {
                "detailed_findings": "Parse error",
                "revised_sentiment_score": 0.0,
                "is_ambiguity_resolved": False
            }
    
    def _create_default_report(self, ticker: str) -> SAReport:
        """
        Create a default neutral SAReport when no news is available.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Default SAReport with neutral values
        """
        return SAReport(
            ticker=ticker,
            impactful_events=[MarketEvent.NONE],
            top_keywords=["no-news", "data-unavailable"],
            news_summary="No recent news articles found for sentiment analysis. Using neutral default.",
            qualitative_sentiment_assessment="Neutral - No recent news articles available for analysis",
            causal_narrative="No causal narrative available due to lack of news data",
            expectation_gap="No expectation analysis available",
            paradoxes_or_tensions="None identified"
        )

