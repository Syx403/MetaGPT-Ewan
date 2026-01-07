"""
Filename: MetaGPT-Ewan/MAT/actions/retrieve_rag_data.py
Created Date: Sunday, December 29th 2025
Author: Ewan Su
Description: RetrieveRAGData action using RAGFlow API for fundamental analysis.

This action retrieves fundamental data for a specific ticker by querying
the RAGFlow API with targeted questions about Revenue Growth, Gross Margin,
Guidance, and Key Risks. The retrieved chunks are then distilled by an LLM
(Claude) into a structured FAReport.

Configuration:
    API endpoint, dataset_id, and api_key should be configured in config/config2.yaml.
    Placeholders are provided for easy customization.
"""

import asyncio
import json
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

from metagpt.actions import Action
from metagpt.logs import logger
from pydantic import Field

from ..schemas import FAReport, FinancialMetric
from ..config_loader import get_config


def html_table_to_markdown(html_content: str) -> str:
    """
    Convert HTML tables to Markdown format using BeautifulSoup (Production-Grade).

    This utility uses BeautifulSoup to properly parse HTML and convert tables to
    Markdown pipe syntax with correct separator rows. Critical for financial table
    rendering in evidence_md fields.

    Args:
        html_content: Raw HTML content potentially containing <table> tags

    Returns:
        Markdown-formatted content with tables converted to pipe syntax

    Example:
        Input:  "<table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2021</td><td>$365B</td></tr></table>"
        Output: "| Year | Revenue |\n|:---|:---|\n| 2021 | $365B |"
    """
    if not html_content:
        return html_content

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback to regex if BeautifulSoup not available
        logger.warning("BeautifulSoup not installed, using regex fallback")
        return _html_table_to_markdown_regex(html_content)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all tables
    tables = soup.find_all('table')

    if not tables:
        # No tables found, just clean remaining HTML tags
        return soup.get_text(separator=' ', strip=True)

    # Process each table
    for table in tables:
        md_lines = []

        # Extract header row (first <tr> with <th> tags)
        header_row = table.find('tr')
        headers = []

        if header_row:
            # Check if it has <th> tags
            th_cells = header_row.find_all('th')
            if th_cells:
                headers = [cell.get_text(strip=True) for cell in th_cells]
                num_cols = len(headers)

                # Create header row
                md_lines.append('| ' + ' | '.join(headers) + ' |')

                # CRITICAL: Create separator row with exact number of columns
                # Use |:---|:---| format for left-aligned columns
                md_lines.append('|' + '|'.join([':---'] * num_cols) + '|')

                # Find all data rows (skip header row)
                data_rows = table.find_all('tr')[1:]
            else:
                # First row has <td> tags, treat all rows as data
                data_rows = table.find_all('tr')
                num_cols = len(header_row.find_all('td'))
        else:
            data_rows = table.find_all('tr')
            num_cols = 0

        # Process data rows
        for row in data_rows:
            cells = row.find_all(['td', 'th'])
            if cells:
                cell_texts = [cell.get_text(strip=True) for cell in cells]

                # Update num_cols if not set
                if num_cols == 0:
                    num_cols = len(cell_texts)

                # Pad or truncate to match column count
                while len(cell_texts) < num_cols:
                    cell_texts.append('')
                cell_texts = cell_texts[:num_cols]

                md_lines.append('| ' + ' | '.join(cell_texts) + ' |')

        # Replace table with Markdown
        table_md = '\n'.join(md_lines)
        table.replace_with(BeautifulSoup(table_md, 'html.parser'))

    # Clean up remaining HTML tags and return text
    # Remove <sup>, <span>, <div>, etc.
    for tag in soup.find_all(['sup', 'sub', 'span', 'div', 'p', 'br']):
        tag.replace_with(tag.get_text() if tag.string else '')

    # Get final text
    result = soup.get_text(separator='\n', strip=True)

    # Clean up excessive whitespace
    result = re.sub(r'\n\s*\n', '\n\n', result)
    result = result.strip()

    return result


def _html_table_to_markdown_regex(html_content: str) -> str:
    """Regex-based fallback for HTML to Markdown conversion."""
    content = html_content

    # Replace table row tags
    content = re.sub(r'<tr[^>]*>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'</tr>', '', content, flags=re.IGNORECASE)

    # Replace header cells
    content = re.sub(r'<th[^>]*>', '| ', content, flags=re.IGNORECASE)
    content = re.sub(r'</th>', ' ', content, flags=re.IGNORECASE)

    # Replace data cells
    content = re.sub(r'<td[^>]*>', '| ', content, flags=re.IGNORECASE)
    content = re.sub(r'</td>', ' ', content, flags=re.IGNORECASE)

    # Remove table tags
    content = re.sub(r'</?table[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</?thead[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</?tbody[^>]*>', '', content, flags=re.IGNORECASE)

    # Clean up other tags
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\n\s*\n', '\n\n', content)

    return content.strip()


class RetrieveRAGData(Action):
    """
    Retrieve fundamental data from RAGFlow API for fundamental analysis.
    
    This action performs targeted queries to RAGFlow to retrieve financial
    information about a specific ticker, then uses an LLM to synthesize
    the information into a structured FAReport.
    
    Key Features:
    1. Performs 4 targeted queries: Revenue Growth, Gross Margin, Guidance, Key Risks
    2. Sends POST requests to RAGFlow API endpoint
    3. Combines retrieved chunks from all queries
    4. Uses LLM (Claude) to distill information into FAReport
    5. Handles API errors gracefully with fallback mechanisms
    
    Configuration:
        RAGFlow settings should be configured in config/config2.yaml:
        - ragflow_endpoint: The RAGFlow API endpoint URL
        - ragflow_api_key: Your RAGFlow API key
        - ragflow_dataset_id: The dataset ID for financial data
    
    Example:
        ragflow:
          endpoint: "http://your-ragflow-ip:9380/api/v1/retrieval"
          api_key: "your-ragflow-api-key-here"
          dataset_id: "your-dataset-id-here"
    """
    
    name: str = "RetrieveRAGData"

    # RAGFlow API configuration (loaded from config/config2.yaml)
    ragflow_endpoint: str = Field(default="http://<your-ragflow-ip>:9380/api/v1/retrieval")
    ragflow_api_key: str = Field(default="<your-api-key>")
    ragflow_dataset_id: str = Field(default="<your-dataset-id>")
    rerank_id: Optional[str] = Field(default=None, description="Advanced reranking model ID")

    # Query parameters
    top_k: int = Field(default=5, description="Number of chunks to retrieve per query")

    # Output directory for saving RAG results (for debugging/audit)
    output_dir: Path = Field(default=Path("MAT/report/RA"))

    # UI base URL for constructing clickable source links
    ui_base_url: str = Field(default="")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load configuration from config/config2.yaml
        config = get_config()

        # Get RAGFlow settings from config
        ragflow_config = config.get_ragflow_config()
        self.ragflow_endpoint = ragflow_config.get("endpoint", self.ragflow_endpoint)
        self.ragflow_api_key = ragflow_config.get("api_key", self.ragflow_api_key)
        self.ragflow_dataset_id = ragflow_config.get("dataset_id", self.ragflow_dataset_id)
        self.top_k = ragflow_config.get("top_k", self.top_k)
        self.rerank_id = ragflow_config.get("rerank_id", self.rerank_id)

        # Note: output_dir uses Field default (MAT/report/RA) unless overridden in config

        # Extract base IP from endpoint and construct UI base URL
        # Example: "http://13.229.88.43:9380/api/v1/retrieval" -> "http://13.229.88.43"
        self.ui_base_url = self._extract_ui_base_url(self.ragflow_endpoint)

        # Log initialization status
        if self.ragflow_api_key != "<your-api-key>":
            rerank_status = f" [Reranking: {self.rerank_id}]" if self.rerank_id else ""
            logger.info(f"📚 RetrieveRAGData initialized (endpoint={self.ragflow_endpoint}){rerank_status}")
            logger.info(f"🔗 UI Base URL: {self.ui_base_url}")
        else:
            logger.warning("⚠️ RetrieveRAGData: RAGFlow API not configured in config/config2.yaml")
            logger.warning("   Please add 'ragflow' section with endpoint, api_key, and dataset_id")

    def _extract_ui_base_url(self, endpoint: str) -> str:
        """
        Extract base IP/domain from RAGFlow API endpoint for UI link construction.

        Args:
            endpoint: Full RAGFlow API endpoint (e.g., "http://13.229.88.43:9380/api/v1/retrieval")

        Returns:
            Base URL with protocol and host (e.g., "http://13.229.88.43")
        """
        import re
        # Extract protocol + host (with optional port)
        match = re.match(r'(https?://[^/]+)', endpoint)
        if match:
            # Remove port if present, return just protocol + host
            base_with_port = match.group(1)
            # Extract protocol + IP without port
            match2 = re.match(r'(https?://[^:]+)', base_with_port)
            if match2:
                return match2.group(1)
            return base_with_port
        return ""
    
    async def run(
        self,
        ticker: str,
        fiscal_year: Optional[int] = None,
        llm_callback: Optional[Any] = None
    ) -> FAReport:
        """
        Retrieve fundamental data from RAGFlow and generate FAReport.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
            fiscal_year: Optional fiscal year to focus on (e.g., 2021, 2022)
            llm_callback: Optional callback for LLM analysis (default: use self._aask)
            
        Returns:
            FAReport with fundamental analysis data
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📚 RETRIEVE RAG DATA ACTION: {ticker}")
        if fiscal_year:
            logger.info(f"📅 Target Fiscal Year: {fiscal_year}")
        logger.info(f"{'='*80}")
        
        try:
            # Step 1: Check if RAGFlow is configured
            if not self._is_ragflow_configured():
                logger.warning("⚠️ RAGFlow not configured, returning default FAReport")
                return self._create_default_report(ticker)
            
            # Step 2: Perform 5 targeted queries to RAGFlow
            # (This also saves top-k chunks via _save_topk_chunks)
            all_chunks = await self._retrieve_all_queries(ticker, fiscal_year)

            if not all_chunks:
                logger.warning("⚠️ No data retrieved from RAGFlow")
                return self._create_default_report(ticker)

            # Step 3: Use LLM to distill chunks into structured FAReport
            fa_report = await self._synthesize_fa_report(
                ticker=ticker,
                chunks=all_chunks,
                llm_callback=llm_callback,
                fiscal_year=fiscal_year
            )

            # Step 4: Save final structured FAReport
            if fiscal_year:
                self._save_report(fa_report, fiscal_year)
            else:
                logger.warning("⚠️ fiscal_year not provided - skipping FAReport save")

            logger.info(f"\n{'='*80}")
            logger.info(f"✅ FINANCIAL AUDIT COMPLETE for {ticker}")
            logger.info(f"📊 Revenue Performance: {fa_report.revenue_performance.value or 'DATA_GAP'}")
            logger.info(f"📊 Profitability: {fa_report.profitability_audit.value or 'DATA_GAP'}")
            logger.info(f"📊 Cash Flow: {fa_report.cash_flow_stability.value or 'DATA_GAP'}")
            logger.info(f"📊 Key Risks: {len(fa_report.key_risks_evidence)} identified")
            logger.info(f"{'='*80}\n")

            return fa_report
            
        except Exception as e:
            logger.error(f"❌ RetrieveRAGData failed: {e}")
            return self._create_default_report(ticker, error_message=str(e))
    
    def _is_ragflow_configured(self) -> bool:
        """
        Check if RAGFlow is properly configured.
        
        Returns:
            True if all required RAGFlow settings are configured
        """
        is_configured = (
            self.ragflow_api_key != "<your-api-key>" and
            self.ragflow_dataset_id != "<your-dataset-id>" and
            "<your-ragflow-ip>" not in self.ragflow_endpoint
        )
        return is_configured
    
    async def _retrieve_all_queries(self, ticker: str, fiscal_year: Optional[int] = None) -> List[Dict]:
        """
        Perform expert-level multi-query strategy to maximize RAG recall.

        Updated Query Strategy (Expert-Level):
        1. Segment Revenue Drivers - Detailed breakdown of revenue sources by product/geography
        2. Margin and Cost Analysis - Profitability trends with operating leverage analysis
        3. Cash Flow Generation - FCF trends, capital allocation, and sustainability
        4. Management Guidance & Strategic Outlook - Forward guidance with strategic commentary
        5. Risk Factors and Uncertainties - Company-specific risks with impact assessment

        Args:
            ticker: Stock ticker symbol
            fiscal_year: Optional fiscal year to focus on (e.g., 2021, 2022)

        Returns:
            List of all retrieved chunks from all queries with category metadata
        """
        logger.info("🔍 Performing expert-level multi-query strategy to RAGFlow...")

        # Ticker to full company name mapping for anchor querying
        COMPANY_NAMES = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com, Inc.",
            "TSLA": "Tesla, Inc.",
            "NVDA": "NVIDIA Corporation",
            "META": "Meta Platforms, Inc.",
            "KO": "The Coca-Cola Company",
        }

        # Get full company name for anchor querying (fallback to ticker if not in mapping)
        company_name = COMPANY_NAMES.get(ticker, ticker)

        # Add year context to queries if specified
        year_context = f" in fiscal year {fiscal_year} or FY{fiscal_year}" if fiscal_year else ""

        # Define 5 high-precision queries with SEC ITEM code anchoring
        # ITEM 8: Financial Statements (Revenue, Margin, Cash Flow tables)
        # ITEM 7: MD&A (Management's Discussion and Analysis)
        # ITEM 1A: Risk Factors
        queries = [
            {
                "category": "Segment Revenue Drivers",
                "question": f"ITEM 8 {company_name} {fiscal_year if fiscal_year else ''} Consolidated Statement of Operations Net Sales Net Operating Revenues by product segment geography business line."
            },
            {
                "category": "Margin and Cost Analysis",
                "question": f"ITEM 8 {company_name} {fiscal_year if fiscal_year else ''} Consolidated Statement of Operations gross profit gross margin operating income net income cost of sales."
            },
            {
                "category": "Cash Flow Generation",
                "question": f"ITEM 8 {company_name} {fiscal_year if fiscal_year else ''} Consolidated Statement of Cash Flows net cash provided by operating activities capital expenditures free cash flow."
            },
            {
                "category": "Management Guidance",
                "question": f"ITEM 7 {company_name} {fiscal_year if fiscal_year else ''} Management's Discussion and Analysis MD&A forward guidance strategic outlook future projections."
            },
            {
                "category": "Risk Factors",
                "question": f"ITEM 1A {company_name} {fiscal_year if fiscal_year else ''} Risk Factors operational regulatory competitive macroeconomic risks uncertainties."
            }
        ]

        # Execute queries sequentially (parallel execution causes timeouts with RAGFlow)
        all_chunks = []
        chunks_by_category = {}  # Track chunks by category for dual-output saving

        for query_info in queries:
            category = query_info["category"]
            question = query_info["question"]

            logger.info(f"📋 Query [{category}]: {question[:80]}...")

            # Send POST request to RAGFlow
            chunks = await self._query_ragflow(question, ticker=ticker)

            if chunks:
                logger.info(f"   ✅ Retrieved {len(chunks)} chunks")
                # [DEBUG] Track chunk accumulation per category
                logger.info(f"   🔍 [DEBUG] Category '{category}': Adding {len(chunks)} chunks to all_chunks (current total: {len(all_chunks)})")

                # Add category metadata to chunks
                for chunk in chunks:
                    chunk["query_category"] = category

                # Track chunks by category for UI-ready output
                chunks_by_category[category] = chunks

                # Also maintain flat list for synthesis
                all_chunks.extend(chunks)
                logger.info(f"   🔍 [DEBUG] New total after adding '{category}': {len(all_chunks)} chunks")
            else:
                logger.warning(f"   ⚠️ No chunks retrieved for {category}")
                # Add empty list for this category
                chunks_by_category[category] = []

        logger.info(f"✅ Total chunks retrieved: {len(all_chunks)}")

        # Save top-k chunks for UI display
        if fiscal_year:
            self._save_topk_chunks(ticker, fiscal_year, chunks_by_category)
        else:
            logger.warning("⚠️ fiscal_year not provided - skipping top-k chunks save")

        # NOTE: Physical isolation filtering removed - causes 100% data loss
        # RAGFlow chunks do not consistently return document_name in expected fields
        # Intelligent content-based filtering will be implemented in Task B

        # Extract top_1 chunk for each category for evidence capture
        top_chunks_by_category = {}
        for category in ["Segment Revenue Drivers", "Margin and Cost Analysis", "Cash Flow Generation"]:
            category_chunks = [c for c in all_chunks if c.get("query_category") == category]
            if category_chunks:
                # Take the first chunk (highest relevance after reranking)
                top_chunks_by_category[category] = category_chunks[0].get("content", "No content available")

        # Store top chunks for later use in synthesis
        self._top_evidence_chunks = top_chunks_by_category

        return all_chunks
    
    async def _query_ragflow(self, question: str, ticker: Optional[str] = None) -> List[Dict]:
        """
        Send a POST request to RAGFlow API to retrieve relevant chunks.

        Standard API implementation - no metadata filtering or hallucinated parameters.

        Args:
            question: The query question to send to RAGFlow
            ticker: Optional ticker symbol (not used in API - only for logging)

        Returns:
            List of retrieved chunks with content and metadata
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("❌ aiohttp not installed. Run: pip install aiohttp")
            return []

        # Prepare hybrid search API payload for optimal reranker engagement
        # Hybrid search (Vector + Keyword) is essential for reranker effectiveness
        payload = {
            "question": question,
            "dataset_ids": [self.ragflow_dataset_id],
            "top_k": self.top_k,
            "keyword": True,  # Enable keyword search for hybrid retrieval
            "vector_similarity_weight": 0.3  # Give keywords more weight (0.3 vector, 0.7 keyword)
        }

        # Add reranking if configured (improves table data recall)
        if self.rerank_id:
            payload["rerank_id"] = self.rerank_id

        # Prepare headers with API key
        headers = {
            "Authorization": f"Bearer {self.ragflow_api_key}",
            "Content-Type": "application/json"
        }

        # CRITICAL: Log full payload for manual verification of rerank_id
        logger.info(f"📤 RAGFlow API Request Payload:\n{json.dumps(payload, indent=2)}")
        # [DEBUG] Log requested top_k and rerank status
        logger.info(f"🔍 [DEBUG] Payload Top_K: {self.top_k} | Rerank ID: {self.rerank_id}")

        try:
            # Send POST request to RAGFlow
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ragflow_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)  # Increased for reranking
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Extract chunks from response
                        # RAGFlow returns: {"code": 0, "data": {"chunks": [...]}}
                        if data.get("code") == 0:
                            chunks = data.get("data", {}).get("chunks", [])
                            # [DEBUG] Log raw response count
                            logger.info(f"🔍 [DEBUG] API Response Count: {len(chunks)} chunks received for query: '{question[:40]}...'")
                        else:
                            logger.warning(f"RAGFlow returned error: {data.get('message', 'Unknown error')}")
                            chunks = []
                        
                        # Normalize chunk format and preserve ALL metadata
                        normalized_chunks = []
                        for chunk in chunks:
                            # Extract raw content
                            raw_content = chunk.get("content_with_weight", chunk.get("content", ""))

                            # Convert HTML tables to Markdown
                            content_md = html_table_to_markdown(raw_content)

                            # Extract metadata for source URL construction
                            chunk_id = chunk.get("id", "")
                            document_id = chunk.get("document_id", "")
                            dataset_id = chunk.get("dataset_id", self.ragflow_dataset_id)
                            positions = chunk.get("positions", [])

                            # Extract page number from positions if available
                            page_num = None
                            if positions and len(positions) > 0:
                                # positions format: [page_num, x0, y0, x1, y1]
                                page_num = positions[0] if isinstance(positions[0], (int, float)) else None

                            # Construct clickable RAGFlow source URL
                            # Format: http://{base_ip}/chunk/parsed/chunks?id={dataset_id}&doc_id={doc_id}&page=1
                            source_url = "No source URL available"
                            if self.ui_base_url and document_id and dataset_id:
                                page_param = page_num if page_num is not None else 1
                                source_url = f"{self.ui_base_url}/chunk/parsed/chunks?id={dataset_id}&doc_id={document_id}&page={page_param}"

                            normalized_chunk = {
                                "content": content_md,  # Now Markdown-cleaned
                                "score": chunk.get("similarity", chunk.get("score", 0.0)),
                                "chunk_id": chunk_id,
                                "document_name": chunk.get("document_keyword", "Unknown Document"),
                                "document_id": document_id,
                                "dataset_id": dataset_id,
                                "positions": positions,
                                "page_num": page_num,
                                "source_url": source_url,  # NEW: Clickable URL
                                "metadata": chunk.get("metadata", {})
                            }
                            normalized_chunks.append(normalized_chunk)

                        # [DEBUG] Log score distribution to identify drop-off
                        if len(normalized_chunks) > 0:
                            scores = [c["score"] for c in normalized_chunks]
                            logger.info(f"🔍 [DEBUG] Score Distribution (first 10):")
                            for idx, score in enumerate(scores[:10], 1):
                                marker = " <- TOP_K CUTOFF" if idx == self.top_k else ""
                                logger.info(f"   Chunk {idx}: {score:.4f}{marker}")
                            if len(scores) > 10:
                                logger.info(f"   ... and {len(scores) - 10} more chunks")

                        # CRITICAL: Client-side slicing to enforce top_k limit
                        # RAGFlow server ignores top_k parameter and returns 30 chunks
                        logger.info(f"🔍 [DEBUG] Pre-slicing count: {len(normalized_chunks)} chunks")
                        normalized_chunks = normalized_chunks[:self.top_k]
                        logger.info(f"🔍 [DEBUG] Post-slicing count: {len(normalized_chunks)} chunks (enforced top_k={self.top_k})")

                        logger.info(f"   ✅ Retrieved {len(normalized_chunks)} chunks")
                        return normalized_chunks
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ RAGFlow API error {response.status}: {error_text}")
                        return []
        
        except Exception as e:
            logger.error(f"❌ RAGFlow query failed: {e}")
            return []
    
    def _save_topk_chunks(self, ticker: str, fiscal_year: int, chunks_by_category: Dict[str, List[Dict]]):
        """
        Save top-k chunks retrieved for each query category to a single JSON file for UI display.

        This method outputs a UI-ready chunk file where each category contains up to 5 chunks
        (enforced by client-side slicing in _query_ragflow).

        Args:
            ticker: Stock ticker symbol
            fiscal_year: Fiscal year for filename
            chunks_by_category: Dictionary mapping category names to lists of normalized chunks

        Output Path:
            MAT/report/RA/Topk_chunk_{ticker}_{fiscal_year}.json
        """
        from datetime import datetime

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        filename = f"Topk_chunk_{ticker}_{fiscal_year}.json"
        output_path = self.output_dir / filename

        # Build output data structure
        output_data = {
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "generated_at": datetime.now().isoformat(),
            "top_k_per_query": self.top_k,
            "categories": chunks_by_category
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📁 Top-K chunks saved: {output_path}")

            # Log chunk counts per category
            for category, chunks in chunks_by_category.items():
                logger.info(f"   - {category}: {len(chunks)} chunks")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save top-k chunks: {e}")

    def _save_report(self, report: FAReport, fiscal_year: int):
        """
        Save final structured FAReport to JSON file.

        This method outputs the synthesized FAReport object with all financial metrics,
        analyses, and source traceability.

        Args:
            report: FAReport Pydantic object to save
            fiscal_year: Fiscal year for filename

        Output Path:
            MAT/report/RA/RA_report_{ticker}_{fiscal_year}.json
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        filename = f"RA_report_{report.ticker}_{fiscal_year}.json"
        output_path = self.output_dir / filename

        try:
            # Use Pydantic's model_dump_json for proper serialization
            report_json = report.model_dump_json(indent=2, exclude_none=False)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_json)

            logger.info(f"📁 FAReport saved: {output_path}")
            logger.info(f"   - Revenue Performance: {report.revenue_performance.value}")
            logger.info(f"   - Profitability Audit: {report.profitability_audit.value}")
            logger.info(f"   - Cash Flow Stability: {report.cash_flow_stability.value}")
            logger.info(f"   - Key Risks: {len(report.key_risks_evidence)} identified")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save FAReport: {e}")
    
    async def _synthesize_fa_report(
        self,
        ticker: str,
        chunks: List[Dict],
        llm_callback: Optional[Any] = None,
        fiscal_year: Optional[int] = None
    ) -> FAReport:
        """
        Use LLM (Claude) to synthesize retrieved chunks into structured FAReport.
        
        The LLM is given all retrieved chunks and instructed to extract:
        - revenue_growth_yoy: Year-over-year revenue growth rate
        - gross_margin: Gross profit margin
        - fcf_growth: Free cash flow growth
        - guidance_sentiment: Management guidance sentiment (-1 to 1)
        - key_risks: Top 3 risk factors
        - is_growth_healthy: Whether fundamentals meet threshold
        
        Args:
            ticker: Stock ticker symbol
            chunks: Retrieved chunks from RAGFlow
            llm_callback: Optional callback for LLM (uses self._aask if None)
            fiscal_year: Optional fiscal year to focus on (e.g., 2021, 2022)
            
        Returns:
            FAReport with synthesized fundamental data
        """
        # Prepare chunks content for LLM
        chunks_content = self._prepare_chunks_for_llm(chunks)
        
        # Build year-specific instruction
        if fiscal_year:
            year_instruction = f"Fiscal Year {fiscal_year}"
        else:
            year_instruction = "most recent fiscal year available"

        # Build expert-level synthesis prompt with source traceability
        prompt = f"""You are an expert Financial Auditor conducting fundamental analysis for {ticker} ({year_instruction}).

=== RAG CHUNKS (INCLUDES METADATA) ===
{chunks_content}

=== CRITICAL AUDIT PROTOCOLS ===

**1. ENTITY INTEGRITY (THE ANTI-CONTAMINATION RULE):**
Every chunk has metadata (Document Name, ID). Verify it belongs to {ticker}:
- If auditing {ticker}, and the chunk discusses a different entity (e.g. Coca-Cola vs Apple), IGNORE it.
- DATA CROSS-CONTAMINATION IS A TERMINATION OFFENSE.

**2. TABLE ANCHORING (ELIMINATING DATA_GAP):**
Search specifically for these anchors in the chunks to find missing metrics:
- Revenue: "Consolidated Statements of Operations", "Net Sales", "Net Operating Revenues".
- Cash Flow: "Consolidated Statements of Cash Flows", "Net cash provided by operating activities".
- Profitability: "Gross Margin", "Operating Income".
Reconstruct the full table logic if values are split across chunks.

**3. SOURCE TRACEABILITY:**
- You MUST provide the exact **Document Name** (e.g. APPLE_2022_10K.pdf) and **Chunk ID** for every metric.
- This information is provided at the start of each chunk.

=== EXTRACTION TASK ===

**1. Revenue Performance**:
   - value: Raw metric (e.g., "$394.3B, +8% YoY")
   - analysis: BECAUSE-THEN causal logic explaining segment drivers.
   - evidence_md: Copy the FULL Markdown-cleaned TEXT of the Top 1 source chunk (tables converted to Markdown format).
   - source_link: Construct a reference: [Document Name] | [Chunk ID]
   - source_url: Copy the source URL from the chunk metadata (for clickable PDF preview).
   - metadata: Include page_num and relevance_score from chunk metadata.

**2. Profitability Audit**:
   - value: Gross/Operating margins.
   - analysis: BECAUSE-THEN logic (e.g., "Margin dropped BECAUSE of FX headwinds").
   - evidence_md: Copy the FULL Markdown-cleaned TEXT of the source chunk.
   - source_link: [Document Name] | [Chunk ID]
   - source_url: Copy the source URL from chunk metadata.
   - metadata: Include page_num and relevance_score.

**3. Cash Flow Stability**:
   - value: "Net cash provided by operating activities" (Operating Cash Flow).
   - analysis: Assessment of cash generation sustainability using BECAUSE-THEN logic.
   - evidence_md: Copy the FULL Markdown-cleaned TEXT of the source chunk.
   - source_link: [Document Name] | [Chunk ID]
   - source_url: Copy the source URL from chunk metadata.
   - metadata: Include page_num and relevance_score.

**4. Management Guidance & Risks**:
   - Audit tone and specific projections.
   - List 3-5 structural risks with specific impacts and source citations.

=== OUTPUT FORMAT (STRICT JSON) ===
{{
    "ticker": "{ticker}",
    "revenue_performance": {{
        "value": "string",
        "analysis": "string",
        "evidence_md": "string (Markdown-cleaned chunk text)",
        "source_link": "DocumentName | ID",
        "source_url": "string (clickable RAGFlow URL)",
        "metadata": {{"page_num": int, "relevance_score": float}}
    }},
    "profitability_audit": {{
        "value": "string",
        "analysis": "string",
        "evidence_md": "string (Markdown-cleaned chunk text)",
        "source_link": "DocumentName | ID",
        "source_url": "string (clickable RAGFlow URL)",
        "metadata": {{"page_num": int, "relevance_score": float}}
    }},
    "cash_flow_stability": {{
        "value": "string",
        "analysis": "string",
        "evidence_md": "string (Markdown-cleaned chunk text)",
        "source_link": "DocumentName | ID",
        "source_url": "string (clickable RAGFlow URL)",
        "metadata": {{"page_num": int, "relevance_score": float}}
    }},
    "management_guidance_audit": "string",
    "key_risks_evidence": ["string (Risk | Impact | Source)"]
}}

CRITICAL: Provide ONLY the JSON. No markdown code blocks, no explanations.
NOTE: source_citations field removed - all source integrity is tracked within each FinancialMetric via evidence_md, source_link, source_url, and metadata.
"""
        
        try:
            # Call LLM
            if llm_callback:
                response = await llm_callback(prompt)
            else:
                response = await self._aask(prompt)
            
            # Parse LLM response
            parsed = self._parse_llm_response(response, ticker)
            
            # Create FAReport from parsed data
            fa_report = FAReport(**parsed)
            
            logger.info("✅ FAReport synthesized successfully")
            return fa_report
            
        except Exception as e:
            logger.error(f"❌ LLM synthesis failed: {e}")
            return self._create_default_report(ticker, error_message=f"LLM synthesis error: {str(e)}")
    
    def _prepare_chunks_for_llm(self, chunks: List[Dict]) -> str:
        """
        Prepare retrieved chunks for LLM analysis.
        
        Args:
            chunks: Retrieved chunks from RAGFlow
            
        Returns:
            Formatted string for LLM prompt
        """
        content_parts = []
        
        # Group chunks by category
        categories = {}
        for chunk in chunks:
            category = chunk.get("query_category", "Unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(chunk)
        
        # Format chunks by category WITH METADATA
        for category, cat_chunks in categories.items():
            content_parts.append(f"\n=== {category.upper()} ===")

            for i, chunk in enumerate(cat_chunks[:5], 1):  # Limit to top 5 chunks per category
                content = chunk.get("content", "")
                score = chunk.get("score", 0.0)
                doc_name = chunk.get("document_name", "Unknown")
                chunk_id = chunk.get("chunk_id", "Unknown")
                source_url = chunk.get("source_url", "No source URL available")
                page_num = chunk.get("page_num", "Unknown")

                part = f"""
--- Chunk {i} (Relevance: {score:.2f}) ---
[Document: {doc_name}] [Chunk ID: {chunk_id}]
[Page: {page_num}] [Source URL: {source_url}]
{content[:1500]}
"""
                content_parts.append(part)
        
        return "\n".join(content_parts)
    
    def _parse_llm_response(self, response: str, ticker: str) -> Dict:
        """
        Parse LLM JSON response with robust error handling.

        Args:
            response: LLM response string
            ticker: Stock ticker symbol

        Returns:
            Parsed dictionary with FAReport data (NEW schema)
        """
        try:
            # Parse JSON robustly (handles Markdown wrapping)
            from MAT.roles.base_agent import BaseInvestmentAgent
            data = BaseInvestmentAgent.parse_json_robustly(response)

            # Parse NEW schema fields matching FAReport structure
            # Support both old field names (evidence_chunk) and new (evidence_md) for backward compatibility
            parsed = {
                "ticker": ticker,
                "revenue_performance": {
                    "value": data.get("revenue_performance", {}).get("value"),
                    "analysis": data.get("revenue_performance", {}).get("analysis", "No analysis available"),
                    "evidence_md": data.get("revenue_performance", {}).get("evidence_md") or data.get("revenue_performance", {}).get("evidence_chunk", "No evidence available"),
                    "source_link": data.get("revenue_performance", {}).get("source_link", "No source link available"),
                    "source_url": data.get("revenue_performance", {}).get("source_url", "No source URL available"),
                    "metadata": data.get("revenue_performance", {}).get("metadata", {})
                },
                "profitability_audit": {
                    "value": data.get("profitability_audit", {}).get("value"),
                    "analysis": data.get("profitability_audit", {}).get("analysis", "No analysis available"),
                    "evidence_md": data.get("profitability_audit", {}).get("evidence_md") or data.get("profitability_audit", {}).get("evidence_chunk", "No evidence available"),
                    "source_link": data.get("profitability_audit", {}).get("source_link", "No source link available"),
                    "source_url": data.get("profitability_audit", {}).get("source_url", "No source URL available"),
                    "metadata": data.get("profitability_audit", {}).get("metadata", {})
                },
                "cash_flow_stability": {
                    "value": data.get("cash_flow_stability", {}).get("value"),
                    "analysis": data.get("cash_flow_stability", {}).get("analysis", "No analysis available"),
                    "evidence_md": data.get("cash_flow_stability", {}).get("evidence_md") or data.get("cash_flow_stability", {}).get("evidence_chunk", "No evidence available"),
                    "source_link": data.get("cash_flow_stability", {}).get("source_link", "No source link available"),
                    "source_url": data.get("cash_flow_stability", {}).get("source_url", "No source URL available"),
                    "metadata": data.get("cash_flow_stability", {}).get("metadata", {})
                },
                "management_guidance_audit": data.get("management_guidance_audit", "No guidance analysis available"),
                "key_risks_evidence": data.get("key_risks_evidence", [])
                # source_citations removed - integrity tracked in each FinancialMetric
            }

            logger.debug(f"✅ Successfully parsed FAReport with {len(parsed['key_risks_evidence'])} risks")

            return parsed

        except Exception as e:
            logger.error(f"❌ Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response[:500]}...")

            # Return default structure matching FAReport schema (NEW schema with evidence_md, source_url, metadata)
            return {
                "ticker": ticker,
                "revenue_performance": {
                    "value": None,
                    "analysis": "No analysis available",
                    "evidence_md": "No evidence available",
                    "source_link": "No source link available",
                    "source_url": "No source URL available",
                    "metadata": {}
                },
                "profitability_audit": {
                    "value": None,
                    "analysis": "No analysis available",
                    "evidence_md": "No evidence available",
                    "source_link": "No source link available",
                    "source_url": "No source URL available",
                    "metadata": {}
                },
                "cash_flow_stability": {
                    "value": None,
                    "analysis": "No analysis available",
                    "evidence_md": "No evidence available",
                    "source_link": "No source link available",
                    "source_url": "No source URL available",
                    "metadata": {}
                },
                "management_guidance_audit": "No guidance analysis available",
                "key_risks_evidence": []
                # source_citations removed - integrity tracked in each FinancialMetric
            }
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))
    
    def _create_default_report(self, ticker: str, error_message: str = "") -> FAReport:
        """
        Create a default FAReport when RAGFlow is not available or analysis fails.

        Args:
            ticker: Stock ticker symbol
            error_message: Optional error message for logging

        Returns:
            FAReport with DATA_GAP indicators and error context
        """
        if error_message:
            logger.warning(f"⚠️ Creating default FAReport: {error_message}")

        error_context = f"RAGFlow data retrieval failed: {error_message}" if error_message else "RAGFlow not configured or data unavailable"

        return FAReport(
            ticker=ticker,
            revenue_performance=FinancialMetric(
                value="DATA_GAP",
                analysis=f"Revenue data unavailable BECAUSE {error_context}. Unable to assess revenue drivers or growth trends without RAG data."
            ),
            profitability_audit=FinancialMetric(
                value="DATA_GAP",
                analysis=f"Profitability metrics unavailable BECAUSE {error_context}. Cannot analyze margin trends without financial statements."
            ),
            cash_flow_stability=FinancialMetric(
                value="DATA_GAP",
                analysis=f"Cash flow data unavailable BECAUSE {error_context}. Unable to assess FCF generation or capital allocation."
            ),
            management_guidance_audit="No guidance data available - RAGFlow retrieval failed or not configured",
            key_risks_evidence=[f"DATA UNAVAILABLE: {error_context}"]
            # source_citations removed - integrity tracked in each FinancialMetric
        )

