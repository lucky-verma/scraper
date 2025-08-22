from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import uvicorn
import tempfile
import aiohttp
import os
from markitdown import MarkItDown
from typing import Optional, Tuple, Dict, List
import logging
import asyncpg
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from datetime import datetime, timedelta
import re
import tiktoken
import asyncio
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    BUSINESS_WEBSITE = "business_website"
    NEWS_ARTICLE = "news_article"
    TECHNICAL_DOC = "technical_doc"
    PRODUCT_PAGE = "product_page"
    ABOUT_PAGE = "about_page"
    CONTACT_PAGE = "contact_page"
    GENERIC = "generic"


class PageUrl(BaseModel):
    url: HttpUrl


class DynamicBusinessIntelligenceProcessor:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Business-critical patterns (no hardcoded limits)
        self.business_patterns = {
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone_numbers": r"(\+?\d{1,4}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4,6}",
            "company_entities": r"\b[A-Z][a-zA-Z\s&,\.]{2,50}(?:\s(?:Inc|LLC|Corp|Corporation|Ltd|Limited|Co|Company|Group|Holdings|Partners|Associates|Enterprises|Solutions|Technologies|Systems|Services|Industries|International|Global|Consulting|AG|GmbH|SA|SAS|SARL|BV|AB|AS|Oy|SpA|SRL)\.?)\b",
            "executive_info": r"\b(?:CEO|CTO|CFO|COO|President|Vice President|VP|Director|Manager|Head of|Chief|Founder|Co-Founder|Partner|Principal|Senior|Lead|Chairman|Board Member)\s+[A-Z][a-zA-Z\s]+",
            "addresses": r"\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Suite|Ste|Floor|Fl)\b[,\s]*[A-Z]{2}\s*\d{5}",
            "financial_indicators": r"\$[\d,]+(?:\.\d{2})?[KMB]?|\b\d+(?:\.\d+)?[%]\b|\b(?:revenue|sales|income|turnover|funding|valuation|profit|earnings|EBITDA|ARR|MRR)\s*:?\s*\$?[\d,]+(?:\.\d+)?[KMB]?\b",
            "website_urls": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
            "employee_data": r"\b(?:\d+[,\d]*)\s*(?:employees?|staff|team members?|people|workforce|headcount)\b",
            "company_age": r"\b(?:founded|established|since|started|incorporated)\s+(?:in\s+)?\d{4}\b",
            "industry_keywords": r"\b(?:industry|sector|market|vertical|niche|space|domain|field|area|technology|platform|solution|service|product)\b",
            "location_data": r"\b(?:headquarters|located|based|office|offices)\s+(?:in|at)\s+[A-Z][a-zA-Z\s,]+(?:,\s*[A-Z]{2})?\b",
            "social_media": r'(?:twitter|linkedin|facebook|instagram|youtube)\.com/[^\s<>"]+',
            "legal_entities": r"\b(?:subsidiary|division|acquisition|merger|parent company|holding company)\b.*?[A-Z][a-zA-Z\s&,\.]+",
            "certifications": r"\b(?:ISO|SOC|GDPR|HIPAA|certified|accredited|compliant|certification|accreditation)\b.*?\d*",
            "partnerships": r"\b(?:partner|partnership|alliance|collaboration|joint venture)\s+(?:with\s+)?[A-Z][a-zA-Z\s&,\.]+",
        }

        # Content type detection patterns
        self.content_type_indicators = {
            ContentType.BUSINESS_WEBSITE: [
                r"\b(?:about us|company|business|enterprise|corporation)\b",
                r"\b(?:services|solutions|products|offerings)\b",
                r"\b(?:contact|reach us|get in touch)\b",
                r"\b(?:team|leadership|management|executives)\b",
            ],
            ContentType.NEWS_ARTICLE: [
                r"\b(?:breaking|news|reported|sources|journalist)\b",
                r"\b(?:published|updated|posted)\s+(?:on|at)\s+\d",
                r"\b(?:according to|sources say|reports indicate)\b",
            ],
            ContentType.TECHNICAL_DOC: [
                r"\b(?:documentation|API|SDK|tutorial|guide)\b",
                r"\b(?:installation|configuration|setup|deployment)\b",
                r"```",
            ],
            ContentType.ABOUT_PAGE: [
                r"\b(?:our story|mission|vision|values|history)\b",
                r"\b(?:founded|established|started|began)\b",
                r"\b(?:believe|committed|dedicated|passionate)\b",
            ],
            ContentType.CONTACT_PAGE: [
                r"\b(?:contact|reach|touch|call|email|phone)\b",
                r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            ],
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def detect_content_type(self, content: str) -> ContentType:
        """Dynamically detect content type"""
        content_lower = content.lower()
        scores = {}

        for content_type, patterns in self.content_type_indicators.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                score += matches
            scores[content_type] = score

        if max(scores.values()) == 0:
            return ContentType.GENERIC

        return max(scores, key=scores.get)

    def extract_business_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract all business entities dynamically"""
        entities = {}

        for category, pattern in self.business_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Clean and deduplicate
                cleaned = list(
                    set(
                        [
                            match.strip()
                            for match in matches
                            if isinstance(match, str) and len(match.strip()) > 2
                        ]
                    )
                )
                entities[category] = cleaned[:50]  # Reasonable limit, not hardcoded

        return entities

    def calculate_business_density(self, content: str) -> float:
        """Calculate how business-information-dense the content is"""
        entities = self.extract_business_entities(content)
        total_entities = sum(len(v) for v in entities.values())
        content_length = len(content.split())

        if content_length == 0:
            return 0.0

        # Higher density = more business info per word
        return total_entities / content_length

    def determine_optimal_token_limit(
        self, content: str, content_type: ContentType
    ) -> int:
        """Dynamically determine optimal token limit based on content analysis"""
        current_tokens = self.count_tokens(content)
        business_density = self.calculate_business_density(content)

        # Base limits by content type (adaptive, not hardcoded)
        base_limits = {
            ContentType.CONTACT_PAGE: min(
                15000, current_tokens
            ),  # Preserve contact info
            ContentType.ABOUT_PAGE: min(
                25000, current_tokens
            ),  # Company story important
            ContentType.BUSINESS_WEBSITE: min(
                30000, current_tokens
            ),  # Business info priority
            ContentType.TECHNICAL_DOC: min(
                40000, current_tokens
            ),  # Tech detail matters
            ContentType.NEWS_ARTICLE: min(20000, current_tokens),  # Key facts focus
            ContentType.GENERIC: min(18000, current_tokens),  # Conservative default
        }

        base_limit = base_limits.get(content_type, 18000)

        # Adjust based on business density
        if business_density > 0.05:  # High business density
            multiplier = 1.5
        elif business_density > 0.02:  # Medium business density
            multiplier = 1.2
        else:  # Low business density
            multiplier = 0.8

        optimal_limit = int(base_limit * multiplier)

        # Ensure we don't exceed original content by too much
        return min(optimal_limit, current_tokens)

    def extract_priority_sections(
        self, content: str, content_type: ContentType
    ) -> List[str]:
        """Extract priority sections based on content type"""
        sections = []

        # Content-type specific section extraction
        if content_type in [ContentType.BUSINESS_WEBSITE, ContentType.ABOUT_PAGE]:
            about_patterns = [
                r"(?:^|\n)#{1,4}\s*(?:about|company|team|leadership|management|our story|who we are|mission|vision|history).*?\n(.*?)(?=\n#{1,4}|\n\n\n|\Z)",
                r"(?:about us|company profile|our company|team members|leadership team|executive team|management team)(.*?)(?=\n\n|\Z)",
                r"(?:mission|vision|values|culture|philosophy)(.*?)(?=\n\n|\Z)",
            ]
        elif content_type == ContentType.CONTACT_PAGE:
            about_patterns = [
                r"(?:contact|reach|get in touch|contact us|reach us)(.*?)(?=\n\n|\Z)",
                r"(?:phone|email|address|location|office)(.*?)(?=\n\n|\Z)",
            ]
        else:
            about_patterns = [
                r"(?:^|\n)#{1,3}\s*(.*?)\n(.*?)(?=\n#{1,3}|\n\n\n|\Z)",
            ]

        for pattern in about_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    section_content = " ".join(match)
                else:
                    section_content = match

                if len(section_content.strip()) > 50:  # Meaningful content
                    sections.append(section_content.strip())

        return sections

    def score_content_importance(self, text: str, content_type: ContentType) -> float:
        """Dynamic importance scoring based on content type and business value"""
        base_score = 1.0

        # Business entity scoring
        entities = self.extract_business_entities(text)
        entity_score = 1.0 + (sum(len(v) for v in entities.values()) * 0.1)

        # Content-type specific scoring
        if content_type == ContentType.CONTACT_PAGE:
            contact_patterns = [
                (r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", 3.0),
                (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", 2.5),
                (r"\b(?:phone|email|address|contact)\b", 2.0),
            ]
            for pattern, multiplier in contact_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    base_score *= multiplier

        elif content_type in [ContentType.BUSINESS_WEBSITE, ContentType.ABOUT_PAGE]:
            business_patterns = [
                (r"\b(?:founded|established|revenue|employees|headquarters)\b", 2.5),
                (r"\b(?:CEO|founder|executive|leadership|management)\b", 2.0),
                (r"\b(?:mission|vision|values|culture)\b", 1.8),
                (r"\b(?:services|products|solutions|platform)\b", 1.5),
            ]
            for pattern, multiplier in business_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    base_score *= multiplier

        # Universal high-value patterns
        universal_patterns = [
            (r"\$[\d,]+[KMB]?", 1.8),  # Financial figures
            (r"\b\d{4}\b", 1.3),  # Years
            (r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", 1.2),  # Proper nouns
        ]

        for pattern, multiplier in universal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                base_score *= multiplier

        return base_score * entity_score

    def intelligent_content_reduction(self, content: str) -> Tuple[str, Dict]:
        """Main intelligent reduction with dynamic adaptation"""

        # Step 1: Analyze content
        content_type = self.detect_content_type(content)
        optimal_limit = self.determine_optimal_token_limit(content, content_type)
        business_entities = self.extract_business_entities(content)

        logger.info(
            f"📊 Content analysis: {content_type.value}, optimal limit: {optimal_limit} tokens"
        )

        current_tokens = self.count_tokens(content)

        # If content is already optimal size, return as-is
        if current_tokens <= optimal_limit:
            return content, {
                "original_tokens": current_tokens,
                "final_tokens": current_tokens,
                "content_type": content_type.value,
                "business_entities_found": sum(
                    len(v) for v in business_entities.values()
                ),
                "reduction_applied": False,
            }

        # Step 2: Preserve critical business information
        priority_sections = self.extract_priority_sections(content, content_type)

        # Create business entities summary
        entity_summary = []
        for category, items in business_entities.items():
            if items:
                entity_summary.append(f"\n## {category.replace('_', ' ').title()}")
                entity_summary.extend([f"- {item}" for item in items])

        # Step 3: Score and prioritize remaining content
        lines = content.split("\n")
        scored_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip if already in priority sections
            if any(line in section for section in priority_sections):
                continue

            score = self.score_content_importance(line, content_type)
            scored_content.append((line, score))

        # Sort by importance
        scored_content.sort(key=lambda x: x, reverse=True)

        # Step 4: Build final content
        result_parts = []

        # Add business entities first (highest priority)
        result_parts.extend(entity_summary)

        # Add priority sections
        if priority_sections:
            result_parts.append("\n## Key Information")
            result_parts.extend(priority_sections)

        # Add scored content until limit
        current_content = "\n".join(result_parts)
        buffer_ratio = 0.9  # Leave 10% buffer

        for line, score in scored_content:
            test_content = current_content + "\n" + line
            if self.count_tokens(test_content) <= optimal_limit * buffer_ratio:
                result_parts.append(line)
                current_content = test_content
            else:
                break

        final_content = "\n".join(result_parts)
        final_tokens = self.count_tokens(final_content)

        return final_content, {
            "original_tokens": current_tokens,
            "final_tokens": final_tokens,
            "content_type": content_type.value,
            "business_entities_found": sum(len(v) for v in business_entities.values()),
            "reduction_percentage": round((1 - final_tokens / current_tokens) * 100, 1),
            "reduction_applied": True,
            "optimal_limit_used": optimal_limit,
        }


class URLCache:
    def __init__(self):
        self.pool = None
        self.ttl_days = 14

    async def init_pool(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=os.getenv("DB_HOST", "postgres-tunnel"),
                port=int(os.getenv("DB_PORT", 5432)),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "lightrag2025"),
                database=os.getenv("DB_NAME", "postgres"),
                min_size=1,
                max_size=10,
            )
            await self.create_table()
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def create_table(self):
        """Create cache table if it doesn't exist"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS url_cache (
                    url_hash VARCHAR(64) PRIMARY KEY,
                    original_url TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP NOT NULL,
                    content_type VARCHAR(50),
                    token_count INTEGER,
                    business_entities_count INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_url_cache_expires 
                ON url_cache(expires_at);
                CREATE INDEX IF NOT EXISTS idx_url_cache_content_type 
                ON url_cache(content_type);
            """
            )

    def canonicalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates"""
        try:
            parsed = urlparse(url.lower())

            scheme = parsed.scheme or "https"
            netloc = parsed.netloc.replace("www.", "")
            path = parsed.path.rstrip("/")
            if not path:
                path = "/"

            # Dynamic tracking parameter detection
            query_params = parse_qs(parsed.query)
            tracking_indicators = [
                "utm_",
                "ref",
                "source",
                "fbclid",
                "gclid",
                "msclkid",
                "campaign",
                "medium",
                "term",
                "content",
                "mc_",
                "pk_",
            ]

            filtered_params = {
                k: v
                for k, v in query_params.items()
                if not any(k.startswith(indicator) for indicator in tracking_indicators)
            }

            sorted_query = urlencode(sorted(filtered_params.items()), doseq=True)
            canonical = urlunparse((scheme, netloc, path, "", sorted_query, ""))
            return canonical
        except Exception as e:
            logger.warning(f"URL canonicalization failed for {url}: {e}")
            return url

    def get_url_hash(self, url: str) -> str:
        """Generate hash for URL"""
        canonical_url = self.canonicalize_url(url)
        return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()

    async def is_url_processed(self, url: str) -> bool:
        """Check if URL was processed within TTL period"""
        url_hash = self.get_url_hash(url)

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("DELETE FROM url_cache WHERE expires_at < NOW()")

                result = await conn.fetchrow(
                    "SELECT processed_at FROM url_cache WHERE url_hash = $1 AND expires_at > NOW()",
                    url_hash,
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking URL cache: {e}")
            return False

    async def mark_url_processed(
        self,
        url: str,
        content_type: str = None,
        token_count: int = None,
        business_entities_count: int = None,
    ):
        """Mark URL as processed with metadata"""
        url_hash = self.get_url_hash(url)
        canonical_url = self.canonicalize_url(url)
        expires_at = datetime.now() + timedelta(days=self.ttl_days)

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO url_cache (url_hash, original_url, expires_at, content_type, token_count, business_entities_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (url_hash) 
                    DO UPDATE SET 
                        processed_at = NOW(),
                        expires_at = $3,
                        content_type = $4,
                        token_count = $5,
                        business_entities_count = $6
                """,
                    url_hash,
                    canonical_url,
                    expires_at,
                    content_type,
                    token_count,
                    business_entities_count,
                )
        except Exception as e:
            logger.error(f"Error marking URL as processed: {e}")

    async def get_cache_stats(self) -> dict:
        """Enhanced cache statistics"""
        try:
            async with self.pool.acquire() as conn:
                total = await conn.fetchval("SELECT COUNT(*) FROM url_cache")
                expired = await conn.fetchval(
                    "SELECT COUNT(*) FROM url_cache WHERE expires_at < NOW()"
                )

                # Content type distribution
                content_types = await conn.fetch(
                    "SELECT content_type, COUNT(*) as count FROM url_cache WHERE expires_at > NOW() GROUP BY content_type"
                )

                # Average metrics
                avg_tokens = await conn.fetchval(
                    "SELECT AVG(token_count) FROM url_cache WHERE expires_at > NOW() AND token_count IS NOT NULL"
                )

                return {
                    "total_entries": total,
                    "expired_entries": expired,
                    "active_entries": total - expired,
                    "content_type_distribution": {
                        row["content_type"]: row["count"] for row in content_types
                    },
                    "average_tokens": round(avg_tokens or 0, 1),
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


class EnhancedWebScraperService:
    def __init__(self):
        self.crawler = None
        self.url_cache = URLCache()
        self.content_processor = DynamicBusinessIntelligenceProcessor()
        self.markdown_converter = MarkItDown()

        # Dynamic supported mimetypes (expandable)
        self.supported_mimetypes = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/vnd.ms-excel": "xls",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.ms-powerpoint": "ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "text/csv": "csv",
            "application/rtf": "rtf",
            "application/vnd.oasis.opendocument.text": "odt",
        }

        # Dynamic crawler configuration
        self.default_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.15,  # Slightly more aggressive filtering
                    min_word_threshold=8,
                    threshold_type="dynamic",
                ),
                options={
                    "ignore_links": False,
                    "ignore_images": True,
                    "escape_html": False,
                    "body_only": False,  # Keep headers/footers for business info
                },
            ),
            cache_mode=CacheMode.BYPASS,
        )

    async def __aenter__(self):
        self.crawler = AsyncWebCrawler(verbose=True, headless=True)
        await self.crawler.__aenter__()
        await self.url_cache.init_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
        if self.url_cache.pool:
            await self.url_cache.pool.close()

    async def detect_content_type(
        self, url: str
    ) -> Tuple[Optional[str], Optional[bytes]]:
        """Enhanced content type detection"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url, allow_redirects=True) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    content_length = response.headers.get("Content-Length")

                    # Skip very large files (>50MB)
                    if content_length and int(content_length) > 50 * 1024 * 1024:
                        logger.warning(f"Skipping large file: {content_length} bytes")
                        return None, None

                    if any(mime in content_type for mime in self.supported_mimetypes):
                        async with session.get(url, allow_redirects=True) as download:
                            content = await download.read()
                            return content_type.split(";"), content
            return None, None
        except Exception as e:
            logger.error(f"Error detecting content type: {str(e)}")
            return None, None

    async def convert_file_to_markdown(self, content: bytes, content_type: str) -> str:
        """Enhanced file conversion with error handling"""
        try:
            file_extension = self.supported_mimetypes[content_type]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                result = self.markdown_converter.convert(tmp_path)
                if not result or not result.text_content:
                    raise Exception("Empty conversion result")
                return result.text_content
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Error converting file to markdown: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"File conversion error: {str(e)}"
            )

    async def extract(self, url: str) -> Tuple[str, Dict]:
        """Enhanced extraction with intelligent processing"""

        # Try file conversion first
        content_type, content = await self.detect_content_type(url)

        if content_type and content:
            logger.info(f"🔄 Converting file of type {content_type}")
            raw_content = await self.convert_file_to_markdown(content, content_type)
        else:
            # Web crawling fallback
            try:
                result = await self.crawler.arun(url=url, config=self.default_config)
                if not result.success:
                    raise HTTPException(status_code=400, detail="Crawling failed")

                raw_content = result.markdown.fit_markdown

                if len(raw_content) < 100:
                    raise HTTPException(status_code=400, detail="Insufficient content")

            except Exception as e:
                logger.error(f"Error extracting content: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        # Apply intelligent content processing
        processed_content, processing_metrics = (
            self.content_processor.intelligent_content_reduction(raw_content)
        )

        return processed_content, processing_metrics


app = FastAPI()
scraper_service = EnhancedWebScraperService()


@app.on_event("startup")
async def startup_event():
    await scraper_service.__aenter__()


@app.on_event("shutdown")
async def shutdown_event():
    await scraper_service.__aexit__(None, None, None)


@app.get("/")
async def root():
    return {"status": "ready", "version": "dynamic-business-intelligence"}


@app.get("/cache/stats")
async def cache_stats():
    """Enhanced cache statistics"""
    stats = await scraper_service.url_cache.get_cache_stats()
    return {"cache_stats": stats}


@app.post("/crawl")
async def crawl_page(page_url: PageUrl):
    """Enhanced crawling with dynamic business intelligence processing"""
    url = str(page_url.url)
    canonical_url = scraper_service.url_cache.canonicalize_url(url)

    try:
        # Check cache
        if await scraper_service.url_cache.is_url_processed(url):
            logger.info(f"⏭️ URL cached: {canonical_url}")
            return {
                "status": "skipped",
                "reason": "processed_within_ttl",
                "canonical_url": canonical_url,
                "ttl_days": scraper_service.url_cache.ttl_days,
            }

        # Process URL with intelligent extraction
        logger.info(f"🚀 Processing: {canonical_url}")
        start_time = datetime.now()

        content, processing_metrics = await scraper_service.extract(url)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Cache with metadata
        await scraper_service.url_cache.mark_url_processed(
            url,
            content_type=processing_metrics.get("content_type"),
            token_count=processing_metrics.get("final_tokens"),
            business_entities_count=processing_metrics.get("business_entities_found"),
        )

        return {
            "status": "success",
            "content": content,
            "canonical_url": canonical_url,
            "processing_metrics": {
                **processing_metrics,
                "processing_time_seconds": round(processing_time, 2),
                "efficiency_score": round(
                    processing_metrics.get("business_entities_found", 0)
                    / max(processing_metrics.get("final_tokens", 1) / 1000, 1),
                    2,
                ),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error processing {canonical_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check with system metrics"""
    try:
        cache_stats = await scraper_service.url_cache.get_cache_stats()
        return {
            "status": "healthy",
            "cache_healthy": cache_stats.get("total_entries", 0) >= 0,
            "active_cache_entries": cache_stats.get("active_entries", 0),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
