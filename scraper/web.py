"""
Generic web scraper for Seshat.

Scrapes content from arbitrary websites, blogs, and forums.
"""

from typing import Optional, Generator, Dict, Any, List, Set
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re

from scraper.base import BaseScraper, ScrapedContent
from scraper.rate_limiter import RateLimiter


class WebScraper(BaseScraper):
    """
    Generic web scraper for websites and blogs.

    Extracts text content from web pages with configurable
    selectors and crawling behavior.
    """

    def __init__(
        self,
        rate_limit: float = 1.0,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        respect_robots: bool = True,
        max_depth: int = 3,
    ):
        """
        Initialize web scraper.

        Args:
            rate_limit: Seconds between requests
            proxy: Optional proxy URL
            user_agent: Optional user agent
            respect_robots: Respect robots.txt
            max_depth: Maximum crawl depth
        """
        super().__init__(rate_limit, proxy, user_agent)
        self.respect_robots = respect_robots
        self.max_depth = max_depth
        self._rate_limiter = RateLimiter.for_generic()
        self._visited: Set[str] = set()

    @property
    def platform_name(self) -> str:
        return "web"

    def scrape_user(
        self,
        username: str,
        limit: int = 100,
        **kwargs,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Not applicable for generic web scraping.

        Use scrape_url or scrape_blog instead.
        """
        raise NotImplementedError(
            "Generic web scraper does not support user scraping. "
            "Use scrape_url() or scrape_blog() instead."
        )

    def scrape_search(
        self,
        query: str,
        limit: int = 100,
        **kwargs,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Not applicable for generic web scraping.

        Use platform-specific scrapers for search.
        """
        raise NotImplementedError(
            "Generic web scraper does not support search. "
            "Use platform-specific scrapers instead."
        )

    def scrape_url(
        self,
        url: str,
        extract_text: bool = True,
        content_selector: Optional[str] = None,
    ) -> Optional[ScrapedContent]:
        """
        Scrape content from a single URL.

        Args:
            url: URL to scrape
            extract_text: Extract main text content
            content_selector: CSS selector for content area

        Returns:
            ScrapedContent or None if failed
        """
        self._rate_limiter.wait()

        try:
            with self._get_session() as session:
                response = session.get(url)

                if response.status_code != 200:
                    return None

                html = response.text
                text, metadata = self._extract_content(html, content_selector)

                if not text or len(text.strip()) < 10:
                    return None

                return ScrapedContent(
                    text=text,
                    source_url=url,
                    platform=self.platform_name,
                    author=metadata.get("author"),
                    timestamp=metadata.get("published_date"),
                    metadata=metadata,
                )

        except Exception:
            return None

    def scrape_blog(
        self,
        blog_url: str,
        max_posts: int = 100,
        follow_pagination: bool = True,
        content_selector: Optional[str] = None,
        post_link_selector: Optional[str] = None,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape posts from a blog.

        Args:
            blog_url: Blog homepage URL
            max_posts: Maximum posts to scrape
            follow_pagination: Follow pagination links
            content_selector: CSS selector for post content
            post_link_selector: CSS selector for post links

        Yields:
            ScrapedContent for each blog post
        """
        self._visited.clear()
        count = 0

        post_links = self._find_post_links(blog_url, post_link_selector)

        for link in post_links:
            if count >= max_posts:
                break

            if link in self._visited:
                continue

            self._visited.add(link)
            content = self.scrape_url(link, content_selector=content_selector)

            if content:
                yield content
                count += 1

        if follow_pagination and count < max_posts:
            pagination_links = self._find_pagination_links(blog_url)

            for page_url in pagination_links:
                if count >= max_posts:
                    break

                self._rate_limiter.wait()
                page_posts = self._find_post_links(page_url, post_link_selector)

                for link in page_posts:
                    if count >= max_posts:
                        break

                    if link in self._visited:
                        continue

                    self._visited.add(link)
                    content = self.scrape_url(link, content_selector=content_selector)

                    if content:
                        yield content
                        count += 1

    def scrape_sitemap(
        self,
        sitemap_url: str,
        limit: int = 100,
        url_filter: Optional[str] = None,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape pages listed in a sitemap.

        Args:
            sitemap_url: URL of sitemap.xml
            limit: Maximum pages to scrape
            url_filter: Regex filter for URLs

        Yields:
            ScrapedContent for each page
        """
        self._rate_limiter.wait()

        try:
            with self._get_session() as session:
                response = session.get(sitemap_url)

                if response.status_code != 200:
                    return

                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "xml")
                urls = [loc.text for loc in soup.find_all("loc")]

                if url_filter:
                    pattern = re.compile(url_filter)
                    urls = [u for u in urls if pattern.search(u)]

                count = 0
                for url in urls:
                    if count >= limit:
                        break

                    content = self.scrape_url(url)
                    if content:
                        yield content
                        count += 1

        except Exception:
            return

    def crawl(
        self,
        start_url: str,
        max_pages: int = 100,
        same_domain_only: bool = True,
        url_pattern: Optional[str] = None,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Crawl a website starting from a URL.

        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            same_domain_only: Only follow same-domain links
            url_pattern: Regex pattern to filter URLs

        Yields:
            ScrapedContent for each page
        """
        self._visited.clear()
        to_visit = [start_url]
        domain = urlparse(start_url).netloc
        count = 0

        pattern = re.compile(url_pattern) if url_pattern else None

        while to_visit and count < max_pages:
            url = to_visit.pop(0)

            if url in self._visited:
                continue

            if same_domain_only and urlparse(url).netloc != domain:
                continue

            if pattern and not pattern.search(url):
                continue

            self._visited.add(url)
            self._rate_limiter.wait()

            try:
                with self._get_session() as session:
                    response = session.get(url)

                    if response.status_code != 200:
                        continue

                    html = response.text
                    text, metadata = self._extract_content(html)

                    if text and len(text.strip()) >= 50:
                        yield ScrapedContent(
                            text=text,
                            source_url=url,
                            platform=self.platform_name,
                            author=metadata.get("author"),
                            timestamp=metadata.get("published_date"),
                            metadata=metadata,
                        )
                        count += 1

                    links = self._extract_links(html, url)
                    for link in links:
                        if link not in self._visited and link not in to_visit:
                            to_visit.append(link)

            except Exception:
                continue

    def _extract_content(
        self,
        html: str,
        content_selector: Optional[str] = None,
    ) -> tuple:
        """Extract main content from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")

            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            metadata = self._extract_metadata(soup)

            if content_selector:
                content = soup.select_one(content_selector)
                if content:
                    return content.get_text(separator=" ", strip=True), metadata

            content_selectors = [
                "article",
                '[role="main"]',
                ".post-content",
                ".entry-content",
                ".article-content",
                ".content",
                "main",
            ]

            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    text = content.get_text(separator=" ", strip=True)
                    if len(text) > 100:
                        return text, metadata

            body = soup.find("body")
            if body:
                return body.get_text(separator=" ", strip=True), metadata

            return soup.get_text(separator=" ", strip=True), metadata

        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text, {}

    def _extract_metadata(self, soup) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}

        title = soup.find("title")
        if title:
            metadata["title"] = title.get_text(strip=True)

        author_selectors = [
            ('meta[name="author"]', "content"),
            ('meta[property="article:author"]', "content"),
            (".author", None),
            ('[rel="author"]', None),
        ]

        for selector, attr in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                if attr:
                    metadata["author"] = elem.get(attr)
                else:
                    metadata["author"] = elem.get_text(strip=True)
                break

        date_selectors = [
            ('meta[property="article:published_time"]', "content"),
            ('meta[name="date"]', "content"),
            ("time", "datetime"),
            (".published", None),
            (".date", None),
        ]

        for selector, attr in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                date_str = elem.get(attr) if attr else elem.get_text(strip=True)
                if date_str:
                    try:
                        if "T" in date_str:
                            metadata["published_date"] = datetime.fromisoformat(
                                date_str.replace("Z", "+00:00")
                            )
                        break
                    except ValueError:
                        pass

        desc = soup.select_one('meta[name="description"]')
        if desc:
            metadata["description"] = desc.get("content")

        return metadata

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            links = []

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("#") or href.startswith("javascript:"):
                    continue

                full_url = urljoin(base_url, href)
                full_url = full_url.split("#")[0]

                if full_url.startswith(("http://", "https://")):
                    links.append(full_url)

            return links

        except ImportError:
            return []

    def _find_post_links(
        self,
        blog_url: str,
        link_selector: Optional[str] = None,
    ) -> List[str]:
        """Find blog post links on a page."""
        self._rate_limiter.wait()

        try:
            with self._get_session() as session:
                response = session.get(blog_url)

                if response.status_code != 200:
                    return []

                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "lxml")

                if link_selector:
                    links = soup.select(link_selector)
                    return [urljoin(blog_url, a.get("href")) for a in links if a.get("href")]

                post_selectors = [
                    "article a",
                    ".post a",
                    ".entry a",
                    ".blog-post a",
                    "h2 a",
                    "h3 a",
                ]

                for selector in post_selectors:
                    links = soup.select(selector)
                    if links:
                        return [
                            urljoin(blog_url, a.get("href"))
                            for a in links
                            if a.get("href") and not a.get("href").startswith("#")
                        ]

                return []

        except Exception:
            return []

    def _find_pagination_links(self, page_url: str) -> List[str]:
        """Find pagination links on a page."""
        try:
            with self._get_session() as session:
                response = session.get(page_url)

                if response.status_code != 200:
                    return []

                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "lxml")

                pagination_selectors = [
                    ".pagination a",
                    ".pager a",
                    ".page-numbers",
                    'a[rel="next"]',
                    ".nav-links a",
                ]

                links = []
                for selector in pagination_selectors:
                    for a in soup.select(selector):
                        href = a.get("href")
                        if href and not href.startswith("#"):
                            full_url = urljoin(page_url, href)
                            if full_url not in links:
                                links.append(full_url)

                return links

        except Exception:
            return []
