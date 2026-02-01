"""
Twitter/X scraper for Seshat.

Scrapes tweets and user content from Twitter/X.
"""

from typing import Optional, Generator, Dict, Any, List
from datetime import datetime
import re

from scraper.base import BaseScraper, ScrapedContent
from scraper.rate_limiter import RateLimiter


class TwitterScraper(BaseScraper):
    """
    Scraper for Twitter/X platform.

    Uses various methods to collect tweets including
    guest API access and browser automation.
    """

    def __init__(
        self,
        rate_limit: float = 2.0,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        use_browser: bool = False,
    ):
        """
        Initialize Twitter scraper.

        Args:
            rate_limit: Seconds between requests
            proxy: Optional proxy URL
            user_agent: Optional user agent
            use_browser: Use browser automation (slower but more reliable)
        """
        super().__init__(rate_limit, proxy, user_agent)
        self.use_browser = use_browser
        self._rate_limiter = RateLimiter.for_twitter()

    @property
    def platform_name(self) -> str:
        return "twitter"

    def scrape_user(
        self,
        username: str,
        limit: int = 100,
        include_replies: bool = False,
        include_retweets: bool = False,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape tweets from a user's timeline.

        Args:
            username: Twitter username (without @)
            limit: Maximum tweets to scrape
            include_replies: Include reply tweets
            include_retweets: Include retweets
            since: Only tweets after this date
            until: Only tweets before this date

        Yields:
            ScrapedContent for each tweet
        """
        username = username.lstrip("@")

        try:
            yield from self._scrape_with_nitter(
                username, limit, include_replies, since, until
            )
        except Exception:
            yield from self._scrape_with_browser(
                username, limit, include_replies, since, until
            )

    def _scrape_with_nitter(
        self,
        username: str,
        limit: int,
        include_replies: bool,
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> Generator[ScrapedContent, None, None]:
        """Scrape using Nitter instances (public Twitter frontend)."""
        nitter_instances = [
            "nitter.net",
            "nitter.unixfox.eu",
            "nitter.it",
        ]

        count = 0

        for instance in nitter_instances:
            if count >= limit:
                break

            try:
                url = f"https://{instance}/{username}"
                if include_replies:
                    url += "/with_replies"

                with self._get_session() as session:
                    self._rate_limiter.wait()
                    response = session.get(url)

                    if response.status_code != 200:
                        continue

                    for tweet in self._parse_nitter_html(
                        response.text, username, instance
                    ):
                        if since and tweet.timestamp and tweet.timestamp < since:
                            continue
                        if until and tweet.timestamp and tweet.timestamp > until:
                            continue

                        yield tweet
                        count += 1

                        if count >= limit:
                            break

                    break

            except Exception:
                continue

    def _parse_nitter_html(
        self,
        html: str,
        username: str,
        instance: str,
    ) -> Generator[ScrapedContent, None, None]:
        """Parse tweets from Nitter HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")

            for tweet_div in soup.select(".timeline-item"):
                content_div = tweet_div.select_one(".tweet-content")
                if not content_div:
                    continue

                text = content_div.get_text(strip=True)
                if not text:
                    continue

                link = tweet_div.select_one(".tweet-link")
                tweet_url = ""
                if link and link.get("href"):
                    tweet_url = f"https://twitter.com{link['href']}"

                timestamp = None
                time_elem = tweet_div.select_one(".tweet-date a")
                if time_elem and time_elem.get("title"):
                    try:
                        timestamp = datetime.strptime(
                            time_elem["title"], "%b %d, %Y · %I:%M %p %Z"
                        )
                    except ValueError:
                        pass

                yield ScrapedContent(
                    text=text,
                    source_url=tweet_url or f"https://twitter.com/{username}",
                    platform=self.platform_name,
                    author=username,
                    timestamp=timestamp,
                    metadata={
                        "scrape_method": "nitter",
                        "nitter_instance": instance,
                    },
                )

        except ImportError:
            return

    def _scrape_with_browser(
        self,
        username: str,
        limit: int,
        include_replies: bool,
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> Generator[ScrapedContent, None, None]:
        """Scrape using browser automation (Playwright)."""
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1280, "height": 720},
                )
                page = context.new_page()

                url = f"https://twitter.com/{username}"
                if include_replies:
                    url += "/with_replies"

                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(3000)

                tweets_seen = set()
                count = 0

                while count < limit:
                    tweet_articles = page.query_selector_all('article[data-testid="tweet"]')

                    for article in tweet_articles:
                        if count >= limit:
                            break

                        try:
                            text_div = article.query_selector('[data-testid="tweetText"]')
                            if not text_div:
                                continue

                            text = text_div.inner_text()
                            if not text or text in tweets_seen:
                                continue

                            tweets_seen.add(text)

                            time_elem = article.query_selector("time")
                            timestamp = None
                            if time_elem:
                                dt_str = time_elem.get_attribute("datetime")
                                if dt_str:
                                    timestamp = datetime.fromisoformat(
                                        dt_str.replace("Z", "+00:00")
                                    )

                            if since and timestamp and timestamp < since:
                                continue
                            if until and timestamp and timestamp > until:
                                continue

                            link_elem = article.query_selector('a[href*="/status/"]')
                            tweet_url = ""
                            if link_elem:
                                href = link_elem.get_attribute("href")
                                if href:
                                    tweet_url = f"https://twitter.com{href}"

                            yield ScrapedContent(
                                text=text,
                                source_url=tweet_url or url,
                                platform=self.platform_name,
                                author=username,
                                timestamp=timestamp,
                                metadata={"scrape_method": "browser"},
                            )

                            count += 1

                        except Exception:
                            continue

                    page.keyboard.press("End")
                    page.wait_for_timeout(2000)

                    new_articles = page.query_selector_all('article[data-testid="tweet"]')
                    if len(new_articles) == len(tweet_articles):
                        break

                browser.close()

        except ImportError:
            return

    def scrape_search(
        self,
        query: str,
        limit: int = 100,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape tweets matching a search query.

        Args:
            query: Search query
            limit: Maximum tweets to scrape
            since: Only tweets after this date
            until: Only tweets before this date

        Yields:
            ScrapedContent for each matching tweet
        """
        search_query = query
        if since:
            search_query += f" since:{since.strftime('%Y-%m-%d')}"
        if until:
            search_query += f" until:{until.strftime('%Y-%m-%d')}"

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=self.user_agent)
                page = context.new_page()

                encoded_query = search_query.replace(" ", "%20")
                url = f"https://twitter.com/search?q={encoded_query}&src=typed_query&f=live"

                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(3000)

                tweets_seen = set()
                count = 0

                while count < limit:
                    tweet_articles = page.query_selector_all('article[data-testid="tweet"]')

                    for article in tweet_articles:
                        if count >= limit:
                            break

                        try:
                            text_div = article.query_selector('[data-testid="tweetText"]')
                            if not text_div:
                                continue

                            text = text_div.inner_text()
                            if not text or text in tweets_seen:
                                continue

                            tweets_seen.add(text)

                            user_link = article.query_selector('a[href^="/"][role="link"]')
                            author = None
                            if user_link:
                                href = user_link.get_attribute("href")
                                if href:
                                    author = href.strip("/").split("/")[0]

                            time_elem = article.query_selector("time")
                            timestamp = None
                            if time_elem:
                                dt_str = time_elem.get_attribute("datetime")
                                if dt_str:
                                    timestamp = datetime.fromisoformat(
                                        dt_str.replace("Z", "+00:00")
                                    )

                            yield ScrapedContent(
                                text=text,
                                source_url=url,
                                platform=self.platform_name,
                                author=author,
                                timestamp=timestamp,
                                metadata={
                                    "scrape_method": "browser",
                                    "search_query": query,
                                },
                            )

                            count += 1

                        except Exception:
                            continue

                    page.keyboard.press("End")
                    page.wait_for_timeout(2000)

                browser.close()

        except ImportError:
            return

    def scrape_thread(
        self,
        thread_url: str,
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape all tweets in a thread.

        Args:
            thread_url: URL of the thread's first tweet

        Yields:
            ScrapedContent for each tweet in thread
        """
        match = re.search(r"twitter\.com/(\w+)/status/(\d+)", thread_url)
        if not match:
            return

        username = match.group(1)

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                page.goto(thread_url, wait_until="networkidle")
                page.wait_for_timeout(3000)

                tweets_seen = set()

                tweet_articles = page.query_selector_all('article[data-testid="tweet"]')

                for article in tweet_articles:
                    try:
                        text_div = article.query_selector('[data-testid="tweetText"]')
                        if not text_div:
                            continue

                        text = text_div.inner_text()
                        if not text or text in tweets_seen:
                            continue

                        tweets_seen.add(text)

                        yield ScrapedContent(
                            text=text,
                            source_url=thread_url,
                            platform=self.platform_name,
                            author=username,
                            metadata={"scrape_method": "browser", "is_thread": True},
                        )

                    except Exception:
                        continue

                browser.close()

        except ImportError:
            return
