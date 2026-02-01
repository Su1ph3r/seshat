"""
Reddit scraper for Seshat.

Scrapes posts and comments from Reddit.
"""

from typing import Optional, Generator, Dict, Any, List
from datetime import datetime

from scraper.base import BaseScraper, ScrapedContent
from scraper.rate_limiter import RateLimiter


class RedditScraper(BaseScraper):
    """
    Scraper for Reddit platform.

    Uses PRAW (Python Reddit API Wrapper) or JSON API.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        rate_limit: float = 1.0,
        proxy: Optional[str] = None,
    ):
        """
        Initialize Reddit scraper.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
            rate_limit: Seconds between requests
            proxy: Optional proxy URL
        """
        default_ua = "Seshat/1.0 (Stylometric Analysis Tool)"
        super().__init__(rate_limit, proxy, user_agent or default_ua)

        self.client_id = client_id
        self.client_secret = client_secret
        self._reddit = None
        self._rate_limiter = RateLimiter.for_reddit()

    @property
    def platform_name(self) -> str:
        return "reddit"

    def _get_reddit_client(self):
        """Get or create PRAW Reddit client."""
        if self._reddit is not None:
            return self._reddit

        if self.client_id and self.client_secret:
            try:
                import praw

                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
                return self._reddit
            except ImportError:
                pass

        return None

    def scrape_user(
        self,
        username: str,
        limit: int = 100,
        include_posts: bool = True,
        include_comments: bool = True,
        sort: str = "new",
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape content from a Reddit user.

        Args:
            username: Reddit username (without u/)
            limit: Maximum items to scrape
            include_posts: Include user's posts
            include_comments: Include user's comments
            sort: Sort order (new, hot, top, controversial)

        Yields:
            ScrapedContent for each post/comment
        """
        username = username.lstrip("u/")

        reddit = self._get_reddit_client()

        if reddit:
            yield from self._scrape_user_praw(
                reddit, username, limit, include_posts, include_comments, sort
            )
        else:
            yield from self._scrape_user_json(
                username, limit, include_posts, include_comments, sort
            )

    def _scrape_user_praw(
        self,
        reddit,
        username: str,
        limit: int,
        include_posts: bool,
        include_comments: bool,
        sort: str,
    ) -> Generator[ScrapedContent, None, None]:
        """Scrape user using PRAW."""
        try:
            user = reddit.redditor(username)
            count = 0

            if include_posts:
                submissions = getattr(user.submissions, sort)(limit=limit)
                for submission in submissions:
                    if count >= limit:
                        break

                    self._rate_limiter.wait()

                    text = submission.selftext if submission.is_self else submission.title
                    if not text.strip():
                        text = submission.title

                    yield ScrapedContent(
                        text=text,
                        source_url=f"https://reddit.com{submission.permalink}",
                        platform=self.platform_name,
                        author=username,
                        timestamp=datetime.fromtimestamp(submission.created_utc),
                        metadata={
                            "type": "post",
                            "subreddit": submission.subreddit.display_name,
                            "title": submission.title,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                        },
                    )
                    count += 1

            if include_comments and count < limit:
                remaining = limit - count
                comments = getattr(user.comments, sort)(limit=remaining)

                for comment in comments:
                    if count >= limit:
                        break

                    self._rate_limiter.wait()

                    if not comment.body.strip():
                        continue

                    yield ScrapedContent(
                        text=comment.body,
                        source_url=f"https://reddit.com{comment.permalink}",
                        platform=self.platform_name,
                        author=username,
                        timestamp=datetime.fromtimestamp(comment.created_utc),
                        metadata={
                            "type": "comment",
                            "subreddit": comment.subreddit.display_name,
                            "score": comment.score,
                            "parent_id": comment.parent_id,
                        },
                    )
                    count += 1

        except Exception:
            return

    def _scrape_user_json(
        self,
        username: str,
        limit: int,
        include_posts: bool,
        include_comments: bool,
        sort: str,
    ) -> Generator[ScrapedContent, None, None]:
        """Scrape user using Reddit JSON API."""
        base_url = f"https://www.reddit.com/user/{username}"
        count = 0
        after = None

        with self._get_session() as session:
            while count < limit:
                self._rate_limiter.wait()

                url = f"{base_url}.json?sort={sort}&limit=100"
                if after:
                    url += f"&after={after}"

                try:
                    response = session.get(url)
                    if response.status_code != 200:
                        break

                    data = response.json()
                    children = data.get("data", {}).get("children", [])

                    if not children:
                        break

                    for child in children:
                        if count >= limit:
                            break

                        kind = child.get("kind")
                        item_data = child.get("data", {})

                        if kind == "t3" and include_posts:
                            text = item_data.get("selftext") or item_data.get("title", "")
                            if not text.strip():
                                continue

                            yield ScrapedContent(
                                text=text,
                                source_url=f"https://reddit.com{item_data.get('permalink', '')}",
                                platform=self.platform_name,
                                author=username,
                                timestamp=datetime.fromtimestamp(item_data.get("created_utc", 0)),
                                metadata={
                                    "type": "post",
                                    "subreddit": item_data.get("subreddit"),
                                    "title": item_data.get("title"),
                                    "score": item_data.get("score"),
                                },
                            )
                            count += 1

                        elif kind == "t1" and include_comments:
                            text = item_data.get("body", "")
                            if not text.strip():
                                continue

                            yield ScrapedContent(
                                text=text,
                                source_url=f"https://reddit.com{item_data.get('permalink', '')}",
                                platform=self.platform_name,
                                author=username,
                                timestamp=datetime.fromtimestamp(item_data.get("created_utc", 0)),
                                metadata={
                                    "type": "comment",
                                    "subreddit": item_data.get("subreddit"),
                                    "score": item_data.get("score"),
                                },
                            )
                            count += 1

                    after = data.get("data", {}).get("after")
                    if not after:
                        break

                except Exception:
                    break

    def scrape_search(
        self,
        query: str,
        limit: int = 100,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        time_filter: str = "all",
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape posts matching a search query.

        Args:
            query: Search query
            limit: Maximum posts to scrape
            subreddit: Limit to specific subreddit
            sort: Sort order (relevance, hot, top, new, comments)
            time_filter: Time filter (all, hour, day, week, month, year)

        Yields:
            ScrapedContent for each matching post
        """
        reddit = self._get_reddit_client()

        if reddit:
            yield from self._search_praw(reddit, query, limit, subreddit, sort, time_filter)
        else:
            yield from self._search_json(query, limit, subreddit, sort, time_filter)

    def _search_praw(
        self,
        reddit,
        query: str,
        limit: int,
        subreddit: Optional[str],
        sort: str,
        time_filter: str,
    ) -> Generator[ScrapedContent, None, None]:
        """Search using PRAW."""
        try:
            if subreddit:
                sub = reddit.subreddit(subreddit)
                results = sub.search(query, sort=sort, time_filter=time_filter, limit=limit)
            else:
                results = reddit.subreddit("all").search(
                    query, sort=sort, time_filter=time_filter, limit=limit
                )

            count = 0
            for submission in results:
                if count >= limit:
                    break

                self._rate_limiter.wait()

                text = submission.selftext if submission.is_self else submission.title
                if not text.strip():
                    text = submission.title

                yield ScrapedContent(
                    text=text,
                    source_url=f"https://reddit.com{submission.permalink}",
                    platform=self.platform_name,
                    author=submission.author.name if submission.author else "[deleted]",
                    timestamp=datetime.fromtimestamp(submission.created_utc),
                    metadata={
                        "type": "post",
                        "subreddit": submission.subreddit.display_name,
                        "title": submission.title,
                        "score": submission.score,
                        "search_query": query,
                    },
                )
                count += 1

        except Exception:
            return

    def _search_json(
        self,
        query: str,
        limit: int,
        subreddit: Optional[str],
        sort: str,
        time_filter: str,
    ) -> Generator[ScrapedContent, None, None]:
        """Search using JSON API."""
        if subreddit:
            base_url = f"https://www.reddit.com/r/{subreddit}/search.json"
        else:
            base_url = "https://www.reddit.com/search.json"

        count = 0
        after = None

        with self._get_session() as session:
            while count < limit:
                self._rate_limiter.wait()

                url = f"{base_url}?q={query}&sort={sort}&t={time_filter}&limit=100"
                if after:
                    url += f"&after={after}"

                try:
                    response = session.get(url)
                    if response.status_code != 200:
                        break

                    data = response.json()
                    children = data.get("data", {}).get("children", [])

                    if not children:
                        break

                    for child in children:
                        if count >= limit:
                            break

                        item_data = child.get("data", {})
                        text = item_data.get("selftext") or item_data.get("title", "")

                        if not text.strip():
                            continue

                        yield ScrapedContent(
                            text=text,
                            source_url=f"https://reddit.com{item_data.get('permalink', '')}",
                            platform=self.platform_name,
                            author=item_data.get("author", "[deleted]"),
                            timestamp=datetime.fromtimestamp(item_data.get("created_utc", 0)),
                            metadata={
                                "type": "post",
                                "subreddit": item_data.get("subreddit"),
                                "title": item_data.get("title"),
                                "score": item_data.get("score"),
                                "search_query": query,
                            },
                        )
                        count += 1

                    after = data.get("data", {}).get("after")
                    if not after:
                        break

                except Exception:
                    break

    def scrape_subreddit(
        self,
        subreddit: str,
        limit: int = 100,
        sort: str = "new",
        time_filter: str = "all",
    ) -> Generator[ScrapedContent, None, None]:
        """
        Scrape posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            limit: Maximum posts to scrape
            sort: Sort order (new, hot, top, rising, controversial)
            time_filter: Time filter for top/controversial

        Yields:
            ScrapedContent for each post
        """
        subreddit = subreddit.lstrip("r/")

        reddit = self._get_reddit_client()

        if reddit:
            try:
                sub = reddit.subreddit(subreddit)
                submissions = getattr(sub, sort)(limit=limit, time_filter=time_filter)

                for submission in submissions:
                    self._rate_limiter.wait()

                    text = submission.selftext if submission.is_self else submission.title

                    yield ScrapedContent(
                        text=text,
                        source_url=f"https://reddit.com{submission.permalink}",
                        platform=self.platform_name,
                        author=submission.author.name if submission.author else "[deleted]",
                        timestamp=datetime.fromtimestamp(submission.created_utc),
                        metadata={
                            "type": "post",
                            "subreddit": subreddit,
                            "title": submission.title,
                            "score": submission.score,
                        },
                    )

            except Exception:
                return
        else:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}&t={time_filter}"

            with self._get_session() as session:
                self._rate_limiter.wait()

                try:
                    response = session.get(url)
                    if response.status_code != 200:
                        return

                    data = response.json()
                    for child in data.get("data", {}).get("children", []):
                        item_data = child.get("data", {})

                        text = item_data.get("selftext") or item_data.get("title", "")

                        yield ScrapedContent(
                            text=text,
                            source_url=f"https://reddit.com{item_data.get('permalink', '')}",
                            platform=self.platform_name,
                            author=item_data.get("author", "[deleted]"),
                            timestamp=datetime.fromtimestamp(item_data.get("created_utc", 0)),
                            metadata={
                                "type": "post",
                                "subreddit": subreddit,
                                "title": item_data.get("title"),
                                "score": item_data.get("score"),
                            },
                        )

                except Exception:
                    return
