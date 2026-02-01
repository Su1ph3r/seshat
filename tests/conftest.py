"""
Pytest configuration and fixtures for Seshat tests.
"""

import pytest
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text():
    """Return sample text for testing."""
    return """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter
    of the alphabet. I think it's quite remarkable, don't you? We use it frequently
    in typography and font testing.

    Writing style analysis involves examining various linguistic features. These include
    vocabulary choices, sentence structure, punctuation patterns, and more. Each author
    has a unique "fingerprint" in their writing.

    The analysis can reveal interesting patterns about the author's background,
    education level, and even psychological traits. It's a fascinating field that
    combines linguistics, statistics, and computer science.
    """


@pytest.fixture
def short_text():
    """Return minimal valid text."""
    return "Hello world, this is a test."


@pytest.fixture
def empty_text():
    """Return empty string."""
    return ""


@pytest.fixture
def sample_tweets():
    """Return sample tweet-like texts."""
    return [
        "Just had the best coffee ever! ☕ #morningvibes",
        "Can't believe it's already Friday... time flies when you're having fun!",
        "Working on a new project. Really excited about the possibilities!!! 🚀",
        "The weather today is absolutely gorgeous. Perfect for a walk 🌞",
        "Anyone else think pineapple belongs on pizza? 🍕🍍 Controversial opinion lol",
    ]


@pytest.fixture
def formal_text():
    """Return formal writing sample."""
    return """
    Dear Sir or Madam,

    I am writing to express my interest in the position advertised in the newspaper.
    Having reviewed the requirements carefully, I believe my qualifications align
    well with the expectations outlined in your posting.

    Throughout my career, I have demonstrated proficiency in various areas relevant
    to this role. Furthermore, my educational background has provided me with a
    strong foundation in the necessary skills.

    I would welcome the opportunity to discuss my application further. Please do
    not hesitate to contact me at your earliest convenience.

    Yours faithfully,
    John Smith
    """


@pytest.fixture
def informal_text():
    """Return informal writing sample."""
    return """
    hey! so i was thinking about that thing we talked about yesterday and i totally
    agree with you lol. like, it's so obvious when you really think about it!!

    anyway, wanna grab lunch later? there's this new place downtown that's supposed
    to be amazing. my friend went last week and said the tacos are incredible.

    let me know what you think! btw did you see that video i sent? hilarious right??
    """


@pytest.fixture
def text_pairs():
    """Return pairs of texts for comparison testing."""
    return [
        (
            "I believe that education is the key to success. One must work diligently.",
            "I believe that learning is essential for achievement. One must persevere.",
        ),
        (
            "yo whats up, just chillin here, nothin much goin on tbh",
            "Hey what's happening, I'm just relaxing, not much going on honestly",
        ),
    ]


@pytest.fixture
def analyzer():
    """Create Analyzer instance."""
    from seshat.analyzer import Analyzer
    return Analyzer()


@pytest.fixture
def profile_manager(tmp_path):
    """Create ProfileManager with temp storage."""
    from seshat.profile import ProfileManager
    return ProfileManager(storage_dir=str(tmp_path / "profiles"))


@pytest.fixture
def sample_profile(profile_manager, sample_text, analyzer):
    """Create a sample profile for testing."""
    profile = profile_manager.create_profile(
        name="TestAuthor",
        samples=[sample_text],
    )
    return profile
