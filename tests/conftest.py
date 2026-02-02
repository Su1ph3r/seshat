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


# =============================================================================
# Personality Disorder Test Fixtures
# =============================================================================


@pytest.fixture
def paranoid_text():
    """Return text with paranoid markers."""
    return """
    They are always watching me, plotting and scheming against me behind my back.
    I can't trust anyone because everyone is suspicious and deceitful. They all
    conspire against me and lie to my face. I know they're out to get me. I have
    to be vigilant because they're targeting me. Everyone is untrustworthy and
    I must protect myself from their schemes. They blame me for everything but
    it's really their fault. The world is full of liars and manipulators.
    """ * 10


@pytest.fixture
def narcissistic_text():
    """Return text with narcissistic markers."""
    return """
    I am the best at everything I do. My achievements are extraordinary and
    unmatched by anyone else. I deserve special treatment because I am superior
    to everyone around me. People should recognize my exceptional talents and
    brilliant ideas. These trivial concerns are beneath me. I am unique and
    gifted beyond compare. Everyone should admire my accomplishments.
    """ * 10


@pytest.fixture
def negated_text():
    """Return text with negated markers."""
    return """
    I am not suspicious of anyone. I don't distrust people at all. I'm never
    worried about being watched. I don't think anyone is plotting against me.
    I can trust others completely. No one is deceiving me. I'm not vigilant
    or careful about anything. People are not lying to me.
    """ * 10


@pytest.fixture
def clinical_text():
    """Return clinical-style text."""
    return """
    Patient presents with symptoms of anxiety and depression. Assessment reveals
    moderate impairment in functioning. Diagnosis indicates possible adjustment
    disorder. Treatment plan includes cognitive behavioral therapy and medication
    evaluation. Prognosis is favorable with appropriate intervention. Mental
    status examination shows intact orientation and judgment.
    """ * 10


@pytest.fixture
def social_media_text():
    """Return social media style text."""
    return """
    omg this is literally the best day ever!! #blessed #livingmybestlife
    just posted a selfie and its trending lol. everyone follow me for more
    content!! dm me ur thoughts. this is so viral rn. no cap this is lit
    periodt. stan culture is wild. the vibes are immaculate.
    """ * 10
