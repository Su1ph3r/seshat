"""
Seshat - Stylometric Authorship Attribution & Psychological Profiling Tool

Named after the Egyptian goddess of writing, wisdom, and measurement - "She who scribes"
"""

__version__ = "0.1.0"
__author__ = "Seshat Contributors"

from seshat.analyzer import Analyzer
from seshat.profile import AuthorProfile
from seshat.comparator import Comparator

__all__ = [
    "Analyzer",
    "AuthorProfile",
    "Comparator",
    "__version__",
]
