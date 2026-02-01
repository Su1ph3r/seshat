"""
Advanced analysis modules for Seshat.
"""

from seshat.advanced.temporal import TemporalAnalyzer
from seshat.advanced.ai_detection import AIDetector
from seshat.advanced.nli import NativeLanguageIdentifier
from seshat.advanced.multi_author import MultiAuthorDetector
from seshat.advanced.adversarial import AdversarialDetector
from seshat.advanced.demographics import DemographicEstimator
from seshat.advanced.cross_platform import CrossPlatformAnalyzer

__all__ = [
    "TemporalAnalyzer",
    "AIDetector",
    "NativeLanguageIdentifier",
    "MultiAuthorDetector",
    "AdversarialDetector",
    "DemographicEstimator",
    "CrossPlatformAnalyzer",
]
