"""
Machine learning modules for Seshat.
"""

from seshat.ml.classifier import AuthorshipClassifier
from seshat.ml.ensemble import EnsembleClassifier
from seshat.ml.embeddings import TextEmbedder

__all__ = [
    "AuthorshipClassifier",
    "EnsembleClassifier",
    "TextEmbedder",
]
