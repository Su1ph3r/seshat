"""
SQLAlchemy models for Seshat database.

Defines the database schema for profiles, samples, and analyses.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    LargeBinary,
    JSON,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()


class Author(Base):
    """
    Author/Profile model.

    Stores author profiles with aggregated stylometric features.
    """

    __tablename__ = "authors"

    id = Column(Integer, primary_key=True)
    profile_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)
    aggregated_features = Column(JSON, default=dict)
    feature_stats = Column(JSON, default=dict)
    total_words = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    samples = relationship("Sample", back_populates="author", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="author", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_author_name_active", "name", "is_active"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata_json,
            "aggregated_features": self.aggregated_features,
            "feature_stats": self.feature_stats,
            "total_words": self.total_words,
            "sample_count": len(self.samples) if self.samples else 0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get profile summary."""
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "sample_count": len(self.samples) if self.samples else 0,
            "total_words": self.total_words,
            "feature_count": len(self.aggregated_features) if self.aggregated_features else 0,
        }


class Sample(Base):
    """
    Writing sample model.

    Stores individual text samples associated with an author.
    """

    __tablename__ = "samples"

    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, ForeignKey("authors.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    text_hash = Column(String(64), nullable=False, index=True)
    source_url = Column(String(1024))
    source_platform = Column(String(64), index=True)
    original_timestamp = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    word_count = Column(Integer, default=0)
    features = Column(JSON, default=dict)
    metadata_json = Column(JSON, default=dict)

    author = relationship("Author", back_populates="samples")

    __table_args__ = (
        UniqueConstraint("author_id", "text_hash", name="uq_author_sample"),
        Index("idx_sample_platform", "source_platform"),
        Index("idx_sample_timestamp", "original_timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "author_id": self.author_id,
            "text": self.text,
            "text_hash": self.text_hash,
            "source_url": self.source_url,
            "source_platform": self.source_platform,
            "original_timestamp": self.original_timestamp.isoformat() if self.original_timestamp else None,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
            "word_count": self.word_count,
            "features": self.features,
            "metadata": self.metadata_json,
        }


class Analysis(Base):
    """
    Analysis result model.

    Stores analysis results for text comparisons.
    """

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, ForeignKey("authors.id"), index=True)
    analyzed_text_hash = Column(String(64), nullable=False, index=True)
    analyzed_text_preview = Column(String(500))
    confidence_score = Column(Float)
    overall_score = Column(Float)
    confidence_level = Column(String(32))
    feature_breakdown = Column(JSON, default=dict)
    methodology_version = Column(String(32), default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    author = relationship("Author", back_populates="analyses")

    __table_args__ = (
        Index("idx_analysis_author_date", "author_id", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "author_id": self.author_id,
            "analyzed_text_hash": self.analyzed_text_hash,
            "analyzed_text_preview": self.analyzed_text_preview,
            "confidence_score": self.confidence_score,
            "overall_score": self.overall_score,
            "confidence_level": self.confidence_level,
            "feature_breakdown": self.feature_breakdown,
            "methodology_version": self.methodology_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLog(Base):
    """
    Audit log model.

    Tracks all actions for forensic purposes.
    """

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    action = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(64), nullable=False)
    entity_id = Column(String(64))
    user_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    details = Column(JSON, default=dict)
    ip_address = Column(String(45))
    checksum = Column(String(64))

    __table_args__ = (
        Index("idx_audit_action_time", "action", "timestamp"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details": self.details,
            "ip_address": self.ip_address,
            "checksum": self.checksum,
        }
