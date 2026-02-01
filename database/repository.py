"""
Repository for database operations.

Provides data access layer for Seshat database.
"""

from typing import Dict, List, Any, Optional, Type, TypeVar
from datetime import datetime
import hashlib
import os

from sqlalchemy import create_engine, func, or_
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, Author, Sample, Analysis, AuditLog


T = TypeVar("T", bound=Base)


class Repository:
    """
    Data access repository for Seshat.

    Provides CRUD operations and queries for all models.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
    ):
        """
        Initialize repository.

        Args:
            database_url: Database connection URL (defaults to SQLite)
            echo: Echo SQL statements for debugging
        """
        if database_url is None:
            db_path = os.path.expanduser("~/.seshat/seshat.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f"sqlite:///{db_path}"

        self.engine = create_engine(database_url, echo=echo)
        self.SessionLocal = sessionmaker(bind=self.engine)

        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def create_author(
        self,
        name: str,
        profile_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> Author:
        """
        Create a new author profile.

        Args:
            name: Author name
            profile_id: Unique profile ID
            metadata: Optional metadata
            session: Optional existing session

        Returns:
            Created Author object
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            author = Author(
                name=name,
                profile_id=profile_id,
                metadata_json=metadata or {},
            )
            session.add(author)
            session.commit()
            session.refresh(author)

            self._log_action(
                session, "create", "author", profile_id,
                {"name": name},
            )

            return author

        finally:
            if close_session:
                session.close()

    def get_author(
        self,
        name: Optional[str] = None,
        profile_id: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Optional[Author]:
        """
        Get an author by name or profile_id.

        Args:
            name: Author name
            profile_id: Profile ID
            session: Optional existing session

        Returns:
            Author object or None
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            query = session.query(Author).filter(Author.is_active == True)

            if profile_id:
                query = query.filter(Author.profile_id == profile_id)
            elif name:
                query = query.filter(Author.name == name)
            else:
                return None

            return query.first()

        finally:
            if close_session:
                session.close()

    def list_authors(
        self,
        session: Optional[Session] = None,
    ) -> List[Author]:
        """
        List all active authors.

        Args:
            session: Optional existing session

        Returns:
            List of Author objects
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            return session.query(Author).filter(
                Author.is_active == True
            ).order_by(Author.name).all()

        finally:
            if close_session:
                session.close()

    def update_author(
        self,
        author: Author,
        aggregated_features: Optional[Dict[str, float]] = None,
        feature_stats: Optional[Dict[str, Any]] = None,
        total_words: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> Author:
        """
        Update an author profile.

        Args:
            author: Author to update
            aggregated_features: New aggregated features
            feature_stats: New feature statistics
            total_words: New total word count
            metadata: New metadata (merged with existing)
            session: Optional existing session

        Returns:
            Updated Author object
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            if aggregated_features is not None:
                author.aggregated_features = aggregated_features

            if feature_stats is not None:
                author.feature_stats = feature_stats

            if total_words is not None:
                author.total_words = total_words

            if metadata is not None:
                existing = author.metadata_json or {}
                existing.update(metadata)
                author.metadata_json = existing

            author.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(author)

            self._log_action(
                session, "update", "author", author.profile_id,
                {"updated_fields": list(filter(None, [
                    "aggregated_features" if aggregated_features else None,
                    "feature_stats" if feature_stats else None,
                    "total_words" if total_words else None,
                    "metadata" if metadata else None,
                ]))},
            )

            return author

        finally:
            if close_session:
                session.close()

    def delete_author(
        self,
        author: Author,
        hard_delete: bool = False,
        session: Optional[Session] = None,
    ) -> bool:
        """
        Delete an author profile.

        Args:
            author: Author to delete
            hard_delete: Permanently delete (vs soft delete)
            session: Optional existing session

        Returns:
            True if deleted
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            if hard_delete:
                session.delete(author)
            else:
                author.is_active = False
                author.updated_at = datetime.utcnow()

            session.commit()

            self._log_action(
                session, "delete", "author", author.profile_id,
                {"hard_delete": hard_delete},
            )

            return True

        finally:
            if close_session:
                session.close()

    def add_sample(
        self,
        author: Author,
        text: str,
        source_url: Optional[str] = None,
        source_platform: Optional[str] = None,
        original_timestamp: Optional[datetime] = None,
        features: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> Optional[Sample]:
        """
        Add a sample to an author.

        Args:
            author: Author to add sample to
            text: Sample text
            source_url: Source URL
            source_platform: Platform name
            original_timestamp: Original creation time
            features: Extracted features
            metadata: Additional metadata
            session: Optional existing session

        Returns:
            Created Sample or None if duplicate
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            existing = session.query(Sample).filter(
                Sample.author_id == author.id,
                Sample.text_hash == text_hash,
            ).first()

            if existing:
                return None

            word_count = len(text.split())

            sample = Sample(
                author_id=author.id,
                text=text,
                text_hash=text_hash,
                source_url=source_url,
                source_platform=source_platform,
                original_timestamp=original_timestamp,
                word_count=word_count,
                features=features or {},
                metadata_json=metadata or {},
            )

            session.add(sample)
            session.commit()
            session.refresh(sample)

            self._log_action(
                session, "add_sample", "sample", str(sample.id),
                {"author_id": author.profile_id, "word_count": word_count},
            )

            return sample

        finally:
            if close_session:
                session.close()

    def get_samples(
        self,
        author: Author,
        limit: Optional[int] = None,
        offset: int = 0,
        session: Optional[Session] = None,
    ) -> List[Sample]:
        """
        Get samples for an author.

        Args:
            author: Author to get samples for
            limit: Maximum samples to return
            offset: Offset for pagination
            session: Optional existing session

        Returns:
            List of Sample objects
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            query = session.query(Sample).filter(
                Sample.author_id == author.id
            ).order_by(Sample.scraped_at.desc())

            if offset:
                query = query.offset(offset)

            if limit:
                query = query.limit(limit)

            return query.all()

        finally:
            if close_session:
                session.close()

    def save_analysis(
        self,
        author: Author,
        analyzed_text: str,
        overall_score: float,
        confidence_score: float,
        confidence_level: str,
        feature_breakdown: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> Analysis:
        """
        Save an analysis result.

        Args:
            author: Matched author
            analyzed_text: Text that was analyzed
            overall_score: Overall similarity score
            confidence_score: Confidence score
            confidence_level: Confidence level string
            feature_breakdown: Detailed feature comparison
            session: Optional existing session

        Returns:
            Created Analysis object
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            text_hash = hashlib.sha256(analyzed_text.encode("utf-8")).hexdigest()
            preview = analyzed_text[:500] if len(analyzed_text) > 500 else analyzed_text

            analysis = Analysis(
                author_id=author.id,
                analyzed_text_hash=text_hash,
                analyzed_text_preview=preview,
                overall_score=overall_score,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                feature_breakdown=feature_breakdown or {},
            )

            session.add(analysis)
            session.commit()
            session.refresh(analysis)

            return analysis

        finally:
            if close_session:
                session.close()

    def search_text(
        self,
        query: str,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[Sample]:
        """
        Full-text search in samples.

        Args:
            query: Search query
            limit: Maximum results
            session: Optional existing session

        Returns:
            List of matching Sample objects
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            return session.query(Sample).filter(
                Sample.text.contains(query)
            ).limit(limit).all()

        finally:
            if close_session:
                session.close()

    def get_statistics(
        self,
        session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Get database statistics.

        Args:
            session: Optional existing session

        Returns:
            Statistics dictionary
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            author_count = session.query(func.count(Author.id)).filter(
                Author.is_active == True
            ).scalar()

            sample_count = session.query(func.count(Sample.id)).scalar()

            total_words = session.query(func.sum(Sample.word_count)).scalar() or 0

            analysis_count = session.query(func.count(Analysis.id)).scalar()

            return {
                "author_count": author_count,
                "sample_count": sample_count,
                "total_words": total_words,
                "analysis_count": analysis_count,
            }

        finally:
            if close_session:
                session.close()

    def _log_action(
        self,
        session: Session,
        action: str,
        entity_type: str,
        entity_id: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """Log an action to the audit log."""
        log_entry = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            details=details or {},
            ip_address=ip_address,
        )

        content = f"{action}|{entity_type}|{entity_id}|{log_entry.timestamp}"
        log_entry.checksum = hashlib.sha256(content.encode()).hexdigest()

        session.add(log_entry)
        session.commit()

    def get_audit_log(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[AuditLog]:
        """
        Query audit log.

        Args:
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            action: Filter by action
            since: Only entries after this time
            limit: Maximum entries
            session: Optional existing session

        Returns:
            List of AuditLog entries
        """
        close_session = session is None
        session = session or self.get_session()

        try:
            query = session.query(AuditLog)

            if entity_type:
                query = query.filter(AuditLog.entity_type == entity_type)

            if entity_id:
                query = query.filter(AuditLog.entity_id == entity_id)

            if action:
                query = query.filter(AuditLog.action == action)

            if since:
                query = query.filter(AuditLog.timestamp >= since)

            return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()

        finally:
            if close_session:
                session.close()
