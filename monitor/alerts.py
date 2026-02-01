"""
Alert system for Seshat monitoring.

Sends notifications when significant events occur.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    STYLE_DRIFT = "style_drift"
    NEW_CONTENT = "new_content"
    MATCH_FOUND = "match_found"
    SCRAPE_ERROR = "scrape_error"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class Alert:
    """Represents an alert."""
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


class AlertManager:
    """
    Manage and send alerts.
    """

    def __init__(self):
        """Initialize alert manager."""
        self._alerts: List[Alert] = []
        self._handlers: Dict[str, Callable[[Alert], None]] = {}
        self._max_alerts = 1000

    def add_handler(
        self,
        name: str,
        handler: Callable[[Alert], None],
    ):
        """
        Add an alert handler.

        Args:
            name: Handler name
            handler: Callable that receives Alert
        """
        self._handlers[name] = handler

    def remove_handler(self, name: str):
        """Remove an alert handler."""
        self._handlers.pop(name, None)

    def send(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """
        Send an alert.

        Args:
            alert_type: Type of alert
            level: Alert severity
            title: Alert title
            message: Alert message
            data: Additional data

        Returns:
            Created Alert
        """
        alert = Alert(
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            data=data or {},
        )

        self._alerts.append(alert)

        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        for handler in self._handlers.values():
            try:
                handler(alert)
            except Exception:
                pass

        return alert

    def send_style_drift(
        self,
        profile_name: str,
        drift_score: float,
        threshold: float,
    ) -> Alert:
        """Send style drift alert."""
        level = AlertLevel.CRITICAL if drift_score > 0.3 else AlertLevel.WARNING

        return self.send(
            alert_type=AlertType.STYLE_DRIFT,
            level=level,
            title=f"Style Drift Detected: {profile_name}",
            message=f"Drift score {drift_score:.2f} exceeds threshold {threshold:.2f}",
            data={
                "profile_name": profile_name,
                "drift_score": drift_score,
                "threshold": threshold,
            },
        )

    def send_new_content(
        self,
        platform: str,
        username: str,
        sample_count: int,
    ) -> Alert:
        """Send new content alert."""
        return self.send(
            alert_type=AlertType.NEW_CONTENT,
            level=AlertLevel.INFO,
            title=f"New Content: {username}",
            message=f"Collected {sample_count} new samples from {platform}",
            data={
                "platform": platform,
                "username": username,
                "sample_count": sample_count,
            },
        )

    def send_match_found(
        self,
        profile_name: str,
        score: float,
        confidence: str,
    ) -> Alert:
        """Send authorship match alert."""
        level = AlertLevel.CRITICAL if confidence == "high" else AlertLevel.WARNING

        return self.send(
            alert_type=AlertType.MATCH_FOUND,
            level=level,
            title=f"Authorship Match: {profile_name}",
            message=f"Match score {score:.2%} with {confidence} confidence",
            data={
                "profile_name": profile_name,
                "score": score,
                "confidence": confidence,
            },
        )

    def send_error(
        self,
        operation: str,
        error: str,
    ) -> Alert:
        """Send error alert."""
        return self.send(
            alert_type=AlertType.SCRAPE_ERROR,
            level=AlertLevel.WARNING,
            title=f"Error: {operation}",
            message=error,
            data={"operation": operation, "error": error},
        )

    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        level: Optional[AlertLevel] = None,
        since: Optional[datetime] = None,
        unacknowledged_only: bool = False,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Get alerts with optional filtering.

        Args:
            alert_type: Filter by type
            level: Filter by level
            since: Only alerts after this time
            unacknowledged_only: Only unacknowledged alerts
            limit: Maximum alerts to return

        Returns:
            List of matching alerts
        """
        alerts = self._alerts.copy()

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if level:
            alerts = [a for a in alerts if a.level == level]

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts[:limit]

    def acknowledge(self, alert: Alert):
        """Acknowledge an alert."""
        alert.acknowledged = True
        alert.acknowledged_at = datetime.now()

    def acknowledge_all(self):
        """Acknowledge all alerts."""
        for alert in self._alerts:
            if not alert.acknowledged:
                self.acknowledge(alert)

    def clear(self):
        """Clear all alerts."""
        self._alerts.clear()


def webhook_handler(webhook_url: str) -> Callable[[Alert], None]:
    """
    Create a webhook alert handler.

    Args:
        webhook_url: URL to POST alerts to

    Returns:
        Handler function
    """
    def handler(alert: Alert):
        import httpx

        try:
            httpx.post(
                webhook_url,
                json=alert.to_dict(),
                timeout=10.0,
            )
        except Exception:
            pass

    return handler


def email_handler(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    from_addr: str,
    to_addrs: List[str],
) -> Callable[[Alert], None]:
    """
    Create an email alert handler.

    Args:
        smtp_host: SMTP server hostname
        smtp_port: SMTP server port
        username: SMTP username
        password: SMTP password
        from_addr: From email address
        to_addrs: List of recipient addresses

    Returns:
        Handler function
    """
    def handler(alert: Alert):
        import smtplib
        from email.mime.text import MIMEText

        try:
            msg = MIMEText(f"{alert.message}\n\nData: {json.dumps(alert.data, indent=2)}")
            msg["Subject"] = f"[Seshat {alert.level.value.upper()}] {alert.title}"
            msg["From"] = from_addr
            msg["To"] = ", ".join(to_addrs)

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())

        except Exception:
            pass

    return handler
