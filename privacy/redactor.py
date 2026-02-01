"""
PII redaction for Seshat.

Detects and redacts personally identifiable information.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class RedactionResult:
    """Result of redaction operation."""
    original_text: str
    redacted_text: str
    redactions: List[Dict[str, any]]
    pii_types_found: Set[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "original_length": len(self.original_text),
            "redacted_length": len(self.redacted_text),
            "redaction_count": len(self.redactions),
            "pii_types_found": list(self.pii_types_found),
            "redactions": self.redactions,
        }


class PIIRedactor:
    """
    Detect and redact personally identifiable information.
    """

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    PHONE_PATTERNS = [
        re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        re.compile(r"\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b"),
        re.compile(r"\b\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ]

    SSN_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")

    CREDIT_CARD_PATTERN = re.compile(
        r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"
    )

    IP_ADDRESS_PATTERN = re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )

    URL_PATTERN = re.compile(
        r"https?://[^\s<>\"]+|www\.[^\s<>\"]+"
    )

    USERNAME_PATTERN = re.compile(r"@[A-Za-z0-9_]+")

    DATE_PATTERNS = [
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        redact_emails: bool = True,
        redact_phones: bool = True,
        redact_ssn: bool = True,
        redact_credit_cards: bool = True,
        redact_ips: bool = True,
        redact_urls: bool = False,
        redact_usernames: bool = False,
        redact_dates: bool = False,
        custom_patterns: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize PII redactor.

        Args:
            redact_emails: Redact email addresses
            redact_phones: Redact phone numbers
            redact_ssn: Redact social security numbers
            redact_credit_cards: Redact credit card numbers
            redact_ips: Redact IP addresses
            redact_urls: Redact URLs
            redact_usernames: Redact @usernames
            redact_dates: Redact dates
            custom_patterns: Custom regex patterns to redact
        """
        self.redact_emails = redact_emails
        self.redact_phones = redact_phones
        self.redact_ssn = redact_ssn
        self.redact_credit_cards = redact_credit_cards
        self.redact_ips = redact_ips
        self.redact_urls = redact_urls
        self.redact_usernames = redact_usernames
        self.redact_dates = redact_dates
        self.custom_patterns = custom_patterns or {}

    def redact(self, text: str) -> RedactionResult:
        """
        Redact PII from text.

        Args:
            text: Text to redact

        Returns:
            RedactionResult with redacted text and details
        """
        redacted = text
        redactions = []
        pii_types = set()

        if self.redact_emails:
            redacted, found = self._redact_pattern(
                redacted, self.EMAIL_PATTERN, "[EMAIL]", "email"
            )
            redactions.extend(found)
            if found:
                pii_types.add("email")

        if self.redact_phones:
            for pattern in self.PHONE_PATTERNS:
                redacted, found = self._redact_pattern(
                    redacted, pattern, "[PHONE]", "phone"
                )
                redactions.extend(found)
                if found:
                    pii_types.add("phone")

        if self.redact_ssn:
            redacted, found = self._redact_pattern(
                redacted, self.SSN_PATTERN, "[SSN]", "ssn"
            )
            redactions.extend(found)
            if found:
                pii_types.add("ssn")

        if self.redact_credit_cards:
            redacted, found = self._redact_pattern(
                redacted, self.CREDIT_CARD_PATTERN, "[CREDIT_CARD]", "credit_card"
            )
            redactions.extend(found)
            if found:
                pii_types.add("credit_card")

        if self.redact_ips:
            redacted, found = self._redact_pattern(
                redacted, self.IP_ADDRESS_PATTERN, "[IP_ADDRESS]", "ip_address"
            )
            redactions.extend(found)
            if found:
                pii_types.add("ip_address")

        if self.redact_urls:
            redacted, found = self._redact_pattern(
                redacted, self.URL_PATTERN, "[URL]", "url"
            )
            redactions.extend(found)
            if found:
                pii_types.add("url")

        if self.redact_usernames:
            redacted, found = self._redact_pattern(
                redacted, self.USERNAME_PATTERN, "[USERNAME]", "username"
            )
            redactions.extend(found)
            if found:
                pii_types.add("username")

        if self.redact_dates:
            for pattern in self.DATE_PATTERNS:
                redacted, found = self._redact_pattern(
                    redacted, pattern, "[DATE]", "date"
                )
                redactions.extend(found)
                if found:
                    pii_types.add("date")

        for name, pattern_str in self.custom_patterns.items():
            pattern = re.compile(pattern_str)
            redacted, found = self._redact_pattern(
                redacted, pattern, f"[{name.upper()}]", name
            )
            redactions.extend(found)
            if found:
                pii_types.add(name)

        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            redactions=redactions,
            pii_types_found=pii_types,
        )

    def _redact_pattern(
        self,
        text: str,
        pattern: re.Pattern,
        replacement: str,
        pii_type: str,
    ) -> Tuple[str, List[Dict]]:
        """Redact matches of a pattern."""
        redactions = []

        def replacer(match):
            redactions.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "length": match.end() - match.start(),
            })
            return replacement

        redacted = pattern.sub(replacer, text)
        return redacted, redactions

    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII without redacting.

        Args:
            text: Text to scan

        Returns:
            Dictionary of PII type to list of found values
        """
        found = {}

        if self.redact_emails:
            emails = self.EMAIL_PATTERN.findall(text)
            if emails:
                found["emails"] = emails

        if self.redact_phones:
            phones = []
            for pattern in self.PHONE_PATTERNS:
                phones.extend(pattern.findall(text))
            if phones:
                found["phones"] = phones

        if self.redact_ssn:
            ssns = self.SSN_PATTERN.findall(text)
            if ssns:
                found["ssns"] = ssns

        if self.redact_credit_cards:
            cards = self.CREDIT_CARD_PATTERN.findall(text)
            if cards:
                found["credit_cards"] = cards

        if self.redact_ips:
            ips = self.IP_ADDRESS_PATTERN.findall(text)
            if ips:
                found["ip_addresses"] = ips

        return found

    def is_safe(self, text: str) -> bool:
        """
        Check if text contains any PII.

        Args:
            text: Text to check

        Returns:
            True if no PII detected
        """
        return len(self.detect(text)) == 0
