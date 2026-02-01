"""
Email importer for Seshat.

Imports email content from IMAP, MBOX, and EML files.
"""

from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import email
from email.policy import default as email_policy
import mailbox


@dataclass
class ImportedEmail:
    """Represents an imported email."""
    subject: str
    body: str
    sender: str
    recipients: List[str]
    date: Optional[datetime]
    message_id: str
    source: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "body": self.body,
            "sender": self.sender,
            "recipients": self.recipients,
            "date": self.date.isoformat() if self.date else None,
            "message_id": self.message_id,
            "source": self.source,
            "metadata": self.metadata,
        }


class EmailImporter:
    """
    Import emails from various sources.

    Supports IMAP servers, MBOX files, and individual EML files.
    """

    def __init__(self):
        """Initialize email importer."""
        pass

    def import_mbox(
        self,
        mbox_path: str,
        filter_sender: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Generator[ImportedEmail, None, None]:
        """
        Import emails from an MBOX file.

        Args:
            mbox_path: Path to MBOX file
            filter_sender: Only emails from this sender
            limit: Maximum emails to import

        Yields:
            ImportedEmail for each message
        """
        mbox = mailbox.mbox(mbox_path)
        count = 0

        for message in mbox:
            if limit and count >= limit:
                break

            try:
                imported = self._parse_message(message, f"mbox:{mbox_path}")

                if filter_sender:
                    if filter_sender.lower() not in imported.sender.lower():
                        continue

                yield imported
                count += 1

            except Exception:
                continue

    def import_maildir(
        self,
        maildir_path: str,
        filter_sender: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Generator[ImportedEmail, None, None]:
        """
        Import emails from a Maildir directory.

        Args:
            maildir_path: Path to Maildir
            filter_sender: Only emails from this sender
            limit: Maximum emails to import

        Yields:
            ImportedEmail for each message
        """
        md = mailbox.Maildir(maildir_path)
        count = 0

        for key in md.iterkeys():
            if limit and count >= limit:
                break

            try:
                message = md[key]
                imported = self._parse_message(message, f"maildir:{maildir_path}")

                if filter_sender:
                    if filter_sender.lower() not in imported.sender.lower():
                        continue

                yield imported
                count += 1

            except Exception:
                continue

    def import_eml(self, eml_path: str) -> Optional[ImportedEmail]:
        """
        Import a single EML file.

        Args:
            eml_path: Path to EML file

        Returns:
            ImportedEmail or None if failed
        """
        try:
            with open(eml_path, "rb") as f:
                message = email.message_from_binary_file(f, policy=email_policy)

            return self._parse_message(message, f"eml:{eml_path}")

        except Exception:
            return None

    def import_eml_directory(
        self,
        directory: str,
        recursive: bool = True,
        limit: Optional[int] = None,
    ) -> Generator[ImportedEmail, None, None]:
        """
        Import EML files from a directory.

        Args:
            directory: Directory containing EML files
            recursive: Search subdirectories
            limit: Maximum files to import

        Yields:
            ImportedEmail for each file
        """
        path = Path(directory)
        pattern = "**/*.eml" if recursive else "*.eml"
        count = 0

        for eml_file in path.glob(pattern):
            if limit and count >= limit:
                break

            imported = self.import_eml(str(eml_file))
            if imported:
                yield imported
                count += 1

    def import_imap(
        self,
        host: str,
        username: str,
        password: str,
        folder: str = "INBOX",
        filter_sender: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        use_ssl: bool = True,
    ) -> Generator[ImportedEmail, None, None]:
        """
        Import emails from an IMAP server.

        Args:
            host: IMAP server hostname
            username: Login username
            password: Login password
            folder: Mailbox folder to import from
            filter_sender: Only emails from this sender
            since: Only emails after this date
            limit: Maximum emails to import
            use_ssl: Use SSL connection

        Yields:
            ImportedEmail for each message
        """
        import imaplib

        if use_ssl:
            imap = imaplib.IMAP4_SSL(host)
        else:
            imap = imaplib.IMAP4(host)

        try:
            imap.login(username, password)
            imap.select(folder)

            search_criteria = "ALL"
            if since:
                date_str = since.strftime("%d-%b-%Y")
                search_criteria = f'(SINCE "{date_str}")'

            if filter_sender:
                if search_criteria == "ALL":
                    search_criteria = f'(FROM "{filter_sender}")'
                else:
                    search_criteria = f'({search_criteria} FROM "{filter_sender}")'

            _, message_numbers = imap.search(None, search_criteria)
            message_ids = message_numbers[0].split()

            if limit:
                message_ids = message_ids[-limit:]

            count = 0
            for msg_id in message_ids:
                if limit and count >= limit:
                    break

                try:
                    _, msg_data = imap.fetch(msg_id, "(RFC822)")

                    if msg_data and msg_data[0]:
                        email_body = msg_data[0][1]
                        message = email.message_from_bytes(email_body, policy=email_policy)

                        imported = self._parse_message(message, f"imap:{host}/{folder}")

                        yield imported
                        count += 1

                except Exception:
                    continue

        finally:
            try:
                imap.close()
                imap.logout()
            except Exception:
                pass

    def _parse_message(
        self,
        message: email.message.EmailMessage,
        source: str,
    ) -> ImportedEmail:
        """Parse an email message into ImportedEmail."""
        subject = message.get("Subject", "")
        sender = message.get("From", "")
        recipients = []

        for header in ["To", "Cc"]:
            value = message.get(header)
            if value:
                recipients.extend([addr.strip() for addr in value.split(",")])

        message_id = message.get("Message-ID", "")

        date = None
        date_header = message.get("Date")
        if date_header:
            try:
                date = email.utils.parsedate_to_datetime(date_header)
            except Exception:
                pass

        body = self._extract_body(message)

        metadata = {
            "content_type": message.get_content_type(),
            "has_attachments": any(
                part.get_content_disposition() == "attachment"
                for part in message.walk()
            ),
        }

        in_reply_to = message.get("In-Reply-To")
        if in_reply_to:
            metadata["in_reply_to"] = in_reply_to

        return ImportedEmail(
            subject=subject,
            body=body,
            sender=sender,
            recipients=recipients,
            date=date,
            message_id=message_id,
            source=source,
            metadata=metadata,
        )

    def _extract_body(self, message: email.message.EmailMessage) -> str:
        """Extract text body from email message."""
        body_parts = []

        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                disposition = part.get_content_disposition()

                if disposition == "attachment":
                    continue

                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            body_parts.append(payload.decode(charset, errors="ignore"))
                        except Exception:
                            body_parts.append(payload.decode("utf-8", errors="ignore"))

        else:
            payload = message.get_payload(decode=True)
            if payload:
                charset = message.get_content_charset() or "utf-8"
                try:
                    body_parts.append(payload.decode(charset, errors="ignore"))
                except Exception:
                    body_parts.append(payload.decode("utf-8", errors="ignore"))

        return "\n\n".join(body_parts)

    def get_combined_text(
        self,
        emails: List[ImportedEmail],
        include_subject: bool = True,
    ) -> str:
        """
        Combine emails into a single text.

        Args:
            emails: List of imported emails
            include_subject: Include subject lines

        Returns:
            Combined text
        """
        parts = []

        for email_msg in emails:
            if include_subject and email_msg.subject:
                parts.append(email_msg.subject)
            if email_msg.body:
                parts.append(email_msg.body)

        return "\n\n".join(parts)
