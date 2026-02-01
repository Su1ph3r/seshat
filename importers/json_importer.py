"""
JSON importer for Seshat.

Imports text data from JSON and JSON Lines files.
"""

from typing import Dict, List, Any, Optional, Generator, Union
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ImportedJSONRecord:
    """Represents an imported JSON record."""
    text: str
    record_index: int
    source_file: str
    author: Optional[str]
    timestamp: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "record_index": self.record_index,
            "source_file": self.source_file,
            "author": self.author,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class JSONImporter:
    """
    Import text data from JSON files.

    Supports JSON arrays, JSON objects, and JSON Lines format.
    """

    def __init__(
        self,
        text_field: str = "text",
        author_field: Optional[str] = None,
        timestamp_field: Optional[str] = None,
        records_path: Optional[str] = None,
    ):
        """
        Initialize JSON importer.

        Args:
            text_field: JSON field containing text (supports dot notation)
            author_field: Optional field for author
            timestamp_field: Optional field for timestamp
            records_path: JSON path to array of records (e.g., "data.items")
        """
        self.text_field = text_field
        self.author_field = author_field
        self.timestamp_field = timestamp_field
        self.records_path = records_path

    def import_file(
        self,
        file_path: str,
        min_text_length: int = 10,
        limit: Optional[int] = None,
    ) -> Generator[ImportedJSONRecord, None, None]:
        """
        Import records from a JSON file.

        Args:
            file_path: Path to JSON file
            min_text_length: Minimum text length to include
            limit: Maximum records to import

        Yields:
            ImportedJSONRecord for each valid record
        """
        path = Path(file_path)

        if path.suffix.lower() == ".jsonl":
            yield from self._import_jsonl(path, min_text_length, limit)
        else:
            yield from self._import_json(path, min_text_length, limit)

    def _import_json(
        self,
        path: Path,
        min_text_length: int,
        limit: Optional[int],
    ) -> Generator[ImportedJSONRecord, None, None]:
        """Import from regular JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = self._extract_records(data)
        count = 0

        for idx, record in enumerate(records):
            if limit and count >= limit:
                break

            imported = self._parse_record(record, idx, str(path))

            if imported and len(imported.text.strip()) >= min_text_length:
                yield imported
                count += 1

    def _import_jsonl(
        self,
        path: Path,
        min_text_length: int,
        limit: Optional[int],
    ) -> Generator[ImportedJSONRecord, None, None]:
        """Import from JSON Lines file."""
        count = 0

        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit and count >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    imported = self._parse_record(record, idx, str(path))

                    if imported and len(imported.text.strip()) >= min_text_length:
                        yield imported
                        count += 1

                except json.JSONDecodeError:
                    continue

    def import_directory(
        self,
        directory: str,
        pattern: str = "*.json",
        recursive: bool = True,
        min_text_length: int = 10,
        limit: Optional[int] = None,
    ) -> Generator[ImportedJSONRecord, None, None]:
        """
        Import JSON files from a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for JSON files
            recursive: Search subdirectories
            min_text_length: Minimum text length
            limit: Maximum total records

        Yields:
            ImportedJSONRecord for each valid record
        """
        path = Path(directory)
        count = 0

        patterns = [pattern]
        if "*.json" in pattern:
            patterns.append(pattern.replace("*.json", "*.jsonl"))

        for pat in patterns:
            if limit and count >= limit:
                break

            glob_pattern = f"**/{pat}" if recursive else pat

            for json_file in path.glob(glob_pattern):
                if limit and count >= limit:
                    break

                remaining = limit - count if limit else None

                for record in self.import_file(
                    str(json_file),
                    min_text_length=min_text_length,
                    limit=remaining,
                ):
                    yield record
                    count += 1

                    if limit and count >= limit:
                        break

    def _extract_records(self, data: Any) -> List[Dict[str, Any]]:
        """Extract records array from JSON data."""
        if self.records_path:
            parts = self.records_path.split(".")
            current = data

            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part, [])
                elif isinstance(current, list) and part.isdigit():
                    idx = int(part)
                    current = current[idx] if idx < len(current) else []
                else:
                    return []

            if isinstance(current, list):
                return current
            elif isinstance(current, dict):
                return [current]
            else:
                return []

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []

    def _parse_record(
        self,
        record: Dict[str, Any],
        index: int,
        source_file: str,
    ) -> Optional[ImportedJSONRecord]:
        """Parse a JSON record into ImportedJSONRecord."""
        if not isinstance(record, dict):
            return None

        text = self._get_nested_field(record, self.text_field)
        if not text:
            return None

        author = None
        if self.author_field:
            author = self._get_nested_field(record, self.author_field)

        timestamp = None
        if self.timestamp_field:
            timestamp = self._get_nested_field(record, self.timestamp_field)

        metadata = {}
        for key, value in record.items():
            if key != self.text_field:
                metadata[key] = value

        return ImportedJSONRecord(
            text=str(text).strip(),
            record_index=index,
            source_file=source_file,
            author=str(author) if author else None,
            timestamp=str(timestamp) if timestamp else None,
            metadata=metadata,
        )

    def _get_nested_field(
        self,
        data: Dict[str, Any],
        field_path: str,
    ) -> Any:
        """Get a field from nested dictionary using dot notation."""
        parts = field_path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            else:
                return None

            if current is None:
                return None

        return current

    def preview(
        self,
        file_path: str,
        num_records: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Preview JSON file contents.

        Args:
            file_path: Path to JSON file
            num_records: Number of records to preview

        Returns:
            List of record dictionaries
        """
        records = []

        for record in self.import_file(file_path, min_text_length=0, limit=num_records):
            records.append(record.to_dict())

        return records

    def get_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze JSON file structure.

        Args:
            file_path: Path to JSON file

        Returns:
            Structure information
        """
        path = Path(file_path)

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() == ".jsonl":
                first_line = f.readline().strip()
                if first_line:
                    sample = json.loads(first_line)
                    return {
                        "format": "jsonl",
                        "fields": list(sample.keys()) if isinstance(sample, dict) else [],
                        "sample": sample,
                    }
                return {"format": "jsonl", "fields": [], "sample": None}
            else:
                data = json.load(f)

                if isinstance(data, list):
                    sample = data[0] if data else {}
                    return {
                        "format": "json_array",
                        "record_count": len(data),
                        "fields": list(sample.keys()) if isinstance(sample, dict) else [],
                        "sample": sample,
                    }
                elif isinstance(data, dict):
                    return {
                        "format": "json_object",
                        "fields": list(data.keys()),
                        "nested_arrays": [
                            k for k, v in data.items()
                            if isinstance(v, list)
                        ],
                        "sample": data,
                    }
                else:
                    return {"format": "unknown", "fields": [], "sample": data}
