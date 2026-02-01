"""
CSV importer for Seshat.

Imports text data from CSV files.
"""

from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass
class ImportedRow:
    """Represents an imported CSV row."""
    text: str
    row_number: int
    source_file: str
    author: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "row_number": self.row_number,
            "source_file": self.source_file,
            "author": self.author,
            "metadata": self.metadata,
        }


class CSVImporter:
    """
    Import text data from CSV files.

    Configurable column mapping for text, author, and metadata.
    """

    def __init__(
        self,
        text_column: str = "text",
        author_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        skip_header: bool = True,
    ):
        """
        Initialize CSV importer.

        Args:
            text_column: Column name or index for text
            author_column: Optional column for author
            metadata_columns: Optional columns to include in metadata
            delimiter: CSV delimiter
            encoding: File encoding
            skip_header: Skip header row
        """
        self.text_column = text_column
        self.author_column = author_column
        self.metadata_columns = metadata_columns or []
        self.delimiter = delimiter
        self.encoding = encoding
        self.skip_header = skip_header

    def import_file(
        self,
        file_path: str,
        min_text_length: int = 10,
        limit: Optional[int] = None,
    ) -> Generator[ImportedRow, None, None]:
        """
        Import rows from a CSV file.

        Args:
            file_path: Path to CSV file
            min_text_length: Minimum text length to include
            limit: Maximum rows to import

        Yields:
            ImportedRow for each valid row
        """
        path = Path(file_path)
        count = 0

        with open(path, "r", encoding=self.encoding, newline="") as f:
            if self._is_column_name(self.text_column):
                reader = csv.DictReader(f, delimiter=self.delimiter)
            else:
                reader = csv.reader(f, delimiter=self.delimiter)

                if self.skip_header:
                    next(reader, None)

            for row_num, row in enumerate(reader, start=1):
                if limit and count >= limit:
                    break

                try:
                    imported = self._parse_row(row, row_num, str(path))

                    if imported and len(imported.text.strip()) >= min_text_length:
                        yield imported
                        count += 1

                except Exception:
                    continue

    def import_directory(
        self,
        directory: str,
        pattern: str = "*.csv",
        recursive: bool = True,
        min_text_length: int = 10,
        limit: Optional[int] = None,
    ) -> Generator[ImportedRow, None, None]:
        """
        Import CSV files from a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for CSV files
            recursive: Search subdirectories
            min_text_length: Minimum text length
            limit: Maximum total rows

        Yields:
            ImportedRow for each valid row
        """
        path = Path(directory)
        glob_pattern = f"**/{pattern}" if recursive else pattern
        count = 0

        for csv_file in path.glob(glob_pattern):
            if limit and count >= limit:
                break

            remaining = limit - count if limit else None

            for row in self.import_file(
                str(csv_file),
                min_text_length=min_text_length,
                limit=remaining,
            ):
                yield row
                count += 1

                if limit and count >= limit:
                    break

    def _is_column_name(self, column: str) -> bool:
        """Check if column is a name (string) or index (int)."""
        try:
            int(column)
            return False
        except ValueError:
            return True

    def _parse_row(
        self,
        row: Any,
        row_num: int,
        source_file: str,
    ) -> Optional[ImportedRow]:
        """Parse a CSV row into ImportedRow."""
        if isinstance(row, dict):
            return self._parse_dict_row(row, row_num, source_file)
        else:
            return self._parse_list_row(row, row_num, source_file)

    def _parse_dict_row(
        self,
        row: Dict[str, Any],
        row_num: int,
        source_file: str,
    ) -> Optional[ImportedRow]:
        """Parse a dictionary row (from DictReader)."""
        text = row.get(self.text_column, "")

        if not text or not text.strip():
            return None

        author = None
        if self.author_column:
            author = row.get(self.author_column)

        metadata = {}
        for col in self.metadata_columns:
            if col in row:
                metadata[col] = row[col]

        for key, value in row.items():
            if key not in (self.text_column, self.author_column):
                if key not in metadata:
                    metadata[key] = value

        return ImportedRow(
            text=str(text).strip(),
            row_number=row_num,
            source_file=source_file,
            author=str(author).strip() if author else None,
            metadata=metadata,
        )

    def _parse_list_row(
        self,
        row: List[Any],
        row_num: int,
        source_file: str,
    ) -> Optional[ImportedRow]:
        """Parse a list row (from regular reader)."""
        try:
            text_idx = int(self.text_column)
            text = row[text_idx] if text_idx < len(row) else ""
        except (ValueError, IndexError):
            return None

        if not text or not text.strip():
            return None

        author = None
        if self.author_column:
            try:
                author_idx = int(self.author_column)
                author = row[author_idx] if author_idx < len(row) else None
            except (ValueError, IndexError):
                pass

        metadata = {"columns": row}

        return ImportedRow(
            text=str(text).strip(),
            row_number=row_num,
            source_file=source_file,
            author=str(author).strip() if author else None,
            metadata=metadata,
        )

    def get_column_names(self, file_path: str) -> List[str]:
        """
        Get column names from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of column names
        """
        with open(file_path, "r", encoding=self.encoding, newline="") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            header = next(reader, None)
            return header if header else []

    def preview(
        self,
        file_path: str,
        num_rows: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Preview CSV file contents.

        Args:
            file_path: Path to CSV file
            num_rows: Number of rows to preview

        Returns:
            List of row dictionaries
        """
        rows = []

        with open(file_path, "r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)

            for i, row in enumerate(reader):
                if i >= num_rows:
                    break
                rows.append(dict(row))

        return rows
