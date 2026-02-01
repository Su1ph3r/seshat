"""
File readers for various input formats.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass


@dataclass
class TextSample:
    """Container for a text sample with metadata."""
    text: str
    source: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TextReader:
    """
    Universal text reader supporting multiple formats.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".json": "json",
        ".jsonl": "jsonl",
        ".csv": "csv",
        ".md": "text",
        ".html": "html",
        ".htm": "html",
    }

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize reader.

        Args:
            encoding: Default text encoding
        """
        self.encoding = encoding

    def read(self, path: str) -> List[TextSample]:
        """
        Read text samples from a file.

        Args:
            path: Path to file

        Returns:
            List of TextSample objects
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(extension, "text")

        if file_type == "text":
            return self._read_text(path)
        elif file_type == "json":
            return self._read_json(path)
        elif file_type == "jsonl":
            return self._read_jsonl(path)
        elif file_type == "csv":
            return self._read_csv(path)
        elif file_type == "html":
            return self._read_html(path)
        else:
            return self._read_text(path)

    def read_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[TextSample]:
        """
        Read all text files from a directory.

        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            extensions: List of extensions to include (default: all supported)

        Returns:
            List of TextSample objects
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        if extensions is None:
            extensions = list(self.SUPPORTED_EXTENSIONS.keys())

        samples = []
        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            for path in directory.glob(f"{pattern}{ext}"):
                try:
                    file_samples = self.read(str(path))
                    for sample in file_samples:
                        if sample.source is None:
                            sample.source = str(path)
                        samples.append(sample)
                except Exception:
                    continue

        return samples

    def _read_text(self, path: Path) -> List[TextSample]:
        """Read plain text file."""
        with open(path, "r", encoding=self.encoding) as f:
            text = f.read()

        return [TextSample(text=text, source=str(path))]

    def _read_json(self, path: Path) -> List[TextSample]:
        """Read JSON file (array of samples or single object)."""
        with open(path, "r", encoding=self.encoding) as f:
            data = json.load(f)

        if isinstance(data, list):
            return [self._parse_sample(item, str(path)) for item in data]
        elif isinstance(data, dict):
            return [self._parse_sample(data, str(path))]
        else:
            return [TextSample(text=str(data), source=str(path))]

    def _read_jsonl(self, path: Path) -> List[TextSample]:
        """Read JSON Lines file (one JSON object per line)."""
        samples = []

        with open(path, "r", encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        samples.append(self._parse_sample(data, str(path)))
                    except json.JSONDecodeError:
                        continue

        return samples

    def _read_csv(self, path: Path) -> List[TextSample]:
        """Read CSV file."""
        samples = []

        with open(path, "r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f)

            text_columns = ["text", "content", "body", "message", "post"]
            text_column = None

            if reader.fieldnames:
                for col in text_columns:
                    if col in reader.fieldnames:
                        text_column = col
                        break

                if text_column is None and reader.fieldnames:
                    text_column = reader.fieldnames[0]

            for row in reader:
                if text_column and row.get(text_column):
                    sample = TextSample(
                        text=row[text_column],
                        source=str(path),
                        author=row.get("author") or row.get("username") or row.get("user"),
                        timestamp=row.get("timestamp") or row.get("date") or row.get("created_at"),
                        metadata={k: v for k, v in row.items() if k != text_column},
                    )
                    samples.append(sample)

        return samples

    def _read_html(self, path: Path) -> List[TextSample]:
        """Read HTML file and extract text content."""
        try:
            from bs4 import BeautifulSoup

            with open(path, "r", encoding=self.encoding) as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n", strip=True)

            return [TextSample(text=text, source=str(path))]

        except ImportError:
            with open(path, "r", encoding=self.encoding) as f:
                text = f.read()
            import re
            text = re.sub(r'<[^>]+>', '', text)
            return [TextSample(text=text, source=str(path))]

    def _parse_sample(self, data: Dict[str, Any], source: str) -> TextSample:
        """Parse a dictionary into a TextSample."""
        text_keys = ["text", "content", "body", "message", "post"]
        text = None

        for key in text_keys:
            if key in data:
                text = data[key]
                break

        if text is None:
            text = str(data)

        return TextSample(
            text=text,
            source=data.get("source", source),
            author=data.get("author") or data.get("username") or data.get("user"),
            timestamp=data.get("timestamp") or data.get("date") or data.get("created_at"),
            metadata={k: v for k, v in data.items() if k not in text_keys + ["source", "author", "username", "user", "timestamp", "date", "created_at"]},
        )


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read a plain text file.

    Args:
        path: Path to file
        encoding: Text encoding

    Returns:
        File contents as string
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def read_json_samples(
    path: str,
    text_field: str = "text",
    encoding: str = "utf-8",
) -> List[Dict[str, Any]]:
    """
    Read samples from a JSON file.

    Args:
        path: Path to JSON file
        text_field: Field name containing text
        encoding: Text encoding

    Returns:
        List of sample dictionaries
    """
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    else:
        return [data]


def read_csv_samples(
    path: str,
    text_column: Optional[str] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> List[Dict[str, Any]]:
    """
    Read samples from a CSV file.

    Args:
        path: Path to CSV file
        text_column: Column containing text (auto-detected if None)
        delimiter: CSV delimiter
        encoding: Text encoding

    Returns:
        List of sample dictionaries
    """
    samples = []

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if text_column is None and reader.fieldnames:
            text_columns = ["text", "content", "body", "message", "post"]
            for col in text_columns:
                if col in reader.fieldnames:
                    text_column = col
                    break

        for row in reader:
            samples.append(dict(row))

    return samples


def stream_large_file(
    path: str,
    chunk_size: int = 1000,
    encoding: str = "utf-8",
) -> Generator[List[str], None, None]:
    """
    Stream a large file in chunks.

    Args:
        path: Path to file
        chunk_size: Number of lines per chunk
        encoding: Text encoding

    Yields:
        Chunks of lines
    """
    path = Path(path)

    with open(path, "r", encoding=encoding) as f:
        chunk = []
        for line in f:
            chunk.append(line.rstrip("\n"))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk


def detect_file_encoding(path: str) -> str:
    """
    Attempt to detect file encoding.

    Args:
        path: Path to file

    Returns:
        Detected encoding name
    """
    try:
        import chardet

        with open(path, "rb") as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            return result.get("encoding", "utf-8") or "utf-8"
    except ImportError:
        return "utf-8"
