"""Debug session for capturing ingestion artifacts and logs.

This module provides a debug session system for knowledge ingestion that
creates timestamped directories for storing various artifacts generated
during the ingestion process.

Usage:
    # Option 1: Use default ~/.cache/knowledge_ingestion/ location
    session = IngestionDebugSession()

    # Option 2: Use project-relative or custom runtime folder
    session = IngestionDebugSession(
        custom_root=Path("/tmp/my_debug_root")
    )

    # Option 3: Use explicit runtime directory (no timestamp suffix)
    session = IngestionDebugSession(
        runtime_dir=Path("/tmp/my_exact_debug_folder")
    )

Artifacts are saved in timestamped directories:
    {runtime_dir}/ingest_YYYYMMDD_HHMMSS/
    ├── chunks/           # Raw text chunks before LLM processing
    ├── prompts/          # Prompts sent to LLM
    ├── responses/        # Raw LLM responses
    ├── structured/       # Parsed structured JSON from LLM
    ├── merged/           # Final merged knowledge data
    └── logs/             # Ingestion logs
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional


logger: logging.Logger = logging.getLogger(__name__)


def get_knowledge_base_dir() -> Path:
    """Get knowledge ingestion base directory.

    Priority:
    1. AGENT_FOUNDATION_CACHE_DIR environment variable
    2. XDG_CACHE_HOME environment variable (XDG standard)
    3. ~/.cache (XDG default)
    """
    if af_cache := os.environ.get("AGENT_FOUNDATION_CACHE_DIR"):
        return Path(af_cache) / "knowledge_ingestion"

    if xdg_cache := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg_cache) / "knowledge_ingestion"

    return Path.home() / ".cache" / "knowledge_ingestion"


def get_ingestion_runtime_dir(
    timestamp: Optional[datetime] = None,
    custom_root: Optional[Path] = None,
) -> Path:
    """Get runtime directory with timestamp suffix for current ingestion session.

    Format: {base_dir}/ingest_YYYYMMDD_HHMMSS/

    Args:
        timestamp: Optional timestamp for the runtime dir. If None, uses current time.
        custom_root: Optional custom root directory.

    Returns:
        Path to the timestamped runtime directory.
    """
    if timestamp is None:
        timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    if custom_root is not None:
        return custom_root / f"ingest_{timestamp_str}"

    return get_knowledge_base_dir() / f"ingest_{timestamp_str}"


def list_all_ingestion_sessions(custom_root: Optional[Path] = None) -> List[Path]:
    """List all ingestion session directories.

    Args:
        custom_root: Optional custom root directory. If None, uses default location.

    Returns:
        List of paths to ingestion session directories, sorted by most recent first.
    """
    base_dir = custom_root if custom_root else get_knowledge_base_dir()
    if not base_dir.exists():
        return []

    return sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("ingest_")],
        reverse=True,
    )


class IngestionDebugSession:
    """Debug session for capturing ingestion artifacts.

    Creates structured directories for storing various artifacts generated
    during the ingestion process: chunks, prompts, responses, structured
    data, merged data, and logs.
    """

    def __init__(
        self,
        runtime_dir: Optional[Path] = None,
        custom_root: Optional[Path] = None,
        enable_file_logging: bool = True,
    ) -> None:
        """Initialize debug session.

        Args:
            runtime_dir: Optional exact runtime directory for this session.
            custom_root: Optional custom root directory for artifacts.
                        Creates a timestamped directory under this path.
            enable_file_logging: Whether to enable file-based logging.
        """
        if runtime_dir is None:
            runtime_dir = get_ingestion_runtime_dir(custom_root=custom_root)

        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different artifact types
        self.chunks_dir = runtime_dir / "chunks"
        self.prompts_dir = runtime_dir / "prompts"
        self.responses_dir = runtime_dir / "responses"
        self.structured_dir = runtime_dir / "structured"
        self.merged_dir = runtime_dir / "merged"
        self.logs_dir = runtime_dir / "logs"

        for d in [
            self.chunks_dir,
            self.prompts_dir,
            self.responses_dir,
            self.structured_dir,
            self.merged_dir,
            self.logs_dir,
        ]:
            d.mkdir(exist_ok=True)

        # Track files saved
        self._file_handlers: List[logging.FileHandler] = []
        self._files_saved: List[Path] = []

        # Setup file logging if enabled
        if enable_file_logging:
            self._setup_logging()

        logger.info("Debug session created at: %s", self.runtime_dir)

    def _setup_logging(self) -> None:
        """Configure file logging for this session."""
        log_file = self.logs_dir / "ingestion.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handler.setLevel(logging.DEBUG)

        # Add to knowledge module loggers
        modules_to_log = [
            "agent_foundation.knowledge",
            "agent_foundation.knowledge.ingestion",
            "agent_foundation.knowledge.ingestion.document_ingester",
            "agent_foundation.knowledge.ingestion.chunker",
        ]
        for module in modules_to_log:
            module_logger = logging.getLogger(module)
            module_logger.addHandler(handler)
            module_logger.setLevel(logging.DEBUG)

        self._file_handlers.append(handler)
        self._files_saved.append(log_file)
        logger.debug("File logging enabled: %s", log_file)

    def save_chunk(
        self,
        doc_id: str,
        chunk_idx: int,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Save a raw text chunk for debugging.

        Args:
            doc_id: Document identifier.
            chunk_idx: Index of the chunk within the document.
            content: Raw text content of the chunk.
            metadata: Optional metadata about the chunk.

        Returns:
            Path to the saved chunk file.
        """
        safe_doc_id = self._sanitize_filename(doc_id)
        path = self.chunks_dir / f"{safe_doc_id}_chunk_{chunk_idx:03d}.txt"

        output = ""
        if metadata:
            output = f"# Metadata: {json.dumps(metadata)}\n\n"
        output += content

        path.write_text(output)
        self._files_saved.append(path)
        logger.debug("Saved chunk: %s", path.name)
        return path

    def save_prompt(self, doc_id: str, chunk_idx: int, prompt: str) -> Path:
        """Save LLM prompt for debugging.

        Args:
            doc_id: Document identifier.
            chunk_idx: Index of the chunk within the document.
            prompt: The prompt sent to the LLM.

        Returns:
            Path to the saved prompt file.
        """
        safe_doc_id = self._sanitize_filename(doc_id)
        path = self.prompts_dir / f"{safe_doc_id}_chunk_{chunk_idx:03d}_prompt.txt"
        path.write_text(prompt)
        self._files_saved.append(path)
        logger.debug("Saved prompt: %s", path.name)
        return path

    def save_response(self, doc_id: str, chunk_idx: int, response: str) -> Path:
        """Save raw LLM response for debugging.

        Args:
            doc_id: Document identifier.
            chunk_idx: Index of the chunk within the document.
            response: The raw response from the LLM.

        Returns:
            Path to the saved response file.
        """
        safe_doc_id = self._sanitize_filename(doc_id)
        path = self.responses_dir / f"{safe_doc_id}_chunk_{chunk_idx:03d}_response.txt"
        path.write_text(response)
        self._files_saved.append(path)
        logger.debug("Saved response: %s", path.name)
        return path

    def save_structured(
        self, doc_id: str, chunk_idx: int, data: dict
    ) -> Path:
        """Save parsed structured JSON from LLM response.

        Args:
            doc_id: Document identifier.
            chunk_idx: Index of the chunk within the document.
            data: The parsed structured data.

        Returns:
            Path to the saved JSON file.
        """
        safe_doc_id = self._sanitize_filename(doc_id)
        path = (
            self.structured_dir / f"{safe_doc_id}_chunk_{chunk_idx:03d}_structured.json"
        )
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        self._files_saved.append(path)
        logger.debug("Saved structured data: %s", path.name)
        return path

    def save_merged(self, doc_id: str, data: dict) -> Path:
        """Save final merged knowledge data for a document.

        Args:
            doc_id: Document identifier.
            data: The merged knowledge data for the entire document.

        Returns:
            Path to the saved JSON file.
        """
        safe_doc_id = self._sanitize_filename(doc_id)
        path = self.merged_dir / f"{safe_doc_id}_merged.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        self._files_saved.append(path)
        logger.debug("Saved merged data: %s", path.name)
        return path

    def save_log(self, doc_id: str, message: str) -> Path:
        """Save a log message for a specific document.

        Args:
            doc_id: Document identifier.
            message: Log message to save.

        Returns:
            Path to the saved log file.
        """
        safe_doc_id = self._sanitize_filename(doc_id)
        path = self.logs_dir / f"{safe_doc_id}_log.txt"

        # Append to existing log file
        with open(path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] {message}\n")

        if path not in self._files_saved:
            self._files_saved.append(path)
        logger.debug("Saved log: %s", path.name)
        return path

    def save_summary(self, summary: dict) -> Path:
        """Save ingestion session summary.

        Args:
            summary: Summary data including stats and any errors.

        Returns:
            Path to the saved summary file.
        """
        path = self.runtime_dir / "session_summary.json"
        summary["runtime_dir"] = str(self.runtime_dir)
        summary["timestamp"] = datetime.now().isoformat()
        summary["files_saved"] = [str(p) for p in self._files_saved]
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        self._files_saved.append(path)
        logger.info("Session summary saved: %s", path)
        return path

    def get_files_saved(self) -> List[Path]:
        """Get list of all files saved in this session."""
        return self._files_saved.copy()

    def close(self) -> None:
        """Close the debug session and cleanup resources."""
        for handler in self._file_handlers:
            handler.close()
            for module in [
                "agent_foundation.knowledge",
                "agent_foundation.knowledge.ingestion",
            ]:
                module_logger = logging.getLogger(module)
                module_logger.removeHandler(handler)

        logger.info(
            "Debug session closed. %d files saved to: %s",
            len(self._files_saved),
            self.runtime_dir,
        )

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
            name = name.replace(char, "_")
        if len(name) > 100:
            name = name[:100]
        return name

    def __enter__(self) -> "IngestionDebugSession":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type] = None,
        exc_val: Optional[BaseException] = None,
        exc_tb: Optional[Any] = None,
    ) -> None:
        """Context manager exit."""
        self.close()
