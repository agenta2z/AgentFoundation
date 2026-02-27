"""
CLI for knowledge operations, used by the TypeScript CLI via subprocess.

Usage:
    python -m agent_foundation.knowledge.cli add "some knowledge text"
    python -m agent_foundation.knowledge.cli search "query"
    python -m agent_foundation.knowledge.cli search "query" --entity-id user123 --domain debugging
    python -m agent_foundation.knowledge.cli list
    python -m agent_foundation.knowledge.cli clear

Outputs JSON to stdout for structured communication with the TS caller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.knowledge_piece import KnowledgePiece
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from rich_python_utils.service_utils.graph_service.file_graph_service import (
    FileGraphService,
)
from rich_python_utils.service_utils.keyvalue_service.file_keyvalue_service import (
    FileKeyValueService,
)
from rich_python_utils.service_utils.retrieval_service.file_retrieval_service import (
    FileRetrievalService,
)


def _get_default_data_dir() -> Path:
    """Return the default data directory for the knowledge CLI.

    Uses ~/.cache/agent_foundation/knowledge, consistent with the
    AgentFoundation cache convention.
    """
    return Path.home() / ".cache" / "agent_foundation" / "knowledge"


def _create_kb(data_dir: Optional[Path] = None) -> KnowledgeBase:
    """Create a file-backed KnowledgeBase instance.

    Args:
        data_dir: Override for the data directory. Defaults to
                  ``~/.cache/agent_foundation/knowledge``.
    """
    if data_dir is None:
        data_dir = _get_default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    kv = FileKeyValueService(base_path=str(data_dir / "metadata"))
    retrieval = FileRetrievalService(base_path=str(data_dir / "pieces"))
    graph = FileGraphService(base_path=str(data_dir / "graph"))

    return KnowledgeBase(
        metadata_store=KeyValueMetadataStore(kv_service=kv),
        piece_store=RetrievalKnowledgePieceStore(retrieval_service=retrieval),
        graph_store=GraphServiceEntityGraphStore(graph_service=graph),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the knowledge CLI."""
    parser = argparse.ArgumentParser(description="Knowledge CLI")
    parser.add_argument("command", choices=["add", "search", "list", "clear"])
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--entity-id", default=None, help="Scope operations to an entity")
    parser.add_argument("--domain", default=None, help="Filter search by domain")
    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the knowledge CLI.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    kb = _create_kb()
    try:
        if args.command == "add":
            if not args.text:
                print(json.dumps({"error": "No text provided"}))
                sys.exit(1)
            piece = KnowledgePiece(content=args.text, entity_id=args.entity_id)
            piece_id = kb.add_piece(piece)
            print(json.dumps({"ok": True, "piece_id": piece_id}))

        elif args.command == "search":
            if not args.text:
                print(json.dumps({"error": "No query provided"}))
                sys.exit(1)
            result = kb.retrieve(
                args.text,
                entity_id=args.entity_id,
                domain=args.domain,
            )
            items = [
                {"piece_id": p.piece_id, "content": p.content[:200], "score": s}
                for p, s in (result.pieces or [])
            ]
            print(json.dumps({"ok": True, "items": items}))

        elif args.command == "list":
            pieces = kb.piece_store.list_all(entity_id=args.entity_id)
            items = [
                {"piece_id": p.piece_id, "content": p.content[:200]}
                for p in pieces
            ]
            print(json.dumps({"ok": True, "items": items}))

        elif args.command == "clear":
            pieces = kb.piece_store.list_all(entity_id=args.entity_id)
            for piece in pieces:
                kb.remove_piece(piece.piece_id)
            print(json.dumps({"ok": True, "cleared": len(pieces)}))

    finally:
        kb.close()


if __name__ == "__main__":
    main()
