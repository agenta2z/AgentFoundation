"""
Shared helpers for loading the grocery mock knowledge store into a KnowledgeBase.

Used by both the mock-LLM integration test (test_retrieval_flow_grocery.py)
and the real-LLM e2e test (test_retrieval_e2e_llm.py).
"""
import shutil
import sys
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

from rich_python_utils.service_utils.graph_service.file_graph_service import FileGraphService
from rich_python_utils.service_utils.retrieval_service.file_retrieval_service import FileRetrievalService
from rich_python_utils.service_utils.keyvalue_service.file_keyvalue_service import FileKeyValueService

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import KeyValueMetadataStore
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import RetrievalKnowledgePieceStore
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import GraphServiceEntityGraphStore


# ── Constants ──────────────────────────────────────────────────────────────

MOCK_STORE_DIR = Path(__file__).resolve().parent / "_mock_knowledge_store"
QUERY = "what is egg prices from safeway"
USER_ENTITY_ID = "user:alex-johnson"


def load_grocery_kb(tmp_path) -> KnowledgeBase:
    """Load the mock knowledge store into a KnowledgeBase.

    Copies the fixture files to ``tmp_path`` so services can read/write
    without modifying the checked-in fixture data.

    Args:
        tmp_path: A temporary directory (typically from pytest's ``tmp_path``).

    Returns:
        A fully-initialized KnowledgeBase ready for retrieval.
    """
    store_copy = Path(tmp_path) / "_knowledge_store"
    shutil.copytree(str(MOCK_STORE_DIR), str(store_copy))

    metadata_store = KeyValueMetadataStore(
        kv_service=FileKeyValueService(base_dir=str(store_copy / "metadata")),
    )
    piece_store = RetrievalKnowledgePieceStore(
        retrieval_service=FileRetrievalService(base_dir=str(store_copy / "pieces")),
    )
    graph_store = GraphServiceEntityGraphStore(
        graph_service=FileGraphService(base_dir=str(store_copy / "graph")),
    )

    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id=USER_ENTITY_ID,
    )
