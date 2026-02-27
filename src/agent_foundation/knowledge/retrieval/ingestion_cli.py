"""
KnowledgeIngestionCLI — CLI tool for ingesting free-form text into a KnowledgeBase via LLM structuring.

Accepts free-form user text, sends it to an LLM with a structuring prompt,
validates the output against the knowledge data file schema, and loads it
into a KnowledgeBase using KnowledgeDataLoader.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.data_loader import KnowledgeDataLoader

logger = logging.getLogger(__name__)

# The LLM structuring prompt template
STRUCTURING_PROMPT = """You are a knowledge structuring assistant. Given free-form user text, extract and organize ALL information into a structured JSON format with three sections: metadata, pieces, and graph.

Output ONLY valid JSON matching this exact schema:
{{
  "metadata": {{
    "<entity_id>": {{
      "entity_type": "<type>",
      "properties": {{ "<key>": "<value>" }}
    }}
  }},
  "pieces": [
    {{
      "piece_id": "<unique_kebab_case_id>",
      "content": "<descriptive text>",
      "knowledge_type": "<one of: fact|instruction|preference|procedure|note|episodic>",
      "info_type": "<one of: user_profile|instructions|context>",
      "tags": ["<tag1>", "<tag2>"],
      "entity_id": "<owner_entity_id or null for global>",
      "embedding_text": "<search-optimized keywords and synonyms>"
    }}
  ],
  "graph": {{
    "nodes": [
      {{ "node_id": "<type:name>", "node_type": "<type>", "label": "<display name>", "properties": {{}} }}
    ],
    "edges": [
      {{ "source_id": "<node_id>", "target_id": "<node_id>", "edge_type": "<RELATIONSHIP_TYPE>", "properties": {{}} }}
    ]
  }}
}}

Knowledge Type Guide (what the content IS):
- "fact": Factual statements about entities (memberships, attributes, relationships)
- "instruction": Behavioral directives, skills, or rules to follow
- "preference": User preferences or opinions
- "procedure": Ordered multi-step workflows (combine related instructions into a single procedure)
- "note": General observations or notes
- "episodic": Past interaction summaries or event records

Info Type Guide (where the content goes in the agent prompt):
- "user_profile": Personal info, user-scoped facts, preferences, memberships, accounts
- "instructions": Skills, procedures, behavioral directives, rules
- "context": World knowledge, ambient/situational info, past interaction summaries, general notes

Structuring Rules:
1. Identity & profile information (name, location, contact, family) → metadata properties under a user entity (entity_id format: "user:<identifier>")
2. Factual attributes about the user's memberships, accounts, or relationships with services → pieces with knowledge_type "fact" and info_type "user_profile" scoped to the user (entity_id = user), plus metadata entries for each external entity with structured fields
3. Skills, rules, instructions, or procedures → pieces with knowledge_type "instruction" or "procedure" and info_type "instructions" with entity_id=null (global knowledge). Combine related instruction steps into a single piece with the full ordered procedure — do NOT create separate pieces per step
4. World knowledge, ambient facts, past interaction summaries → pieces with appropriate knowledge_type and info_type "context" with entity_id=null
5. Create graph nodes for the user and all mentioned entities (services, organizations, tools, people). IMPORTANT: the user's graph node_id MUST use the same "user:<identifier>" format as the metadata entity_id (e.g., if metadata entity_id is "user:tony-chen", then graph node_id must also be "user:tony-chen", NOT "person:tony-chen")
6. Create graph edges for relationships: user→entity (e.g., MEMBER_OF, SHOPS_AT, USES), entity→procedure (e.g., USES_PROCEDURE), user→procedure (HAS_SKILL)
7. When an edge relates to a knowledge piece, include {{"piece_id": "<id>"}} in the edge properties
8. Generate unique piece_id values using descriptive kebab-case names
9. Add embedding_text with search-optimized keywords and synonyms for each piece
10. Do NOT include passwords, API keys, tokens, or other credentials anywhere in the output
11. Preserve all information from the input — do not omit or summarize away details
12. For well-known businesses, services, or organizations mentioned in the input, supplement their official website URL in the piece content and embedding_text (e.g., "Safeway (www.safeway.com)", "QFC (www.qfc.com)", "Whole Foods (www.wholefoodsmarket.com)"). This helps downstream agents navigate directly to the correct website. Only add URLs you are confident are correct for widely recognized entities

User input:
{user_input}

Respond with ONLY the JSON, no explanation."""



class KnowledgeIngestionCLI:
    """CLI tool for ingesting free-form text into a KnowledgeBase via LLM structuring."""

    def __init__(
        self,
        inferencer: Any,
        max_retries: int = 3,
        raw_files_store_path: Optional[str] = None,
    ):
        """Initialize with an LLM inferencer.

        Args:
            inferencer: Any callable implementing the inferencer protocol
                (e.g., ClaudeApiInferencer).
            max_retries: Max LLM call retries on invalid JSON response.
            raw_files_store_path: Optional directory to log each ingestion check-in.
        """
        self.inferencer = inferencer
        self.max_retries = max_retries
        self.raw_files_store_path = raw_files_store_path

    def ingest(
        self,
        user_text: str,
        kb: KnowledgeBase,
        artifacts: Optional[List[str]] = None,
    ) -> dict:
        """Ingest free-form text into a KnowledgeBase.

        1. Send user_text to LLM with STRUCTURING_PROMPT
        2. Parse and validate the JSON response
        3. If raw_files_store_path is configured, save check-in
        4. Use KnowledgeDataLoader to populate kb

        Args:
            user_text: Free-form user input text.
            kb: The KnowledgeBase to populate.
            artifacts: Optional list of file paths to copy into the check-in
                artifacts/ folder.

        Returns:
            Dict with counts from KnowledgeDataLoader.load()

        Raises:
            ValueError: If LLM fails to produce valid JSON after max_retries.
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                llm_response = self._call_llm(user_text)
                structured_data = self._parse_and_validate(llm_response)

                # Save check-in if configured
                if self.raw_files_store_path:
                    self._save_check_in(user_text, structured_data, artifacts)

                # Write to temp file and load via KnowledgeDataLoader
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(structured_data, f)
                    temp_path = f.name
                try:
                    counts = KnowledgeDataLoader.load(kb, temp_path)
                finally:
                    os.unlink(temp_path)

                return counts
            except ValueError as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                continue

        raise ValueError(
            f"Failed to get valid structured data after {self.max_retries} "
            f"attempts: {last_error}"
        )

    def _call_llm(self, user_text: str) -> str:
        """Call the LLM with the structuring prompt.

        Args:
            user_text: The free-form user input text.

        Returns:
            The raw LLM response string.
        """
        prompt = STRUCTURING_PROMPT.format(user_input=user_text)
        response = self.inferencer(prompt)
        # Handle InferencerResponse wrapper if present
        if hasattr(response, "select_response"):
            return response.select_response().response
        return str(response)

    def _parse_and_validate(self, llm_response: str) -> dict:
        """Parse JSON from LLM response and validate schema.

        Validates that the response is valid JSON with the required top-level
        sections (metadata, pieces, graph) and that each piece has the required
        fields (piece_id, content, knowledge_type, info_type).

        Args:
            llm_response: The raw LLM response string.

        Returns:
            The parsed and validated dict.

        Raises:
            ValueError: If response is not valid JSON or missing required sections.
        """
        # Strip markdown code fences if present (e.g. ```json ... ```)
        text = llm_response.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
            # Remove closing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")

        # Validate required top-level sections
        required_sections = ["metadata", "pieces", "graph"]
        missing = [s for s in required_sections if s not in data]
        if missing:
            raise ValueError(
                f"Missing required sections: {', '.join(missing)}"
            )

        # Validate pieces have required fields
        required_piece_fields = [
            "piece_id",
            "content",
            "knowledge_type",
            "info_type",
        ]
        for i, piece in enumerate(data.get("pieces", [])):
            missing_fields = [
                f for f in required_piece_fields if f not in piece
            ]
            if missing_fields:
                raise ValueError(
                    f"Piece {i} missing required fields: "
                    f"{', '.join(missing_fields)}"
                )

        return data

    def _save_check_in(
        self,
        user_text: str,
        structured_data: dict,
        artifacts: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save a check-in record to the raw_files_store_path directory.

        Creates a timestamped folder (e.g., 2025-06-15T10-30-00_my-name-is-tony/)
        containing:
          - raw_input.txt: the original user text
          - structured.json: the LLM-generated structured data
          - artifacts/: copies of any provided artifact files

        Skipped silently if raw_files_store_path is None.

        Args:
            user_text: The original free-form input.
            structured_data: The validated structured JSON dict.
            artifacts: Optional list of file paths to copy into artifacts/.

        Returns:
            Path to the created check-in folder, or None if
            raw_files_store_path is not configured.
        """
        if self.raw_files_store_path is None:
            return None

        # Build timestamped folder name with optional slug from user text
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # Create slug from first 5 words: lowercase, replace non-alphanumeric with hyphens
        words = user_text.split()[:5]
        slug = "-".join(words).lower()
        slug = re.sub(r"[^a-z0-9\-]", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        slug = slug[:50]

        folder_name = f"{ts}_{slug}" if slug else ts
        checkin_path = Path(self.raw_files_store_path) / folder_name
        checkin_path.mkdir(parents=True, exist_ok=True)

        # Write raw_input.txt
        (checkin_path / "raw_input.txt").write_text(user_text, encoding="utf-8")

        # Write structured.json (pretty-printed)
        (checkin_path / "structured.json").write_text(
            json.dumps(structured_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Copy artifacts if provided
        if artifacts:
            artifacts_dir = checkin_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for artifact_path in artifacts:
                src = Path(artifact_path)
                if src.is_file():
                    shutil.copy2(str(src), str(artifacts_dir / src.name))
                else:
                    logger.warning(
                        "Artifact not found or not a file, skipping: %s",
                        artifact_path,
                    )

        logger.info("Saved check-in to %s", checkin_path)
        return str(checkin_path)

    @staticmethod
    def run_cli():
        """Entry point for CLI usage. Reads from stdin or prompts interactively."""
        import sys

        print("Knowledge Ingestion CLI")
        print(
            "Enter your information (press Ctrl+D or Ctrl+Z to finish):"
        )
        user_text = sys.stdin.read()
        if not user_text.strip():
            print("No input provided.")
            return
        print(
            f"Received {len(user_text)} characters. "
            f"Use ingest() with an inferencer to process."
        )
