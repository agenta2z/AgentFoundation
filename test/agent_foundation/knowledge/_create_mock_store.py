"""Generate anonymized mock knowledge store files for testing.

Directory layout must match service conventions:
- Metadata (FileKeyValueService):  namespace = parse_entity_type(entity_id)
    "user:alex-johnson" → namespace "user" → metadata/user/user%3Aalex-johnson.json
    "service:safeway"   → namespace "service" → metadata/service/service%3Asafeway.json
- Pieces (FileRetrievalService):  namespace = entity_id, encoded via _encode_namespace
    "user:alex-johnson"  → user/alex-johnson/  (colon → os.sep)
    None                 → _default/
- Graph (FileGraphService): always in _default/ namespace
    node_id encoded with percent-encoding: "user:alex-johnson" → user%3Aalex-johnson.json
    edge filename: source%7C%7Ctarget%7C%7Ctype.json  (|| → %7C%7C)
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent / "_mock_knowledge_store"


def write_json(rel_path, data):
    full_path = BASE / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  wrote {rel_path}")


TS = "2026-02-10T05:41:52.664765+00:00"

# ── Graph Nodes ──
print("Graph nodes:")
write_json("graph/_default/nodes/user%3Aalex-johnson.json", {
    "node_id": "user:alex-johnson",
    "node_type": "person",
    "label": "Alex Johnson",
    "properties": {
        "location": "1234 Main St, Portland, OR, 97201",
        "spaces": ["personal"],
    },
})
write_json("graph/_default/nodes/service%3Aqfc.json", {
    "node_id": "service:qfc",
    "node_type": "grocery_store",
    "label": "QFC",
    "properties": {"website": "www.qfc.com", "spaces": ["main", "developmental"]},
})
write_json("graph/_default/nodes/service%3Asafeway.json", {
    "node_id": "service:safeway",
    "node_type": "grocery_store",
    "label": "Safeway",
    "properties": {"website": "www.safeway.com", "spaces": ["main", "developmental"]},
})
write_json("graph/_default/nodes/service%3Awhole-foods.json", {
    "node_id": "service:whole-foods",
    "node_type": "grocery_store",
    "label": "Whole Foods",
    "properties": {"website": "www.wholefoodsmarket.com", "spaces": ["main", "developmental"]},
})
write_json("graph/_default/nodes/procedure%3Agrocery-shopping.json", {
    "node_id": "procedure:grocery-shopping",
    "node_type": "procedure",
    "label": "Grocery Store Shopping Procedure",
    "properties": {"spaces": ["main", "developmental"]},
})

# ── Graph Edges: MEMBER_OF ──
print("Graph edges:")
for store, piece_id, extra in [
    ("safeway", "alex-johnson-safeway-membership", {"email": "alex.j@example.com", "auth_method": "google_login"}),
    ("qfc", "alex-johnson-qfc-membership", {"email": "alex.j@example.com"}),
    ("whole-foods", "alex-johnson-whole-foods-membership", {"email": "alex.j@example.edu", "membership_type": "prime"}),
]:
    props = {"piece_id": piece_id, "spaces": ["personal", "main", "developmental"]}
    props.update(extra)
    write_json(f"graph/_default/edges/user%3Aalex-johnson%7C%7Cservice%3A{store}%7C%7CMEMBER_OF.json", {
        "source_id": "user:alex-johnson",
        "target_id": f"service:{store}",
        "edge_type": "MEMBER_OF",
        "properties": props,
    })

# ── Graph Edges: SHOPS_AT ──
for store in ["safeway", "qfc", "whole-foods"]:
    write_json(f"graph/_default/edges/user%3Aalex-johnson%7C%7Cservice%3A{store}%7C%7CSHOPS_AT.json", {
        "source_id": "user:alex-johnson",
        "target_id": f"service:{store}",
        "edge_type": "SHOPS_AT",
        "properties": {"spaces": ["personal", "main", "developmental"]},
    })

# ── Graph Edges: HAS_SKILL ──
write_json("graph/_default/edges/user%3Aalex-johnson%7C%7Cprocedure%3Agrocery-shopping%7C%7CHAS_SKILL.json", {
    "source_id": "user:alex-johnson",
    "target_id": "procedure:grocery-shopping",
    "edge_type": "HAS_SKILL",
    "properties": {"piece_id": "grocery-store-shopping-procedure", "spaces": ["personal", "main", "developmental"]},
})

# ── Graph Edges: USES_PROCEDURE ──
for store in ["safeway", "qfc", "whole-foods"]:
    write_json(f"graph/_default/edges/service%3A{store}%7C%7Cprocedure%3Agrocery-shopping%7C%7CUSES_PROCEDURE.json", {
        "source_id": f"service:{store}",
        "target_id": "procedure:grocery-shopping",
        "edge_type": "USES_PROCEDURE",
        "properties": {"piece_id": "grocery-store-shopping-procedure", "spaces": ["main", "developmental"]},
    })

# ── Metadata ──
# KeyValueMetadataStore uses parse_entity_type(entity_id) as namespace for get().
# "user:alex-johnson" → namespace "user", "service:safeway" → namespace "service"
print("Metadata:")
write_json("metadata/user/user%3Aalex-johnson.json", {
    "entity_id": "user:alex-johnson",
    "entity_type": "user",
    "properties": {
        "name": "Alex Johnson",
        "location": "1234 Main St, Portland, OR, 97201",
        "family_status": "married",
        "child_birth_date": "2024-01-15",
    },
    "created_at": TS,
    "updated_at": TS,
    "spaces": ["personal"],
})
write_json("metadata/service/service%3Aqfc.json", {
    "entity_id": "service:qfc",
    "entity_type": "service",
    "properties": {"name": "QFC", "website": "www.qfc.com", "membership_email": "alex.j@example.com"},
    "created_at": TS,
    "updated_at": TS,
    "spaces": ["main", "developmental"],
})
write_json("metadata/service/service%3Asafeway.json", {
    "entity_id": "service:safeway",
    "entity_type": "service",
    "properties": {"name": "Safeway", "website": "www.safeway.com", "membership_email": "alex.j@example.com", "auth_method": "google_login"},
    "created_at": TS,
    "updated_at": TS,
    "spaces": ["main", "developmental"],
})
write_json("metadata/service/service%3Awhole-foods.json", {
    "entity_id": "service:whole-foods",
    "entity_type": "service",
    "properties": {"name": "Whole Foods", "website": "www.wholefoodsmarket.com", "membership_type": "prime", "membership_email": "alex.j@example.edu"},
    "created_at": TS,
    "updated_at": TS,
    "spaces": ["main", "developmental"],
})

# ── Pieces ──
# FileRetrievalService._encode_namespace converts ":" to os.sep (hierarchy):
#   "user:alex-johnson" → "user/alex-johnson/"
#   None → "_default/"
print("Pieces:")
write_json("pieces/_default/grocery-store-shopping-procedure.json", {
    "doc_id": "grocery-store-shopping-procedure",
    "content": "Complete grocery store shopping procedure: Step 1 - Login first if user is a member to apply member pricing and discounts. Step 2 - Find right store and location first before further operations. Step 3 - Must add items to cart, and apply all coupons, and check out to view final price.",
    "metadata": {
        "knowledge_type": "procedure",
        "info_type": "instructions",
        "tags": ["grocery", "shopping", "workflow", "login", "cart", "checkout", "coupons", "pricing"],
        "entity_id": None,
        "source": None,
        "space": "main",
        "spaces": ["main", "developmental"],
    },
    "embedding_text": "grocery store shopping procedure login member pricing discounts find store location add items cart apply coupons checkout final price workflow steps",
    "created_at": TS,
    "updated_at": TS,
})
# User-specific pieces: "user:alex-johnson" → "user/alex-johnson/" directory
write_json("pieces/user/alex-johnson/alex-johnson-safeway-membership.json", {
    "doc_id": "alex-johnson-safeway-membership",
    "content": "Alex Johnson is a member of Safeway (www.safeway.com) with email alex.j@example.com using Google login authentication",
    "metadata": {
        "knowledge_type": "fact",
        "info_type": "user_profile",
        "tags": ["membership", "grocery", "safeway", "authentication"],
        "entity_id": "user:alex-johnson",
        "source": None,
        "space": "personal",
        "spaces": ["personal"],
    },
    "embedding_text": "Safeway member membership alex.j@example.com google login authentication grocery store www.safeway.com",
    "created_at": TS,
    "updated_at": TS,
})
write_json("pieces/user/alex-johnson/alex-johnson-qfc-membership.json", {
    "doc_id": "alex-johnson-qfc-membership",
    "content": "Alex Johnson is a member of QFC (www.qfc.com) with email alex.j@example.com",
    "metadata": {
        "knowledge_type": "fact",
        "info_type": "user_profile",
        "tags": ["membership", "grocery", "qfc"],
        "entity_id": "user:alex-johnson",
        "source": None,
        "space": "personal",
        "spaces": ["personal"],
    },
    "embedding_text": "QFC member membership alex.j@example.com grocery store www.qfc.com",
    "created_at": TS,
    "updated_at": TS,
})
write_json("pieces/user/alex-johnson/alex-johnson-whole-foods-membership.json", {
    "doc_id": "alex-johnson-whole-foods-membership",
    "content": "Alex Johnson is a Prime member of Whole Foods (www.wholefoodsmarket.com) with email alex.j@example.edu",
    "metadata": {
        "knowledge_type": "fact",
        "info_type": "user_profile",
        "tags": ["membership", "grocery", "whole-foods", "prime"],
        "entity_id": "user:alex-johnson",
        "source": None,
        "space": "personal",
        "spaces": ["personal"],
    },
    "embedding_text": "Whole Foods Prime member membership alex.j@example.edu grocery store amazon prime www.wholefoodsmarket.com",
    "created_at": TS,
    "updated_at": TS,
})

print(f"\nDone! All files written to {BASE}")
