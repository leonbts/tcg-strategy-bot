import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from pinecone import Pinecone
from PIL import Image

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# === Config ===

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ygo-cards")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GPT_VISION_MODEL = os.environ.get("GPT_VISION_MODEL", "gpt-4.1-mini")

# How many Pinecone candidates we want to pull / keep
MAX_CANDIDATES = 15

# === Globals (so we don't reload on every call) ===

_client = OpenAI()
_pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
_index = _pc.Index(PINECONE_INDEX_NAME)
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# === Helpers ===

def _image_path_to_base64(image_path: str) -> str:
    """Load an image from disk and return it as base64 PNG."""
    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _analyze_card_image_with_gpt(image_path: str) -> Dict:
    """
    Use GPT vision to extract structured info from a Yu-Gi-Oh! card image.

    Returns a dict with fields like:
      - primary_guess_name
      - alternative_names
      - card_type
      - race
      - attribute
      - level_or_rank
      - atk
      - def
      - visible_text_fragments
      - confidence
    """
    img_b64 = _image_path_to_base64(image_path)

    system_content = (
        "You are an assistant that analyzes photos of Yu-Gi-Oh! trading cards.\n"
        "Your job:\n"
        "- Look carefully at the uploaded card image.\n"
        "- Extract all useful visible information.\n"
        "- The card text may be in ANY language (German, French, etc.).\n"
        "- If you can, map the localized card name to the official ENGLISH TCG/OCG card name.\n"
        "  For example, if the printed name is in German but you know the English name,\n"
        "  return the English name in primary_guess_name.\n"
        "- If you are not sure of the English name, put the localized name in primary_guess_name\n"
        "  and put any possible English names into alternative_names.\n\n"
        "Always respond with STRICT JSON, no extra commentary.\n"
        "Use null for fields you can't see at all."
    )

    user_text = (
        "Look at this Yu-Gi-Oh! card image and extract structured information.\n\n"
        "Return JSON with this exact schema:\n\n"
        "{\n"
        "  \"primary_guess_name\": string | null,\n"
        "  \"alternative_names\": [string, ...],\n"
        "  \"card_type\": string | null,\n"
        "  \"race\": string | null,\n"
        "  \"attribute\": string | null,\n"
        "  \"level_or_rank\": number | null,\n"
        "  \"atk\": number | null,\n"
        "  \"def\": number | null,\n"
        "  \"visible_text_fragments\": [string, ...],\n"
        "  \"confidence\": \"low\" | \"medium\" | \"high\"\n"
        "}\n\n"
        "Rules:\n"
        "- If the card name is readable, put it in primary_guess_name.\n"
        "- If you are not fully sure, put other plausible card names into alternative_names.\n"
        "- If you cannot read a value (e.g. ATK/DEF), set it to null instead of guessing wildly.\n"
        "- visible_text_fragments can be rough; partial words and phrases are fine."
    )

    response = _client.chat.completions.create(
        model=GPT_VISION_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    # Depending on OpenAI client version, message content may already be a dict
    if isinstance(raw, dict):
        return raw

    try:
        return json.loads(raw)
    except Exception:
        # Fallback: return an empty-ish dict if parsing fails
        return {
            "primary_guess_name": None,
            "alternative_names": [],
            "card_type": None,
            "race": None,
            "attribute": None,
            "level_or_rank": None,
            "atk": None,
            "def": None,
            "visible_text_fragments": [],
            "confidence": "low",
        }


def _build_search_queries_from_vision(vision: Dict) -> List[str]:
    """
    Turn GPT vision output into one or more text search queries
    that match your Pinecone metadata schema (name, type, race, attribute, level, atk, def, archetype).

    We build:
      - pure name queries (for exact matches),
      - name + metadata queries,
      - alternative name queries (for localized/approx names),
      - fragment-based queries (for non-English prints / partial OCR).
    """
    name = (vision.get("primary_guess_name") or "").strip()
    alt_names = [
        n.strip() for n in (vision.get("alternative_names") or []) if n and n.strip()
    ]
    card_type = (vision.get("card_type") or "").strip()
    race = (vision.get("race") or "").strip()
    attribute = (vision.get("attribute") or "").strip()
    level = vision.get("level_or_rank")
    atk = vision.get("atk")
    defe = vision.get("def")
    frags = vision.get("visible_text_fragments") or []

    # Build a small stats string
    parts_stats = []
    if level is not None:
        parts_stats.append(f"Level {level}")
    if atk is not None and defe is not None:
        parts_stats.append(f"ATK {atk} DEF {defe}")
    stats_str = " ".join(map(str, parts_stats))  # cast level to str as well

    # Use only a few text fragments to avoid noise
    frags = [f.strip() for f in frags if f and f.strip()]
    frags_str = " ".join(frags[:5])

    queries: List[str] = []

    # 1) Pure name query (best case: exact English name)
    if name:
        queries.append(name)

    # 2) Name + key metadata
    if name:
        queries.append(
            " ".join(
                p
                for p in [
                    name,
                    card_type,
                    race,
                    attribute,
                    stats_str,
                ]
                if p
            )
        )

    # 3) Alternative names as separate queries (handle localized/approx names)
    for alt in alt_names[:3]:
        # pure alt name
        queries.append(alt)
        # alt name + metadata
        q = " ".join(
            p
            for p in [
                alt,
                card_type,
                race,
                attribute,
                stats_str,
            ]
            if p
        )
        if q:
            queries.append(q)

    # 4) Fragments-based queries (good for non-English prints)
    if frags_str:
        # fragments + metadata
        q_frag = " ".join(
            p for p in [frags_str, card_type, race, attribute, stats_str] if p
        )
        queries.append(q_frag)

        # Also add each fragment as a tiny query if it looks name-ish (>= 4 chars)
        for frag in frags[:3]:
            if len(frag) >= 4:
                queries.append(frag)

    # 5) Very generic fallbacks if everything else is missing
    if not queries:
        fallback = " ".join(
            p for p in [card_type, race, attribute, stats_str, frags_str] if p
        )
        queries.append(fallback or "Yu-Gi-Oh card")

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for q in queries:
        q_norm = q.strip()
        if not q_norm:
            continue
        key = q_norm.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q_norm)

    return unique


def _embed_queries(queries: List[str]) -> List[List[float]]:
    """
    Embed a list of queries using the same HF model as the QA pipeline.
    """
    return _embeddings.embed_documents(queries)


def _search_candidates(
    queries: List[str],
    top_k: int = MAX_CANDIDATES,
) -> List[Dict]:
    """
    Use the same text index as your QA pipeline (ygo-cards) to fetch candidate cards.

    Returns a list of dicts with:
      - id, score
      - name, type, race, attribute, level, atk, def, archetype
    """
    vectors = _embed_queries(queries)
    all_matches = []

    for vec in vectors:
        res = _index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
        )
        all_matches.extend(res.matches)

    # Deduplicate by id, keep highest score
    best_by_id = {}
    for m in all_matches:
        if m.id not in best_by_id or m.score > best_by_id[m.id].score:
            best_by_id[m.id] = m

    # Sort descending by score
    matches = sorted(best_by_id.values(), key=lambda x: x.score, reverse=True)

    candidates: List[Dict] = []
    for m in matches[:top_k]:
        meta = m.metadata or {}
        candidates.append(
            {
                "id": m.id,
                "score": m.score,
                "name": meta.get("name"),
                "type": meta.get("type"),
                "race": meta.get("race"),
                "attribute": meta.get("attribute"),
                "level": meta.get("level"),
                "atk": meta.get("atk"),
                "def": meta.get("def"),
                "archetype": meta.get("archetype"),
            }
        )

    return candidates


def _rerank_with_gpt(
    image_path: str,
    candidates: List[Dict],
) -> Optional[Dict]:
    """
    Given the image + candidate list, ask GPT to pick the single best match
    OR say there is no good match (null_result).
    """
    if not candidates:
        return None

    img_b64 = _image_path_to_base64(image_path)

    system_content = (
        "You are a Yu-Gi-Oh! card expert. You are given a photo of a card and "
        "a small list of candidate cards with their metadata.\n"
        "Your task: choose the single candidate that EXACTLY matches the photo.\n"
        "If none match well, set null_result to true.\n"
        "Be very strict about artwork, name, and stats."
    )

    # Build a compact textual representation of candidates
    cand_lines = []
    for c in candidates:
        cand_lines.append(
            f"id={c['id']}, name={c['name']}, "
            f"type={c['type']}, race={c['race']}, "
            f"attribute={c['attribute']}, level={c['level']}, "
            f"ATK={c['atk']} DEF={c['def']}, archetype={c['archetype']}"
        )
    cand_text = "Candidates:\n" + "\n".join(cand_lines)

    user_text = (
        "Here is the Yu-Gi-Oh! card image and a list of candidate cards.\n\n"
        "Return JSON with:\n"
        "{\n"
        "  \"null_result\": boolean,\n"
        "  \"best_match_id\": string | null,\n"
        "  \"explanation\": string\n"
        "}\n\n"
        "Very important rules:\n"
        "- ONLY choose a candidate if you are VERY confident it is the exact same card.\n"
        "- Compare carefully:\n"
        "    * the artwork (pose, background, colors, composition),\n"
        "    * the card name (even across languages),\n"
        "    * the stats (Level/Rank, ATK, DEF),\n"
        "    * the card type, race, and attribute.\n"
        "- If the printed name or stats clearly do NOT match any candidate, set null_result = true.\n"
        "- When in doubt between multiple candidates, prefer null_result instead of guessing."
    )

    response = _client.chat.completions.create(
        model=GPT_VISION_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        },
                    },
                    {"type": "text", "text": cand_text},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw) if not isinstance(raw, dict) else raw
    except Exception:
        return None

    if result.get("null_result"):
        return None

    best_id = result.get("best_match_id")
    if not best_id:
        return None

    for c in candidates:
        if str(c["id"]) == str(best_id):
            return c

    return None


def identify_card_from_image_gpt(image_path: str) -> Tuple[Optional[str], List[Document]]:
    """
    Main entry point, similar to your old CLIP-based function, but using:
      - GPT vision for parsing
      - Pinecone text index for candidate retrieval
      - GPT vision again for reranking

    Returns:
      (card_name_or_none, docs_list_for_debug_or_future_use)
    """
    # 1) Vision parse
    vision = _analyze_card_image_with_gpt(image_path)
    # (optional) print for debugging:
    # print("[VISION] Parsed card info:", json.dumps(vision, indent=2, ensure_ascii=False))

    # 2) Build text queries
    queries = _build_search_queries_from_vision(vision)
    # print("[VISION] Search queries:", queries)

    # 3) Candidate retrieval from text index
    candidates = _search_candidates(queries, top_k=MAX_CANDIDATES)
    # print("[VISION] Candidates:", candidates)

    # 4) Rerank with GPT
    best = _rerank_with_gpt(image_path, candidates)
    if not best:
        # No confident match
        return None, []

    best_name = best.get("name")

    # Build a simple Document for consistency with the rest of your codebase.
    content_lines = [
        f"Name: {best.get('name')}",
        f"Type: {best.get('type')}",
        f"Race: {best.get('race')}",
        f"Attribute: {best.get('attribute')}",
        f"Level: {best.get('level')}",
        f"ATK: {best.get('atk')}",
        f"DEF: {best.get('def')}",
        f"Archetype: {best.get('archetype')}",
        f"Score: {best.get('score')}",
    ]
    doc = Document(
        page_content="\n".join(content_lines),
        metadata={k: v for k, v in best.items()},
    )

    return best_name, [doc]