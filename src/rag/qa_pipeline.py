import os
import re
from collections import OrderedDict
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ygo-cards")

DEFAULT_MODEL_ID = "gpt-4o-mini"

# How many docs to ask from Pinecone per search & how many to keep overall
BASE_SEARCH_K = 10
NAME_SEARCH_K = 10
MAX_RESULTS = 10


def _build_vectorstore() -> PineconeVectorStore:
    """Load Pinecone vector DB."""
    if not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY is not set.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


# def _build_llm() -> ChatOpenAI:
#     """Load OpenAI chat model."""
#     return ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.2,
#     )

def _build_llm(model_id: str):
    """
    Build an LLM backend based on a simple string ID.

    - GPT models use ChatOpenAI (OpenAI / compatible endpoint).
    - Other IDs use HuggingFaceEndpoint (HF Inference API).
    """
    # All your OpenAI / compatible chat models
    if model_id in {"gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"}:
        return ChatOpenAI(
            model=model_id,
            temperature=0.2,
        )

    # Hugging Face hosted models via Inference API
    if model_id == "mistral-7b-instruct":
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="conversational",          # 👈 important
            max_new_tokens=512,
            temperature=0.2,
        )

    if model_id == "llama-3.1-8b-instruct":
        return HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="conversational",          # 👈 important
            max_new_tokens=512,
            temperature=0.2,
        )

    # Fallback: default GPT model
    return ChatOpenAI(
        model=DEFAULT_MODEL_ID,
        temperature=0.2,
    )


# Singletons so they’re not rebuilt every call
_vectorstore = _build_vectorstore()
#_llm = _build_llm()

# Prompt now includes chat history as well as card context
_prompt = ChatPromptTemplate.from_template("""
You are a Yu-Gi-Oh expert assistant.

You will be given:
- RECENT CHAT HISTORY between you and the player.
- INTERNAL CARD CONTEXT from a database (card texts, stats, archetypes, etc.).

You must follow these rules:

1) Decide first: is the player's latest question actually about Yu-Gi-Oh?
   - If it is clearly NOT about Yu-Gi-Oh (e.g. random math like "one = one",
     everyday life, unrelated images, etc.):
       - Ignore the INTERNAL CARD CONTEXT completely.
       - Answer normally as a helpful assistant.
       - Do NOT try to connect it to Yu-Gi-Oh at all.
   - Otherwise, treat it as a Yu-Gi-Oh question and continue below.

2) For ANY concrete mechanical detail about a card
   (summoning requirements, effect text, materials, ATK, DEF, Level/Rank/Link,
    Type, Attribute, etc.):
   - You MUST take it from the INTERNAL CARD CONTEXT.
   - You MUST be able to point to where it appears in the INTERNAL CARD CONTEXT.
   - Do NOT add or modify effects, stats, or conditions that are not present there.
   - Do NOT rely on your own memory of the game.
   - Do NOT guess or approximate these details.

   If a requested detail is not clearly present in the INTERNAL CARD CONTEXT:
   - Say that you are not sure or that this information is missing.
   - Do NOT try to fill the gap from general knowledge.

3) If the player did NOT explicitly ask about summoning conditions
   or exact effect text:
   - Do NOT describe the full summoning formula in detail.
   - It is enough to say things like "this is a Synchro monster with specific requirements"
     and then focus on strategy, combos, and deckbuilding.

4) For strategy, combos, deckbuilding advice, and general game plans:
   - You MAY use your broader Yu-Gi-Oh knowledge.
   - But any specific reference to a card's effect or stats MUST still be consistent
     with the INTERNAL CARD CONTEXT.

5) If the INTERNAL CARD CONTEXT is literally "NO_RELEVANT_CARD_FOUND":
   - Assume the system could not confidently match any Yu-Gi-Oh card in the database.
   - If you still know which card it is from the image or the chat (for example, "Shooting Star Dragon"):
       - You MAY use its name and talk in very general terms about strategies or support,
         but you MUST NOT state or assume exact effect text, materials, ATK/DEF, or
         other precise mechanics.
       - Be explicit that you do not have this card's details in the database.
   - If the question looks unrelated to Yu-Gi-Oh, just answer normally and ignore the card database.

6) If the INTERNAL CARD CONTEXT looks empty, generic, or unrelated to the card the
   player is asking about:
   - Treat it as if you have no reliable mechanical data.
   - Be explicit that you cannot confirm the exact effect/stats from the database.
   - You may still talk in general terms (e.g. "a typical Synchro Dragon boss monster")
     but avoid precise or detailed mechanical claims.

7) Do NOT mention "context", "internal card context", "database", "retrieval" or similar
   implementation details in your reply. Just answer naturally like a knowledgeable duelist.

RECENT CHAT HISTORY:
{chat_history}

INTERNAL CARD CONTEXT:
{context}

QUESTION (latest player message):
{question}
""")


def _build_context_text(sources: List[Document]) -> str:
    """Combine retrieved documents into a single text block for the prompt."""
    if not sources:
        # Sentinel string
        return "NO_RELEVANT_CARD_FOUND"

    parts: List[str] = []
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        name = meta.get("name", "Unknown")
        card_type = meta.get("type", "Unknown type")
        atk = meta.get("atk", "N/A")
        defe = meta.get("def", "N/A")
        header = f"[Card {i}] {name} ({card_type}) - ATK {atk}, DEF {defe}"
        parts.append(header)
        parts.append(doc.page_content)
        parts.append("")  # blank line

    return "\n".join(parts)


# --- name-aware helpers -------------------------------------------------

_YGO_KEYWORDS = {
    "yugioh", "yu-gi-oh", "duel", "duelist",
    "card", "cards", "monster", "monsters",
    "spell", "trap", "field spell",
    "atk", "def", "attack", "defense",
    "synchro", "fusion", "xyz", "xzy", "rank", "link", "pendulum",
    "tribute", "summon", "summoning",
    "graveyard", "banished", "banish", "hand", "deck", "extra deck",
    "banlist", "banned", "tcg", "ocg",
}


def _looks_like_ygo_question(text: str) -> bool:
    """Heuristic: decide if we should even query the card DB."""
    if not text:
        return False

    # If we can detect a card name candidate, it's definitely YGO-ish
    if _extract_card_name_candidates(text):
        return True

    lower = text.lower()
    return any(kw in lower for kw in _YGO_KEYWORDS)


def _extract_card_name_candidates(question: str) -> List[str]:
    """
    Heuristic: detect sequences that look like Yu-Gi-Oh card names.

    More tolerant than before:
    - Only the FIRST word needs to be capitalized (or look name-ish).
    - Following words can be lowercase (so 'Shoting star dragon' still works).
    - We try to stop before obvious question / utility words like 'is', 'can', 'deck', etc.
    """
    text = question or ""
    raw_tokens = text.split()

    candidates: List[str] = []
    current: List[str] = []

    START_STOP_WORDS = {
        "is", "are", "does", "do", "can",
        "what", "how", "why", "when", "where", "who",
        "for", "the", "this", "that", "these", "those",
        "a", "an",
    }
    END_STOP_WORDS = {
        "effect", "effects", "text", "stats",
        "banlist", "banned",
        "deck", "decks",
        "combo", "combos",
        "ruling", "rulings",
        "support", "supports",
        "help", "build", "play", "plays", "good", "bad",
        "is", "are", "does", "do", "can",
        "what", "how", "why", "when", "where", "who",
    }

    for token in raw_tokens:
        # strip punctuation at edges (.,?! etc.), keep inner chars like - or '
        core = re.sub(r"^[^\\w]+|[^\\w]+$", "", token)
        if not core:
            if current and len(current) >= 2:
                candidates.append(" ".join(current))
            current = []
            continue

        lower = core.lower()
        is_title = core[0].isupper()

        if not current:
            # Start a new potential name:
            # - must be capitalized AND not an obvious question word like "Is", "Can", "What"
            if is_title and lower not in START_STOP_WORDS:
                current = [core]
            else:
                current = []
        else:
            # Decide if this token ends the name
            if lower in END_STOP_WORDS:
                if current and len(current) >= 2:
                    candidates.append(" ".join(current))
                current = []
            else:
                # Continue the name, regardless of case (this allows typos / lowercase parts)
                current.append(core)

    if current and len(current) >= 2:
        candidates.append(" ".join(current))

    # Deduplicate (case-insensitive) while preserving order
    seen = set()
    unique: List[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def _retrieve_with_name_hints(question: str) -> List[Document]:
    """
    Combine:
      - normal semantic retrieval on the full question (with similarity scores)
      - plus name-only semantic searches for detected card names.

    "Zero = zero":
      - If the text doesn't look Yu-Gi-Oh-related at all,
        we skip Pinecone and return an empty list.
    """
    text = question or ""
    if not _looks_like_ygo_question(text):
        print("[RAG] Question does not look Yu-Gi-Oh-related -> skipping card retrieval.")
        return []

    # 1) Base semantic retrieval with scores on the full text
    try:
        docs_and_scores = _vectorstore.similarity_search_with_score(text, k=BASE_SEARCH_K)
    except Exception as e:
        print(f"[RAG] Error during similarity_search_with_score: {e}")
        docs_and_scores = []

    merged: "OrderedDict[str, Document]" = OrderedDict()

    def _key(doc: Document) -> str:
        meta = doc.metadata or {}
        name = str(meta.get("name", "")).strip()
        cid = str(meta.get("id", "")).strip()
        return f"{name}|{cid}"

    if docs_and_scores:
        print("\n[RAG] Base semantic matches:")
        for i, (doc, score) in enumerate(docs_and_scores, start=1):
            meta = doc.metadata or {}
            name = meta.get("name", "Unknown")
            card_id = meta.get("id", "Unknown")
            print(f"  #{i}: {name} (id={card_id}) – score={score:.4f}")

            # store score in metadata for debugging / UI
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["similarity_score"] = float(score)

            merged[_key(doc)] = doc
    else:
        print("[RAG] No base semantic matches from Pinecone.")

    # 2) Detect explicit card names in the text
    name_candidates = _extract_card_name_candidates(text)
    if name_candidates:
        print(f"[RAG] Detected card name candidates: {name_candidates}")

    # 3) For each detected name, run a pure semantic search on the name string
    #    (no strict metadata filter; this also helps with small typos).
    for name in name_candidates:
        try:
            name_docs: List[Document] = _vectorstore.similarity_search(name, k=NAME_SEARCH_K)
        except Exception as e:
            print(f"[RAG] Error during name-only search for '{name}': {e}")
            name_docs = []

        if not name_docs:
            print(f"[RAG] Name-only search found NO docs for '{name}'")
        else:
            top_meta = name_docs[0].metadata or {}
            print(
                f"[RAG] Name-only search for '{name}' returned {len(name_docs)} docs; "
                f"top match: {top_meta.get('name', 'Unknown')} (id={top_meta.get('id', 'Unknown')})"
            )

        for d in name_docs:
            merged.setdefault(_key(d), d)

    # 4) Final merged docs
    docs = list(merged.values())[:MAX_RESULTS]

    if docs:
        print("\n[RAG] Final merged docs:")
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            name = meta.get("name", "Unknown")
            card_id = meta.get("id", "Unknown")
            score = meta.get("similarity_score")
            if score is not None:
                print(f"  {i}. {name} (id={card_id}) – score={score:.4f}")
            else:
                print(f"  {i}. {name} (id={card_id}) – score=N/A (name-based retrieval)")
    else:
        print("[RAG] Retrieval returned zero docs after base + name searches.")

    return docs


# --- main QA function ---------------------------------------------------

# def answer_question(question: str, chat_history: str = "") -> Tuple[str, List[Document]]:
#     """
#     Run the RAG QA with optional chat history and return (answer_text, source_docs).

#     Retrieval is:
#       - base semantic on the question + history
#       - plus name-based search for card names detected there.
#     """
#     # Let name detection see history too (card names mentioned in earlier turns)
#     retrieval_text = f"{chat_history}\n\n{question}".strip()

#     # 1) Retrieve relevant cards from Pinecone (name-aware)
#     sources: List[Document] = _retrieve_with_name_hints(retrieval_text)

#     # 2) Build context text for the prompt
#     context_text = _build_context_text(sources)

#     # 3) Build messages from the prompt template
#     messages = _prompt.format_messages(
#         context=context_text,
#         question=question,
#         chat_history=chat_history or "",
#     )

#     # 4) Call the LLM
#     answer_msg = _llm.invoke(messages)
#     answer_text = answer_msg.content

#     return answer_text, sources

def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    """
    Convert a list of chat messages into a single text prompt for
    non-chat LLMs (like HuggingFaceEndpoint), in a simple role-tagged format.
    """
    lines = []
    for m in messages:
        role = getattr(m, "type", getattr(m, "role", ""))
        content = m.content if hasattr(m, "content") else str(m)
        if role == "system":
            # You can either skip or include system messages at the top
            lines.append(f"System: {content}")
        elif role == "human" or role == "user":
            lines.append(f"User: {content}")
        elif role == "ai" or role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(content)
    lines.append("Assistant:")
    return "\n".join(lines)


def answer_question(
    question: str,
    chat_history: str = "",
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[str, List[Document]]:
    """
    Run the RAG QA with optional chat history and return (answer_text, source_docs).

    model_id controls which LLM backend is used for the final answer.
    """
    # Let name detection see history too (card names mentioned in earlier turns)
    retrieval_text = f"{chat_history}\n\n{question}".strip()

    # 1) Retrieve relevant cards from Pinecone (name-aware)
    sources: List[Document] = _retrieve_with_name_hints(retrieval_text)

    # 2) Build context text for the prompt
    context_text = _build_context_text(sources)

    # 3) Build messages from the prompt template (chat-style)
    messages = _prompt.format_messages(
        context=context_text,
        question=question,
        chat_history=chat_history or "",
    )

    # 4) Select and call the LLM
    llm = _build_llm(model_id)
    print(f"[RAG] Using LLM backend: {type(llm).__name__} for model_id={model_id}")


    def _call_llm(active_llm):
        if isinstance(active_llm, ChatOpenAI):
            msg = active_llm.invoke(messages)
            return msg.content
        else:
            prompt_text = _messages_to_prompt(messages)
            return active_llm.invoke(prompt_text)

    try:
        answer_text = _call_llm(llm)
    except Exception as e:
        # This will show up in your terminal
        print(f"[RAG] Error from model {model_id}: {e}")
        # And you can still fall back to GPT if you want:
        fallback_llm = _build_llm(DEFAULT_MODEL_ID)
        answer_text = _call_llm(fallback_llm)
        answer_text = (
            f"(⚠️ Fallback to {DEFAULT_MODEL_ID} due to error with {model_id}: {e})\n\n"
            + answer_text
        )

    return answer_text, sources


def _print_sources(sources: List[Document]) -> None:
    print("\nTop retrieved cards:")
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        name = meta.get("name", "Unknown")
        card_type = meta.get("type", "Unknown type")
        atk = meta.get("atk", "N/A")
        defe = meta.get("def", "N/A")
        score = meta.get("similarity_score")
        if score is not None:
            print(f"{i}. {name} ({card_type}) - ATK {atk}, DEF {defe} - score={score:.4f}")
        else:
            print(f"{i}. {name} ({card_type}) - ATK {atk}, DEF {defe}")


def main():
    print("Yu-Gi-Oh! RAG QA (Pinecone, name-aware, with chat history)")
    print("Type a question (or 'quit').\n")

    history: List[Tuple[str, str]] = []

    def _format_history(hist: List[Tuple[str, str]]) -> str:
        lines: List[str] = []
        for u, a in hist:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    while True:
        q = input("Your question: ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        chat_history_str = _format_history(history)

        print("Thinking...\n")
        answer, sources = answer_question(q, chat_history_str)

        print("\n=== ANSWER ===")
        print(answer)

        _print_sources(sources)

        # Update history
        history.append((q, answer))


if __name__ == "__main__":
    main()