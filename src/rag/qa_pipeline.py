import os
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ygo-cards")


def _build_vectorstore() -> PineconeVectorStore:
    """Load Pinecone vector DB."""
    if not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY is not set.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


def _build_llm() -> ChatOpenAI:
    """Load OpenAI model."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )


_vectorstore = _build_vectorstore()
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
_llm = _build_llm()

_prompt = ChatPromptTemplate.from_template("""
You are a Yu-Gi-Oh expert AI. Use ONLY the provided context to answer the player's question.
If context is not enough, say so.

CONTEXT:
{context}

QUESTION:
{question}
""")

_rag_chain = (
    RunnableParallel(
        context=_retriever,
        question=RunnablePassthrough()
    )
    | _prompt
    | _llm
)


def answer_question(question: str) -> Tuple[str, List[Document]]:
    """Run the RAG chain and also return the retrieved source docs."""
    answer_msg = _rag_chain.invoke(question)
    answer_text = answer_msg.content

    sources: List[Document] = _retriever.invoke(question)
    return answer_text, sources


def _print_sources(sources: List[Document]) -> None:
    print("\nTop retrieved cards:")
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        name = meta.get("name", "Unknown")
        card_type = meta.get("type", "Unknown type")
        atk = meta.get("atk", "N/A")
        defe = meta.get("def", "N/A")
        print(f"{i}. {name} ({card_type}) - ATK {atk}, DEF {defe}")


def main():
    print("Yu-Gi-Oh! RAG QA (Pinecone, LCEL)")
    print("Type a question (or 'quit').\n")

    while True:
        q = input("Your question: ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        print("Thinking...\n")
        answer, sources = answer_question(q)

        print("\n=== ANSWER ===")
        print(answer)

        _print_sources(sources)


if __name__ == "__main__":
    main()