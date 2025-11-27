import os
import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Make sure we can import from src/ even when running this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.rag.qa_pipeline import answer_question
from src.vision.card_recognition import identify_card_from_image
from src.audio.speech_to_text import WhisperTranscriber
from langchain_core.documents import Document


# Load Whisper once (so we don't reload model on every click)
transcriber = WhisperTranscriber(model_size="small")  # "base" or "tiny" if slow


def build_full_question(
    base_question: str | None,
    audio_path: str | None,
) -> Tuple[str, str]:
    """
    Combine text question + optional audio into a single question string.
    Returns (final_question, transcript_used).
    """
    base_question = (base_question or "").strip()
    transcript = ""

    if audio_path:
        try:
            transcript = transcriber.transcribe(audio_path)
        except Exception as e:
            transcript = ""
            print(f"Error during transcription: {e}")

    if not base_question and transcript:
        final_q = transcript
    elif base_question and transcript:
        final_q = f"{base_question}\n(Spoken clarification: {transcript})"
    else:
        final_q = base_question

    return final_q, transcript


def format_sources(sources: List[Document]) -> str:
    if not sources:
        return "No sources retrieved."
    lines = []
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        name = meta.get("name", "Unknown")
        card_type = meta.get("type", "Unknown type")
        atk = meta.get("atk", "N/A")
        defe = meta.get("def", "N/A")
        lines.append(f"{i}. {name} ({card_type}) - ATK {atk}, DEF {defe}")
    return "\n".join(lines)


def multimodal_qa(image_path, text_question, audio_path):
    """
    Main Gradio handler:
    - image_path: path to card image (or None)
    - text_question: user-typed question (or None/empty)
    - audio_path: path to audio question (or None)
    Returns: answer, recognized_card, retrieved_cards_text, debug_info
    """
    # 1. Build final question string from text + audio
    final_q, transcript = build_full_question(text_question, audio_path)

    if not final_q:
        return (
            "Please provide a question via text or audio.",
            "",
            "",
            "No question provided.",
        )

    # 2. Identify card from image (if provided)
    recognized_card = ""
    card_debug = ""
    if image_path:
        try:
            card_name, docs = identify_card_from_image(image_path)
            recognized_card = card_name
            card_debug = f"Recognized card: {card_name}"
        except Exception as e:
            recognized_card = ""
            card_debug = f"Error identifying card: {e}"

    # 3. Build question for RAG
    if recognized_card:
        full_question = f"For the card {recognized_card}: {final_q}"
    else:
        full_question = final_q

    # 4. Call RAG pipeline
    try:
        answer, sources = answer_question(full_question)
        sources_text = format_sources(sources)
    except Exception as e:
        answer = f"Error during QA: {e}"
        sources_text = ""
    
    # 5. Debug info (for you while devving)
    debug_lines = [
        f"Raw text question: {text_question!r}",
        f"Transcript: {transcript!r}",
        f"Final question sent to RAG: {full_question!r}",
        card_debug,
    ]
    debug_info = "\n".join(debug_lines)

    return answer, recognized_card, sources_text, debug_info


def build_ui():
    with gr.Blocks(title="Yu-Gi-Oh! Multimodal Strategy Assistant") as demo:
        gr.Markdown(
            """
            # 🃏 Yu-Gi-Oh! Multimodal Strategy Assistant
            Upload a **card image**, ask a **question by text or voice**, and get strategy tips.
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="filepath",
                    label="Card image (optional)",
                )
                audio_input = gr.Audio(
                    type="filepath",
                    label="Voice question (optional)",
                )
            with gr.Column():
                text_input = gr.Textbox(
                    lines=3,
                    label="Text question (optional)",
                    placeholder="e.g. How should I use this card in a deck?",
                )
                ask_button = gr.Button("Ask the Duelist AI")

                answer_output = gr.Textbox(
                    lines=6,
                    label="Answer",
                )
                card_output = gr.Textbox(
                    label="Recognized card",
                    interactive=False,
                )
                sources_output = gr.Textbox(
                    lines=6,
                    label="Retrieved cards (from Pinecone)",
                    interactive=False,
                )
                debug_output = gr.Textbox(
                    lines=6,
                    label="Debug info (for development)",
                    interactive=False,
                    visible=True,  # you can set False for final demo
                )

        ask_button.click(
            multimodal_qa,
            inputs=[image_input, text_input, audio_input],
            outputs=[answer_output, card_output, sources_output, debug_output],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()