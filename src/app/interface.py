import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Make sure we can import from src/ even when running this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.rag.qa_pipeline import answer_question
from src.vision.card_recognition_gpt import identify_card_from_image_gpt
from src.audio.speech_to_text import WhisperTranscriber
from langchain_core.documents import Document


# Load Whisper once (so we don't reload model on every message)
transcriber = WhisperTranscriber(model_size="small")  # "base"/"tiny" if you need speed


# ---------- helpers ----------

def format_sources(sources: List[Document]) -> str:
    """Format retrieved docs into a human-readable list, with similarity scores if available."""
    if not sources:
        return "No sources retrieved."

    lines: List[str] = [f"{len(sources)} cards retrieved:"]
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata or {}
        name = meta.get("name", "Unknown")
        card_type = meta.get("type", "Unknown type")
        atk = meta.get("atk", "N/A")
        defe = meta.get("def", "N/A")
        score = meta.get("similarity_score")

        if isinstance(score, (int, float)):
            lines.append(
                f"{i}. {name} ({card_type}) - ATK {atk}, DEF {defe}  [score={score:.3f}]"
            )
        else:
            lines.append(
                f"{i}. {name} ({card_type}) - ATK {atk}, DEF {defe}"
            )

    return "\n".join(lines)


def format_history_for_rag(messages: List[Dict[str, Any]]) -> str:
    """
    Turn Chatbot messages [{"role": "...", "content": "..."}, ...]
    into a "User: ... / Assistant: ..." string for the RAG prompt.
    """
    if not messages:
        return ""

    lines: List[str] = []
    last_user: Optional[str] = None

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            last_user = content
        elif role == "assistant":
            if last_user is not None:
                lines.append(f"User: {last_user}")
                lines.append(f"Assistant: {content}")
                last_user = None

    return "\n".join(lines)


def transcribe_audio(audio_path: Optional[str]) -> str:
    if not audio_path:
        return ""
    try:
        return transcriber.transcribe(audio_path)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""


def recognize_card_from_image(image_path: Optional[str]) -> Tuple[str, str]:
    """
    Run GPT-based card recognition if an image is provided.
    Returns (recognized_card_name_or_empty, debug_string).
    """
    if not image_path:
        return "", "No image provided."

    try:
        card_name, docs = identify_card_from_image_gpt(image_path)
        recognized = card_name or ""
        debug = f"Recognized card: {recognized}" if recognized else "No card recognized."
        return recognized, debug
    except Exception as e:
        return "", f"Error identifying card: {e}"


# ---------- main chat handler ----------

def chat_respond(
    user_message: str,
    chat_history: List[Dict[str, Any]],
    image_path: Optional[str],
    audio_path: Optional[str],
    model_id: str,
):
    """
    Gradio Chatbot handler.

    Inputs:
      - user_message: the text the user just typed
      - chat_history: list of {"role": ..., "content": ...} from the Chatbot
      - image_path: optional card image for this turn
      - audio_path: optional audio question for this turn

    Outputs:
      - new_user_message (for the textbox, usually empty to clear it)
      - updated_chat_history (for Chatbot, as messages)
      - recognized_card (Textbox)
      - sources_text (Textbox)
      - debug_info (Textbox)
    """
    user_message = (user_message or "").strip()
    chat_history = chat_history or []

    if not user_message and not audio_path:
        debug_info = "No text or audio question provided."
        return "", chat_history, "", "", debug_info

    # 1) Transcribe audio if present
    transcript = transcribe_audio(audio_path)

    # 2) Build the final natural-language question for this turn
    if user_message and transcript:
        final_question = f"{user_message}\n(Spoken clarification: {transcript})"
    elif transcript and not user_message:
        final_question = transcript
        # Also show transcript in chat so the user sees what was heard
        user_message = transcript
    else:
        final_question = user_message

    # 3) Recognize card from image (if any)
    recognized_card, card_debug = recognize_card_from_image(image_path)

    # 4) Build the RAG question, including the recognized card name if available
    if recognized_card:
        rag_question = f"For the card {recognized_card}: {final_question}"
    else:
        rag_question = final_question

    # 5) Build chat history string for RAG from messages
    chat_history_str = format_history_for_rag(chat_history)

    # 6) Call RAG pipeline
    score_debug = "No similarity scores."
    try:
        answer, sources = answer_question(rag_question, chat_history_str, model_id=model_id)
        sources_text = format_sources(sources)
        qa_error = None

        # Build a compact similarity-score summary for the debug console
        if sources:
            lines = []
            for i, doc in enumerate(sources, start=1):
                meta = doc.metadata or {}
                name = meta.get("name", "Unknown")
                score = meta.get("similarity_score")
                if isinstance(score, (int, float)):
                    lines.append(f"{i}. {name} – score={score:.4f}")
                else:
                    lines.append(f"{i}. {name} – score=N/A")
            score_debug = "Similarity scores:\n" + "\n".join(lines)
        else:
            score_debug = "Similarity scores: no sources returned."

    except Exception as e:
        answer = f"Error during QA: {e}"
        sources_text = ""
        qa_error = str(e)
        score_debug = f"Similarity scores: unavailable due to error: {e}"

    # 7) Update visible chat history in "messages" format
    if qa_error is None:
        # Append the new user and assistant messages
        chat_history = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]

    # 8) Build debug info
    debug_lines = [
        f"User message: {user_message!r}",
        f"Transcript: {transcript!r}",
        f"RAG question: {rag_question!r}",
        f"History length (messages): {len(chat_history)}",
        card_debug,
        score_debug,
    ]
    debug_info = "\n".join(debug_lines)

    # Return:
    # - "" to clear the textbox
    # - updated chat history for the Chatbot (messages format)
    # - recognized card name
    # - sources list
    # - debug info
    return "", chat_history, recognized_card, sources_text, debug_info


def clear_chat():
    """Handler for 'New chat / Clear chat' button."""
    empty_history: List[Dict[str, Any]] = []
    debug_info = "Chat cleared."
    return "", empty_history, "", "", debug_info


# ---------- UI ----------


def build_ui():
    with gr.Blocks(title="Yu-Gi-Oh! Multimodal Strategy Assistant") as demo:
        gr.Markdown(
            """
            # 🃏 Yu-Gi-Oh! Multimodal Strategy Assistant
            
            - Chat like with a real assistant  
            - Attach a **card image** or **record audio** for each message  
            - Get rulings, stats, and strategy tips
            """
        )

        with gr.Row():
            # Left: chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=600,
                    #show_copy_button=True,
                )

                # Text input + send
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask anything Yu-Gi-Oh-related...",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # Attachments under the textbox
                with gr.Row():
                    image_input = gr.Image(
                        type="filepath",
                        label="Card image",
                    )
                    audio_input = gr.Audio(
                        type="filepath",
                        label="Voice question",
                    )

                clear_btn = gr.Button(
                    "🗑️ New chat / Clear",
                    variant="secondary",
                )

            # Right: tabs + accordion
            with gr.Column(scale=2):
                gr.Markdown("### Card & retrieval info")

                model_choices = [
                    "gpt-4o-mini",
                    "mistral-7b-instruct",
                    "llama-3.1-8b-instruct",
                ]

                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value="gpt-4o-mini",
                    label="LLM model",
                )

                with gr.Tab("Card"):
                    card_output = gr.Textbox(
                        label="Recognized card (from image)",
                        interactive=False,
                        lines=2,
                    )

                with gr.Tab("Retrieved cards"):
                    sources_output = gr.Textbox(
                        lines=12,
                        label="Retrieved cards (from Pinecone)",
                        interactive=False,
                    )

                with gr.Accordion("Debug (development only)", open=False):
                    debug_output = gr.Textbox(
                        lines=10,
                        label="Debug info",
                        interactive=False,
                    )

        # --- Wiring ---
        msg.submit(
            chat_respond,
            inputs=[msg, chatbot, image_input, audio_input, model_dropdown],
            outputs=[msg, chatbot, card_output, sources_output, debug_output],
        )

        send_btn.click(
            chat_respond,
            inputs=[msg, chatbot, image_input, audio_input, model_dropdown],
            outputs=[msg, chatbot, card_output, sources_output, debug_output],
        )

        clear_btn.click(
            clear_chat,
            inputs=None,
            outputs=[msg, chatbot, card_output, sources_output, debug_output],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        theme=gr.themes.Soft(),         # 👈 theme goes here in Gradio 6
        # css=".my-class { color: red; }",  # (optional) custom CSS also goes here now
    )