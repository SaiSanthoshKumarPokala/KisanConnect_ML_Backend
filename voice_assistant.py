from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, File, HTTPException, Request as FastAPIRequest, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


VOICE_ASSISTANT_HTML = Path(__file__).resolve().parent / "static" / "voice_assistant" / "index.html"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

VOICE_SUPPORT_CONTEXT = (
    "We are an agricultural e-commerce and rental platform. Users can buy seeds, fertilizers, "
    "and pesticides. Users can also rent farming equipment like tractors, rotavators, and "
    "harvesters by the hour or day. To rent a tractor, go to the 'Equipment' tab, select your "
    "dates, and upload your ID. Delivery of heavy equipment takes 24 hours. We support payments "
    "via UPI and credit card."
)

VOICE_SUPPORT_CHUNKS = [
    "We are an agricultural e-commerce and rental platform. Users can buy seeds, fertilizers, and pesticides.",
    "Users can also rent farming equipment like tractors, rotavators, and harvesters by the hour or day.",
    "To rent a tractor, go to the Equipment tab, select your dates, and upload your ID.",
    "Delivery of heavy equipment takes 24 hours. We support payments via UPI and credit card.",
]

SARVAM_LANGUAGE_MAP = {
    "en": "en-IN",
    "en-us": "en-IN",
    "en-gb": "en-IN",
    "english": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "hindi": "hi-IN",
    "te": "te-IN",
    "te-in": "te-IN",
    "telugu": "te-IN",
    "ta": "ta-IN",
    "tamil": "ta-IN",
    "bn": "bn-IN",
    "bengali": "bn-IN",
    "gu": "gu-IN",
    "gujarati": "gu-IN",
    "kn": "kn-IN",
    "kannada": "kn-IN",
    "ml": "ml-IN",
    "malayalam": "ml-IN",
    "mr": "mr-IN",
    "marathi": "mr-IN",
    "od": "od-IN",
    "or": "od-IN",
    "odia": "od-IN",
    "pa": "pa-IN",
    "punjabi": "pa-IN",
}

router = APIRouter(tags=["voice-assistant"])


def initialize_voice_assistant(app: Any) -> None:
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        documents = [
            Document(page_content=chunk, metadata={"source": "website_support", "chunk_id": index})
            for index, chunk in enumerate(VOICE_SUPPORT_CHUNKS)
        ]
        app.state.voice_assistant_vectorstore = FAISS.from_documents(documents, embeddings)
        app.state.voice_assistant_error = None
    except Exception as exc:  # pragma: no cover - initialization error path
        app.state.voice_assistant_vectorstore = None
        app.state.voice_assistant_error = str(exc)


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the backend.")
    return Groq(api_key=api_key)


def _extract_transcription_text(transcription: Any) -> str:
    text = getattr(transcription, "text", None)
    if text:
        return str(text).strip()
    if isinstance(transcription, dict):
        return str(transcription.get("text", "")).strip()
    if hasattr(transcription, "model_dump"):
        payload = transcription.model_dump()
        return str(payload.get("text", "")).strip()
    return ""


def _extract_transcription_language(transcription: Any, transcript_text: str) -> str:
    language = getattr(transcription, "language", None)
    if language:
        return str(language).strip()

    if isinstance(transcription, dict) and transcription.get("language"):
        return str(transcription["language"]).strip()

    if any("\u0c00" <= char <= "\u0c7f" for char in transcript_text):
        return "te"
    if any("\u0900" <= char <= "\u097f" for char in transcript_text):
        return "hi"
    return "en"


def _map_to_sarvam_language(language: str, transcript_text: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in SARVAM_LANGUAGE_MAP:
        return SARVAM_LANGUAGE_MAP[normalized]

    if any("\u0c00" <= char <= "\u0c7f" for char in transcript_text):
        return "te-IN"
    if any("\u0900" <= char <= "\u097f" for char in transcript_text):
        return "hi-IN"
    return "en-IN"


def _transcribe_audio(audio_bytes: bytes, filename: str) -> tuple[str, str]:
    client = _get_groq_client()
    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo"),
        response_format="verbose_json",
        temperature=0.0,
    )
    transcript_text = _extract_transcription_text(transcription)
    if not transcript_text:
        raise HTTPException(status_code=502, detail="Groq ASR returned an empty transcript.")

    detected_language = _extract_transcription_language(transcription, transcript_text)
    return transcript_text, detected_language


def _retrieve_context(app: Any, query: str) -> list[Document]:
    vectorstore = getattr(app.state, "voice_assistant_vectorstore", None)
    error = getattr(app.state, "voice_assistant_error", None)
    if vectorstore is None:
        raise HTTPException(
            status_code=500,
            detail=f"Voice assistant RAG store is not available. {error or 'Unknown initialization error.'}",
        )
    return vectorstore.similarity_search(query, k=2)


def _generate_answer(question: str, retrieved_docs: list[Document]) -> str:
    client = _get_groq_client()
    context = "\n\n".join(f"- {doc.page_content}" for doc in retrieved_docs)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a multilingual website customer support assistant. "
                    "Answer using only the provided website context. "
                    "Be concise, practical, and answer in the same language as the user's question. "
                    "If the answer is not in the context, clearly say you do not know from the available website information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Website context:\n{context}\n\n"
                    f"User question:\n{question}\n\n"
                    "Return only the answer."
                ),
            },
        ],
    )
    answer = completion.choices[0].message.content if completion.choices else ""
    answer = (answer or "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="Groq LLM returned an empty answer.")
    return answer


def _synthesize_speech(text: str, language_code: str) -> tuple[bytes, str]:
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY is not configured on the backend.")

    payload = json.dumps(
        {
            "text": text,
            "target_language_code": language_code,
            "speaker": os.getenv("SARVAM_TTS_SPEAKER", "anushka"),
            "model": os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
            "speech_sample_rate": 24000,
            "pace": 1.0,
        }
    ).encode("utf-8")

    request = Request(
        SARVAM_TTS_URL,
        data=payload,
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw_payload = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(error_body)
            detail = parsed.get("error", {}).get("message") or error_body
        except json.JSONDecodeError:
            detail = error_body or str(exc)
        raise HTTPException(status_code=502, detail=f"Sarvam TTS error: {detail}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail="Unable to reach Sarvam TTS.") from exc

    try:
        parsed_response = json.loads(raw_payload)
        audio_base64 = parsed_response["audios"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Sarvam TTS returned an invalid response.") from exc

    try:
        return base64.b64decode(audio_base64), "audio/wav"
    except Exception as exc:  # pragma: no cover - decode error path
        raise HTTPException(status_code=502, detail="Sarvam TTS audio decoding failed.") from exc


@router.get("/voice-assistant")
def voice_assistant_demo() -> FileResponse:
    return FileResponse(VOICE_ASSISTANT_HTML)


@router.post("/chat")
async def voice_assistant_chat(
    request: FastAPIRequest,
    file: UploadFile = File(...),
) -> JSONResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio file was uploaded.")

    transcript, detected_language = _transcribe_audio(audio_bytes, file.filename or "voice-query.webm")
    retrieved_docs = _retrieve_context(request.app, transcript)
    answer = _generate_answer(transcript, retrieved_docs)
    sarvam_language_code = _map_to_sarvam_language(detected_language, transcript)
    audio_bytes_out, audio_mime_type = _synthesize_speech(answer, sarvam_language_code)

    return JSONResponse(
        {
            "transcript": transcript,
            "detected_language": detected_language,
            "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
            "answer": answer,
            "tts_language_code": sarvam_language_code,
            "audio_base64": base64.b64encode(audio_bytes_out).decode("utf-8"),
            "audio_mime_type": audio_mime_type,
            "source_context": VOICE_SUPPORT_CONTEXT,
        }
    )
