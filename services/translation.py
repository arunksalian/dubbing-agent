"""
Translation Service
Uses OpenAI GPT-4o to translate speaker segments while preserving:
  - Emotional tone and register
  - Sentence length (for timing alignment)
  - Speaker grouping consistency
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "pt": "Portuguese", "it": "Italian", "ru": "Russian", "tr": "Turkish",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "no": "Norwegian", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian", "ms": "Malay", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "ur": "Urdu", "fa": "Persian",
}


def _get_language_name(lang_code: str) -> str:
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code)


SYSTEM_PROMPT = """You are a professional dubbing translator.
Translate dialogue naturally for audio dubbing, following these rules:
1. Match the length and rhythm of the original as closely as possible.
2. Preserve emotional tone, register, and speaking style.
3. Do NOT add or remove sentences — translate exactly what is said.
4. Return ONLY the translated text, nothing else.
5. Use natural spoken language, not formal written language."""


async def _translate_single(
    client: AsyncOpenAI,
    text: str,
    target_lang: str,
    speaker_id: str,
) -> str:
    """Translate one segment with exponential-backoff retry."""
    lang_name = _get_language_name(target_lang)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    ):
        with attempt:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Speaker: {speaker_id}\n"
                            f"Target language: {lang_name}\n"
                            f"Translate: {text}"
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            translated = response.choices[0].message.content.strip()
            logger.debug(
                "[%s] '%s' → '%s'", speaker_id, text[:60], translated[:60]
            )
            return translated

    return text  # unreachable — tenacity re-raises


async def translate_segments(
    segments: List[dict],
    target_language: str,
    concurrency: int = 5,
) -> List[dict]:
    """
    Translate all segments concurrently (up to `concurrency` at once).

    Args:
        segments:  list of {"speaker_id", "start", "end", "text"}
        target_language:  BCP-47 language code ("hi", "es", "fr", …)
        concurrency: max parallel API calls

    Returns:
        list of {"speaker_id", "start", "end", "text", "translated_text"}
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(concurrency)

    async def _translate_with_semaphore(seg: dict) -> dict:
        async with semaphore:
            translated = await _translate_single(
                client, seg["text"], target_language, seg["speaker_id"]
            )
            return {**seg, "translated_text": translated}

    tasks = [_translate_with_semaphore(seg) for seg in segments]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    logger.info(
        "Translation complete: %d segments → %s",
        len(results),
        _get_language_name(target_language),
    )
    return list(results)
