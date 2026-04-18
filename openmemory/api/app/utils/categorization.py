import logging
import json
import os
from typing import List

from app.database import SessionLocal
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy import text
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


class MemoryCategories(BaseModel):
    categories: List[str]


def _load_db_llm_settings() -> dict | None:
    db = SessionLocal()

    try:
        row = db.execute(text("select value from configs where key = 'main' limit 1")).fetchone()
    except Exception as error:
        logging.debug("Failed to read llm settings from database config: %s", error)
        return None
    finally:
        try:
            db.close()
        except Exception:
            pass

    if not row or not row[0]:
        return None

    try:
        payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    except json.JSONDecodeError as error:
        logging.debug("Failed to decode sqlite config JSON: %s", error)
        return None

    llm = payload.get("mem0", {}).get("llm")
    return llm if isinstance(llm, dict) else None


def _resolve_categorization_client() -> OpenAI | None:
    llm_settings = _load_db_llm_settings()
    provider = (llm_settings or {}).get("provider") or os.getenv("LLM_PROVIDER") or "openai"
    provider = provider.lower()
    if provider == "auto":
        provider = "ollama" if os.getenv("OLLAMA_BASE_URL") else "openai"
    if provider != "openai":
        return None

    config = (llm_settings or {}).get("config") or {}
    api_key = (
        config.get("api_key")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if isinstance(api_key, str) and api_key.startswith("env:"):
        api_key = os.getenv(api_key.split(":", 1)[1])
    if not api_key:
        return None

    client_kwargs = {"api_key": api_key}
    base_url = config.get("openai_base_url") or os.getenv("LLM_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    completion = None
    try:
        client = _resolve_categorization_client()
        if client is None:
            logging.info("Skipping memory categorization because no OpenAI-compatible categorization client is configured")
            return []

        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        # Let OpenAI handle the pydantic parsing directly
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=MemoryCategories,
            temperature=0
        )

        parsed: MemoryCategories = completion.choices[0].message.parsed
        return [cat.strip().lower() for cat in parsed.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            logging.debug(f"[DEBUG] Raw response: {completion.choices[0].message.content}")
        except Exception as debug_e:
            logging.debug(f"[DEBUG] Could not extract raw response: {debug_e}")
        raise
