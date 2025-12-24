import os
import json
import logging
from typing import Any, Dict, Tuple

import requests
from peercheck import settings as project_settings

logger = logging.getLogger(__name__)


def _build_prompts(transcript_text: str, procedure_text: str) -> Tuple[str, str]:
    system_prompt = """You are an expert industrial procedure auditor. Analyze the transcript and procedure document.

            CRITICAL: The highlight_start_quote and highlight_end_quote MUST be EXACT TEXT from the PROCEDURE DOCUMENT, NOT from the transcript!

            OBJECTIVES:
            1. Find which section number (like "8.4") is being discussed in the transcript
            2. Locate that section in the PROCEDURE DOCUMENT
            3. Copy the EXACT FIRST LINE of that section from the PROCEDURE DOCUMENT as highlight_start_quote
            4. Copy the EXACT LAST LINE of that section from the PROCEDURE DOCUMENT as highlight_end_quote
            5. Analyze 3PC (Statement/Readback/Acknowledgement) patterns

            IMPORTANT RULES:
            - highlight_start_quote = First line/sentence of the section FROM THE PROCEDURE DOCUMENT
            - highlight_end_quote = Last line/sentence of the section FROM THE PROCEDURE DOCUMENT
            - DO NOT use transcript text for quotes! Only procedure document text!
            - Example: If section 8.4 starts with "ENSURE AL-V010 is open" and ends with "Record stroke time in data sheet", use those exact texts.

            3PC Classification:
            - "Match": Statement in procedure + Readback matches + Acknowledgement present
            - "Partial Match": Statement in procedure + Readback does NOT match exactly
            - "Mismatch": Statement NOT in procedure document

            Return ONLY valid JSON:
            {
            "relevant_section_number": "8.4",
            "highlight_start_quote": "EXACT first line FROM PROCEDURE DOCUMENT",
            "highlight_end_quote": "EXACT last line FROM PROCEDURE DOCUMENT",
            "interactions": [
                {"speaker": "Name", "text": "What they said", "role": "Statement", "status": "Match"}
            ]
            }
        """

    user_prompt = f"""Analyze the following procedure document and transcript. Extract the section number being discussed, find the exact start and end text of that section, and identify 3PC interactions.

            PROCEDURE DOCUMENT:
            {procedure_text}

            TRANSCRIPT:
            {transcript_text}

            TASK: Return a JSON object with this exact structure (fill in all fields with actual values, not null or empty):

            Example:
            {{
            "relevant_section_number": "8.4",
            "highlight_start_quote": "Exact first sentence or line from the section",
            "highlight_end_quote": "Exact last sentence or line from the section",
            "interactions": [
                {{
                "speaker": "Speaker name or label",
                "text": "What they said",
                "role": "Statement",
                "status": "Match"
                }}
            ]
            }}

            INSTRUCTIONS:
            1. Search the transcript for section numbers (like "8.4", "7.0", "3.1")
            2. Find that section in the procedure document
            3. Copy the EXACT first line/sentence as highlight_start_quote
            4. Copy the EXACT last line/sentence as highlight_end_quote
            5. List any Statement/Readback/Acknowledgement patterns from the transcript

            Return your JSON analysis now:
        """

    return system_prompt, user_prompt


def analyze_3pc_with_ollama(transcript_text: str, procedure_text: str) -> Dict[str, Any]:
    """
    Analyze the transcript and procedure text using a local Ollama-compatible model.

    NO OpenAI cloud is used here. We call your local Ollama server at:
      - LOCAL_LLM_BASE_URL (e.g. http://202.65.155.124:11434)

    Uses Ollama's native `/api/chat` endpoint (not OpenAI-compatible).
    On any connection/timeout error we log and return an empty analysis so the rest of
    the pipeline can continue without crashing.
    """
    base_url = getattr(project_settings, "LOCAL_LLM_BASE_URL", "").rstrip("/")
    model_name = getattr(
        project_settings,
        "LOCAL_LLM_MODEL",
        "qwen2.5:latest",
    )
    api_key = getattr(project_settings, "LOCAL_LLM_API_KEY", "") or ""

    # Ollama uses /api/chat endpoint, not /chat/completions
    endpoint = f"{base_url}/api/chat"

    logger.info(
        "Connecting to Ollama at: %s (Model: %s)",
        base_url,
        model_name,
    )

    system_prompt, user_prompt = _build_prompts(transcript_text, procedure_text)

    logger.info("--- LOCAL LLM REQUEST START ---")
    logger.info(f"System Prompt: {system_prompt}")
    logger.info(f"User Prompt (first 500 chars): {user_prompt[:500]}...")
    logger.info("--- LOCAL LLM REQUEST END ---")

    try:
        headers = {"Content-Type": "application/json"}
        # Only send auth header if you configured a key (skip if "EMPTY")
        if api_key and api_key.strip() and api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Ollama API format (different from OpenAI)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,  # Non-streaming response
            "format": "json",  # Ollama's JSON mode (forces JSON output)
            "options": {
                "temperature": 0.2,  # Slightly higher for more creative analysis
                "num_predict": 2000,  # Allow longer responses
                "num_ctx": 32768,  # CRITICAL: Increase context window to fit the full 36-page document
            },
        }

        # Log the request payload (use deep copy to avoid modifying original!)
        logger.info("--- OLLAMA REQUEST PAYLOAD ---")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Headers: {headers}")
        import copy
        payload_log = copy.deepcopy(payload)  # DEEP copy to avoid truncating the real payload!
        if len(user_prompt) > 500:
            payload_log["messages"][1]["content"] = user_prompt[:500] + "... [truncated for logging only]"
        logger.info(f"Payload: {json.dumps(payload_log, indent=2)}")

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)

        # Log response status and headers
        logger.info("--- OLLAMA RESPONSE STATUS ---")
        logger.info(f"Status Code: {resp.status_code}")
        logger.info(f"Response Headers: {dict(resp.headers)}")

        resp.raise_for_status()

        data = resp.json()

        # Log the FULL response from Ollama
        logger.info("--- FULL OLLAMA RESPONSE ---")
        logger.info(json.dumps(data, indent=2))

        # Ollama response format: {"message": {"role": "assistant", "content": "..."}, "done": true}
        content = data.get("message", {}).get("content", "")

        # Check if content is empty
        if not content or not content.strip():
            logger.warning("Ollama returned empty or whitespace-only content!")
            logger.warning(f"Full response structure: {list(data.keys())}")
            if "message" in data:
                logger.warning(f"Message keys: {list(data['message'].keys())}")
            return {}

        logger.info("--- EXTRACTED CONTENT FROM RESPONSE ---")
        logger.info(f"Content length: {len(content)} characters")
        logger.info(f"Content (first 1000 chars): {content[:1000]}")

        # Parse JSON safely and normalise field names
        try:
            parsed = json.loads(content)

            # Normalise possible alternate field names
            if "section_number" in parsed and "relevant_section_number" not in parsed:
                parsed["relevant_section_number"] = parsed.pop("section_number")
                logger.info("Normalised field: section_number -> relevant_section_number")

            if "3PC_interactions" in parsed and "interactions" not in parsed:
                parsed["interactions"] = parsed.pop("3PC_interactions")
                logger.info("Normalised field: 3PC_interactions -> interactions")

            # Ensure required keys exist
            parsed.setdefault("relevant_section_number", "")
            parsed.setdefault("highlight_start_quote", "")
            parsed.setdefault("highlight_end_quote", "")
            parsed.setdefault("interactions", [])

            # Log the final keys for debugging
            logger.info("Final parsed keys: %s", list(parsed.keys()))
            logger.info("--- LOCAL LLM RESPONSE END ---")
            return parsed

        except json.JSONDecodeError:
            logger.error("Local LLM response was not valid JSON: %s", content[:500])
            return {}

    except requests.RequestException as e:
        logger.error("Local LLM Analysis Failed (network/timeout): %s", e)
        # Fail soft: let the rest of the pipeline continue without LLM-driven sectioning
        return {}


def analyze_3pc_with_openai_api(transcript_text: str, procedure_text: str) -> Dict[str, Any]:
    """Analyze 3PC using an OpenAI-compatible endpoint."""

    api_key = getattr(project_settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    base_url = getattr(project_settings, "OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    model_name = getattr(project_settings, "OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        logger.warning("OpenAI API key not configured; skipping OpenAI 3PC analysis")
        return {}

    endpoint = f"{base_url}/chat/completions"
    system_prompt, user_prompt = _build_prompts(transcript_text, procedure_text)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        logger.info("--- OPENAI REQUEST START ---")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Model: {model_name}")
        logger.info(f"System Prompt: {system_prompt}")
        logger.info(f"User Prompt (first 500 chars): {user_prompt[:500]}...")
        logger.info("--- OPENAI REQUEST END ---")

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        logger.info("--- OPENAI RAW RESPONSE ---")
        logger.info(json.dumps(data, indent=2))

        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:
            logger.warning("OpenAI returned empty content for 3PC analysis")
            return {}

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.error("OpenAI response was not valid JSON: %s", content[:500])
            return {}

        if "section_number" in parsed and "relevant_section_number" not in parsed:
            parsed["relevant_section_number"] = parsed.pop("section_number")
            logger.info("Normalised field: section_number -> relevant_section_number")

        if "3PC_interactions" in parsed and "interactions" not in parsed:
            parsed["interactions"] = parsed.pop("3PC_interactions")
            logger.info("Normalised field: 3PC_interactions -> interactions")

        parsed.setdefault("relevant_section_number", "")
        parsed.setdefault("highlight_start_quote", "")
        parsed.setdefault("highlight_end_quote", "")
        parsed.setdefault("interactions", [])

        return parsed
    except requests.RequestException as exc:
        logger.error("OpenAI Analysis Failed: %s", exc)
        return {}


def analyze_3pc(transcript_text: str, procedure_text: str, provider: str = "ollama") -> Dict[str, Any]:
    """Dispatch 3PC analysis to the selected provider."""

    provider_normalized = (provider or "ollama").lower()
    if provider_normalized == "openai":
        return analyze_3pc_with_openai_api(transcript_text, procedure_text)

    # Default to Ollama/local provider
    return analyze_3pc_with_ollama(transcript_text, procedure_text)


def analyze_3pc_with_openai(transcript_text: str, procedure_text: str, provider: str = None) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper that now dispatches based on provider.
    Defaults to the local/Ollama pipeline.
    """

    return analyze_3pc(
        transcript_text,
        procedure_text,
        provider=provider or "ollama",
    )
