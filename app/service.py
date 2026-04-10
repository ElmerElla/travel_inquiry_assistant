from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import InquiryFields, InquiryResponse
from .prompts import EXTRACTION_SYSTEM_PROMPT, REPLY_SYSTEM_PROMPT


REQUIRED_FIELDS = ["destination", "travel_time", "group_size", "budget"]


class ServiceConfigError(RuntimeError):
    pass


@dataclass
class ParsedResult:
    fields: InquiryFields
    missing_fields: List[str]


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ServiceConfigError("OPENAI_KEY (or OPENAI_API_KEY) is not set.")
    openai_api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    if not openai_api_base:
        raise ServiceConfigError("OPENAI_API_BASE (or OPENAI_BASE_URL) is not set.")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=openai_api_base,
    )


def process_inquiry(message: str) -> InquiryResponse:
    parsed = _extract_fields(message)
    reply_type = "clarifying_question" if parsed.missing_fields else "initial_response"
    reply = _generate_reply(message, parsed.fields, parsed.missing_fields)

    return InquiryResponse(
        **parsed.fields.model_dump(),
        missing_fields=parsed.missing_fields,
        reply_type=reply_type,
        reply=reply,
    )


def _extract_fields(message: str) -> ParsedResult:
    try:
        fields = _extract_with_llm(message)
    except Exception:
        fields = _extract_with_fallback(message)

    fields = _normalize_fields(fields)
    missing_fields = _get_missing_fields(fields)
    return ParsedResult(fields=fields, missing_fields=missing_fields)


def _extract_with_llm(message: str) -> InquiryFields:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", "Message: {message}"),
        ]
    )
    structured_llm = llm.with_structured_output(InquiryFields)
    chain = prompt | structured_llm
    return chain.invoke({"message": message})


def _extract_with_fallback(message: str) -> InquiryFields:
    lower = message.lower()
    destination = _match_destination(message)
    duration = _match_duration(lower)
    travel_time = _match_travel_time(message)
    group_size = _match_group_size(lower)
    budget = _match_budget(lower)
    trip_type = _match_trip_type(lower)
    intent = _match_intent(lower)

    return InquiryFields(
        destination=destination,
        trip_type=trip_type,
        duration=duration,
        travel_time=travel_time,
        group_size=group_size,
        budget=budget,
        intent=intent,
    )


def _normalize_fields(fields: InquiryFields) -> InquiryFields:
    cleaned = fields.model_dump()
    for key, value in cleaned.items():
        if isinstance(value, str):
            text = value.strip()
            if text.lower() in {"", "unknown", "n/a", "na", "none"}:
                cleaned[key] = None
            else:
                cleaned[key] = text

    if cleaned.get("group_size") is not None and cleaned["group_size"] <= 0:
        cleaned["group_size"] = None
    if cleaned.get("budget") is not None and cleaned["budget"] <= 0:
        cleaned["budget"] = None

    return InquiryFields(**cleaned)


def _get_missing_fields(fields: InquiryFields) -> List[str]:
    missing = []
    for name in REQUIRED_FIELDS:
        value = getattr(fields, name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    return missing


def _generate_reply(message: str, fields: InquiryFields, missing_fields: List[str]) -> str:
    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REPLY_SYSTEM_PROMPT),
                (
                    "human",
                    "Fields: {fields_json}\nMissing: {missing_json}",
                ),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        reply = chain.invoke(
            {
                "fields_json": json.dumps(fields.model_dump()),
                "missing_json": json.dumps(missing_fields),
            }
        )
        return reply.strip()
    except Exception:
        return _fallback_reply(fields, missing_fields)


def _fallback_reply(fields: InquiryFields, missing_fields: List[str]) -> str:
    if missing_fields:
        missing = ", ".join(_friendly_missing_fields(missing_fields))
        detail_text = _summarize_details(fields)
        if detail_text:
            return (
                f"Thanks for your message about {detail_text}. "
                "To help you better, could you share the "
                f"{missing}?"
            )
        return (
            "Thanks for your message. To help you better, could you share the "
            f"{missing}?"
        )

    detail_text = _summarize_details(fields) or "your trip"
    return (
        f"Thanks for your message. We can help with {detail_text}. "
        "What preferences or priorities should we consider?"
    )


def _match_budget(text: str) -> Optional[float]:
    match = re.search(r"(\d{2,}(?:,\d{3})*(?:\.\d+)?)\s*(usd|dollars|\$)", text)
    if not match:
        return None
    value = match.group(1).replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def _match_group_size(text: str) -> Optional[int]:
    match = re.search(r"(?:for|party of|group of)\s+(\d+)", text)
    if not match:
        match = re.search(r"(\d+)\s*(people|travelers|persons|pax)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _match_duration(text: str) -> Optional[str]:
    match = re.search(r"(\d+)\s*(day|days|night|nights|week|weeks)", text)
    if not match:
        return None
    value, unit = match.groups()
    return f"{value} {unit}"


def _match_travel_time(text: str) -> Optional[str]:
    month_match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)",
        text,
        flags=re.IGNORECASE,
    )
    if month_match:
        return month_match.group(1).title()

    relative_match = re.search(r"(next month|this month|next year|this year)", text, re.I)
    if relative_match:
        return relative_match.group(1).lower()

    return None


def _match_destination(text: str) -> Optional[str]:
    match = re.search(r"\bto\s+([A-Z][A-Za-z\s]+)", text)
    if not match:
        match = re.search(r"\bin\s+([A-Z][A-Za-z\s]+)", text)
    if not match:
        return None
    return match.group(1).strip()


def _match_trip_type(text: str) -> Optional[str]:
    if "honeymoon" in text:
        return "honeymoon"
    if "family" in text:
        return "family-friendly"
    if "visa" in text:
        return "visa assistance"
    if "beach" in text:
        return "beach vacation"
    if "cheap" in text or "budget" in text:
        return "budget travel"
    return None


def _match_intent(text: str) -> Optional[str]:
    if "price" in text or "cost" in text:
        return "pricing inquiry"
    if "contact" in text or "call" in text:
        return "contact request"
    if "visa" in text:
        return "visa assistance"
    if "book" in text:
        return "booking inquiry"
    return "trip planning"


def _friendly_missing_fields(missing_fields: List[str]) -> List[str]:
    friendly_map = {
        "destination": "destination",
        "travel_time": "travel time",
        "group_size": "group size",
        "budget": "budget",
    }
    return [friendly_map.get(name, name.replace("_", " ")) for name in missing_fields]


def _summarize_details(fields: InquiryFields) -> str:
    details = []
    if fields.destination:
        details.append(f"a trip to {fields.destination}")
    if fields.duration:
        details.append(fields.duration)
    if fields.travel_time:
        details.append(f"in {fields.travel_time}")
    if fields.group_size:
        details.append(f"for {fields.group_size} travelers")
    if fields.budget:
        details.append(f"with a budget around {fields.budget}")

    return ", ".join(details)
