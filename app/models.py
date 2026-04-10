from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class InquiryRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Raw customer inquiry")


class InquiryFields(BaseModel):
    destination: Optional[str] = None
    trip_type: Optional[str] = None
    duration: Optional[str] = None
    travel_time: Optional[str] = None
    group_size: Optional[int] = None
    budget: Optional[float] = None
    intent: Optional[str] = None


class InquiryResponse(BaseModel):
    destination: Optional[str] = None
    trip_type: Optional[str] = None
    duration: Optional[str] = None
    travel_time: Optional[str] = None
    group_size: Optional[int] = None
    budget: Optional[float] = None
    intent: Optional[str] = None
    missing_fields: List[str]
    reply_type: Literal["initial_response", "clarifying_question"]
    reply: str
