from __future__ import annotations

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from .models import InquiryRequest, InquiryResponse
from .service import ServiceConfigError, process_inquiry

load_dotenv()

app = FastAPI(title="Travel Inquiry Assistant", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/inquiry", response_model=InquiryResponse)
def inquiry(payload: InquiryRequest) -> InquiryResponse:
    try:
        return process_inquiry(payload.message)
    except ServiceConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
