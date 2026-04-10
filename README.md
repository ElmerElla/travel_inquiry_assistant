# Travel Inquiry Assistant (LangChain + FastAPI)

Small prototype that accepts a raw travel inquiry and returns structured fields plus a concise reply.

## Tech stack
- Python
- FastAPI
- LangChain
- OpenAI-compatible API (via langchain-openai)

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the env file and set your API key and base URL:
   ```bash
   copy .env.example .env
   ```
   Then edit `.env` and set `OPENAI_KEY` and `OPENAI_API_BASE`.

## Run the API
```bash
uvicorn app.main:app --reload
```

## Example request
```bash
curl -X POST http://127.0.0.1:8000/inquiry \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Hi, I want to plan a 7-day trip to Japan in October for 2 people. Budget is around 3000 USD.\"}"
```

## Response format
The API returns JSON with extracted fields, `missing_fields`, `reply_type`, and a short `reply`.

## Design notes
- A LangChain LLM step extracts structured fields.
- Missing critical fields (destination, travel_time, group_size, budget) are reported.
- The reply is generated from extracted fields and missing fields to stay consistent.
- A small regex-based fallback is used if the LLM call fails.

## Assumptions and limitations
- Extraction relies on the LLM; results can vary by model.
- Fallback rules are intentionally lightweight.
- This is a minimal prototype, not production-ready.
