# 🏥 Helsana CRM: Identity & Intent API

An AI-augmented CRM middleware built for customer
service identification. This system bridges the gap between messy, real-world inputs (like ASR
transcripts) and secure, intent-based action handling for Helsana insurance.

## 🏗 System Architecture

The API acts as a secure state machine with three distinct phases:

1. **Identification**: Hybrid Exact + LLM Fuzzy matching to find a partner.
2. **Verification**: A challenge-response system using private CRM data.
3. **Intent Injection**: Providing the LLM with verified context and strict
guardrails for specific business processes (Franchise, Address, GP changes).

---

## 🚀 Quickstart

### 1. Prerequisites

* **Python 3.9+**
* **OpenAI API Key** (configured as `OPENAI_API_KEY`)
* **CRM Data**: Ensure `data/helsana-crm.csv` exists in the root.

### 2. Installation & Execution

```bash
pip install fastapi uvicorn pandas openai pydantic
python3 helsana-crm.py

```

*Server starts on `http://0.0.0.0:8003`.*

> **📖 Interactive Docs:** Access the full OpenAPI specification and test
> endpoints directly at: `http://localhost:8003/docs`

---

## 🔍 Core Endpoints

### `POST /search`

The primary entry point. It first attempts a **Deterministic Match** (exact
strings). If no unique result is found, it falls back to a **GPT-5-Mini Fuzzy
Lookup** to handle phonetic misspellings or fragmented data.

**Sample Request:**

```bash
curl -X POST "http://localhost:8003/search" \
    -H "Content-Type: application/json" \
    -d '{"last_name": "tarka", "first_name": "alan"}' | python3 -m json.tool --indent 2

```

### `POST /verify-answer`

Used to validate user responses. It uses a **Guarded LLM Fallback** to be robust
against ASR noise (e.g., "Eco" vs "Echo") while remaining strictly accurate for
numerical data like premiums or franchises.

### `POST /llm-post-verification-context`

The "Unlock" endpoint. Once a user is verified, this returns a system prompt and
data context that restricts the AI agent to only three supported intents:
`FRANCHISE_CHANGE`, `ADDRESS_CHANGE`, and `GP_CHANGE`.

---

## 🛠 VA Platform Integration

The system is designed for Virtual Assistant (VA) platforms that support
**Public/Private state separation**.

### 1. Tool: `helsana_crm_lookup`

* **Purpose**: Identify the caller.
* **Logic**: Sends identity signals to `/search`.
* **Security**: Returns **Public** hits (display names) to the LLM, while
storing **Private** data (expected answers) in the session state.

### 2. Tool: `helsana_verification`

* **Purpose**: Authenticate the caller.
* **Logic**:
* `get_question`: Fetches the next challenge (e.g., "What is your GP?").
* `check_answer`: Forwards the spoken answer to `/verify-answer`.


* **State**: Tracks `total_failures`. If 3 attempts fail, the session is
flagged as `failed`, preventing further access.

### 3. Tool: `helsana_intents`

* **Purpose**: Contextualize the verified session.
* **Logic**: Only callable if `verified: true`. It injects the `llm_instruction`
into the conversation, turning the bot into "Sana," a Concierge who knows
the user's current franchise and GP details.

---

## 🧪 Testing with cURL

Test the full logic flow by mocking a unique identification:

```bash
# 1. Search for a customer
curl -s -X POST "http://localhost:8003/search" \
     -d '{"last_name": "Blessing"}' \
     -H "Content-Type: application/json" | python3 -m json.tool --indent 2

# 2. Verify an answer (Case/ASR insensitive)
curl -s -X POST "http://localhost:8003/verify-answer" \
     -d '{
          "question": "What is your selected franchise?",
          "type": "NUMBER",
          "proposed_answer": "1500",
          "expected_answers": ["1500.0"]
         }' \
     -H "Content-Type: application/json" | python3 -m json.tool --indent 2

```
