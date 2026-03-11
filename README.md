# 🏥 Helsana CRM Search API & Voice Agent Integration

A robust, AI-powered FastAPI service and conversational toolset built for the
Helsana CRM Hackathon.

This project solves a very specific contact-center problem: identifying a
customer in CRM data even when the caller provides incomplete, messy, or
phonetically distorted information, such as ASR output from a voice agent.

The current system is not just a fuzzy search API anymore. It is now a
**three-stage customer handling flow** designed for a voice assistant (VA)
platform:

1. **Customer lookup** using deterministic matching first, then guarded LLM
   fuzzy fallback.
2. **Identity verification** using hidden expected answers stored in private
   tool state.
3. **Post-verification intent handling** using verified CRM context and a
   tightly restricted set of supported intents.

The API is built with FastAPI, uses a lightweight in-memory CSV-backed CRM
corpus, and is designed to work cleanly with a platform that supports both
**public** and **private** tool variables.

---

## ✨ What changed from the earlier version?

The original version focused mainly on fuzzy CRM lookup and returned lookup
results and verification material in one broad response.

The current version introduces a much safer and more production-friendly
pattern:

- **Hybrid search**: exact / deterministic matching is attempted first.
- **LLM fallback**: fuzzy matching is only used when deterministic search is
  not uniquely successful.
- **Public vs private response shaping**: safe data is returned to the model,
  while hidden verification answers and rich CRM details stay private.
- **Dedicated verification flow**: verification now has its own endpoint and
  its own conversational tool behavior.
- **Verified intent unlock**: post-verification account help is only available
  after successful verification.
- **VA platform alignment**: the design explicitly fits voice / chat agents
  that can persist private tool state across turns.

---

## 🏗 Architecture Overview

The system consists of four coordinated layers:

### 1. FastAPI backend (`helsana-crm.py`)

The backend loads the Helsana CRM CSV into memory at startup, normalizes the
columns, and exposes API endpoints for:

- customer lookup,
- verification question retrieval,
- answer checking,
- post-verification summary retrieval,
- verified LLM intent context retrieval.

### 2. Deterministic matcher

The backend first attempts structured matching using normalized fields such as:

- customer ID,
- first name,
- last name,
- mobile number,
- email,
- birth date,
- address parts.

This gives the fastest and safest path when the caller provides clean data.

### 3. LLM fuzzy fallback

If deterministic lookup does not produce a unique person, the backend sends the
raw query text plus the searchable CRM corpus to OpenAI and asks for the most
likely matching `PARTNERNR` values.

This is designed to handle:

- ASR mistakes,
- broken JSON,
- inconsistent punctuation,
- phonetically spelled email addresses,
- lightly misspelled names,
- loosely formatted phone numbers.

### 4. VA platform tool layer

On top of the FastAPI service, the solution provides three conversational tools
for a voice assistant / agent platform:

- `helsana_crm_lookup`
- `helsana_verification`
- `helsana_intents`

These tools are designed for platforms that support:

- public tool output visible to the model,
- private tool output stored in session state,
- multi-turn tool orchestration.

---

## 🔐 Public vs Private Tool Variables

This is one of the most important design changes.

The VA platform can now store both **public** and **private** tool state.
That lets the system separate safe conversational context from sensitive CRM
material.

### Public data

Public data is safe for the model to use directly in the conversation.
Examples:

- lookup outcome,
- safe hit summaries,
- whether verification is required,
- the next verification question text,
- whether the caller has been verified,
- which verified intents are available.

### Private data

Private data is kept hidden from the model and used only by backend logic or
later tool calls. Examples:

- full matched CRM hit payloads,
- expected verification answers,
- post-verification account context,
- verification progress and attempt counters,
- internal tool bookkeeping.

### Why this matters

This prevents the model from accidentally seeing or revealing:

- hidden verification answers,
- sensitive account details before verification,
- internal state that should only drive backend logic.

This is especially important in a voice assistant environment where the model
must be able to continue the conversation naturally without ever leaking the
answers to security questions.

---

## 🔄 End-to-End Conversation Flow

The intended flow in the VA platform is:

1. Call `helsana_crm_lookup` when the caller provides identity details.
2. If the lookup outcome is `none` or `multi`, collect more details and search
   again.
3. If the lookup outcome is `unique`, begin verification.
4. Use `helsana_verification` to fetch the next question and check answers.
5. Only after `verified: true`, call `helsana_intents`.
6. Use the returned verified customer context to help with supported
   post-verification intents.

---

## 🚀 Quickstart

### 1. Prerequisites

Make sure you have:

- Python 3.9+
- an OpenAI API key
- the CRM CSV at `./data/helsana-crm.csv`

### 2. Install dependencies

```bash
pip install fastapi uvicorn pandas openai
```

### 3. Set environment variables

Export your OpenAI API key before starting the service.

**Mac / Linux**

```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

**Windows (Command Prompt)**

```cmd
set OPENAI_API_KEY="sk-your-openai-key-here"
```

### 4. Run the server

```bash
python3 helsana-crm.py
```

The service starts on:

```text
http://0.0.0.0:8003
```

### 5. Open the built-in FastAPI docs

FastAPI automatically serves interactive API documentation.

- Swagger UI: `http://localhost:8003/docs`
- ReDoc: `http://localhost:8003/redoc`

These built-in docs are the best live reference for:

- request bodies,
- endpoint paths,
- response shapes,
- trying endpoints manually during development.

---

## 🧠 Search and Verification Strategy

### Hybrid lookup strategy

The `/search` endpoint now uses a two-stage lookup strategy.

#### Stage 1: deterministic lookup

Structured fields are normalized and checked directly against the in-memory CRM
rows.

This includes normalization for:

- case,
- whitespace,
- phone number formatting,
- email formatting,
- date parsing,
- address field comparison.

If deterministic matching finds:

- **0 persons** → outcome is effectively unresolved
- **1 person** → success with `deterministic_unique`
- **multiple persons** → unresolved / ambiguous

#### Stage 2: LLM fallback lookup

If deterministic lookup is not uniquely successful, the backend falls back to an
LLM-powered fuzzy search using the raw request body.

This makes the service resilient to inputs like:

- `{"last_name": "tarka"}`
- `lastname müller and email is anna at gmail dot com`
- broken JSON from an upstream tool
- raw ASR transcripts

The LLM returns matched `PARTNERNR` values only, and the backend then rebuilds
family and person hits from the CRM rows.

### Verification strategy

Once a unique person is found, the backend generates up to two verification
questions from allowed CRM-derived question types.

The questions are safe in the public response, but the expected answers stay in
private state.

Verification itself uses:

1. a deterministic answer matcher for exact / normalized checks,
2. a guarded LLM fallback for minor ASR or phonetic variation.

The verifier is intentionally strict. It is designed to tolerate light speech
noise without becoming permissive across different entities.

---

## 📦 Data Preparation and In-Memory Model

At startup, the app reads the CRM CSV and normalizes it.

### Data preparation steps

- Read `./data/helsana-crm.csv`
- Normalize the first column to `FAMILY_ID`
- Fill missing values with empty strings
- Trim all cells
- Normalize a known trailing-space column naming issue for
  `ZAHLUNGSMITTEL_INKASSO`

### Searchable columns

The fuzzy search corpus is built from a lightweight subset of columns:

- `FAMILY_ID`
- `PARTNERNR`
- `last_name`
- `first_name`
- `TEL_PORTAL_MOBIL`
- `EMAIL_PORTAL`
- `GEB_D`
- `STRASSE`
- `HAUSNUMMER`
- `PLZ`
- `ORT`

These rows are serialized into `SEARCH_CORPUS_JSON` and used by the LLM fallback
search logic.

---

## 🔍 API Endpoints

The backend exposes the following endpoints.

For the most accurate live contract, always refer to the FastAPI docs at:

- `http://localhost:8003/docs`
- `http://localhost:8003/redoc`

### `POST /search`

Primary customer lookup endpoint.

#### Purpose

- Accept structured JSON or raw text
- Attempt deterministic lookup first
- Fall back to LLM fuzzy search when needed
- Return split public/private lookup results

#### Request behavior

The endpoint reads the raw request body. It supports:

- valid JSON objects,
- broken JSON,
- plain text,
- ASR transcripts.

#### Public response highlights

- `outcome`: `none`, `multi`, or `unique`
- `lookup_strategy`
- safe hit summaries
- `verification_required`
- safe verification question metadata

#### Private response highlights

- full CRM hit data
- verification questions including `expected_answers`
- post-verification summary data
- lookup strategy bookkeeping

### `POST /verification-questions`

Explicit endpoint to regenerate or fetch verification questions for a specific
person.

#### Required input

- `family_id`
- `partnernr`

#### Behavior

- Finds the exact matched person
- Builds up to `count` verification questions
- Returns split public/private question payloads

### `POST /verify-answer`

Checks a caller's answer against hidden expected answers.

#### Required input

- `question`
- `type`
- `proposed_answer`
- `expected_answers`

#### Behavior

- Uses deterministic verification first
- Falls back to guarded LLM verification if needed
- Returns whether the answer matched and by which method

### `POST /post-verification-summary`

Returns the CRM-derived summary used after successful verification.

#### Required input

- `family_id`
- `partnernr`

#### Behavior

Returns split public/private post-verification data including:

- bot-handled intents,
- current franchise context,
- current address context,
- GP / insurance model context.

### `POST /llm-post-verification-context`

Returns the full verified context package intended for downstream LLM-guided
intent handling.

#### Required input

- `family_id`
- `partnernr`

#### Behavior

Returns:

- verified status information,
- globally allowed intents,
- customer-available intents,
- customer context,
- an instruction block for downstream post-verification handling.

---

## ✅ Verification Questions

The backend generates verification questions from a controlled bank of allowed
question types.

Examples include:

- franchise,
- payment method,
- basic insurance model,
- insured products,
- payout bank,
- total premium,
- GP name for applicable insurance models.

Questions are only included when the corresponding CRM data is actually present
and valid for that person.

The public tool response exposes only safe metadata such as:

- tag,
- field,
- question text,
- question type.

The private tool response additionally includes:

- `expected_answers`

These expected answers must never be exposed to the caller.

---

## 🧰 VA Platform Integration

This project is built to plug into a conversational platform where a model can
call tools and the platform can persist both public and private state.

The three main tools are described below.

### 1. `helsana_crm_lookup`

This tool identifies the caller from available identity data.

#### When to use it

Use it when:

- the caller introduces themselves,
- the caller provides personal details,
- you need to locate the CRM record,
- you need to refine a `none` or `multi` result.

#### What it returns

**Public**

- success status,
- search outcome,
- safe hit data,
- verification requirement,
- safe verification question metadata.

**Private**

- full private CRM hits,
- verification questions with expected answers,
- post-verification summary,
- original search payload,
- source tool metadata.

#### Important model rule

A `unique` lookup result means the caller is identified well enough to begin
verification, but **not yet verified**.

---

### 2. `helsana_verification`

This tool manages caller verification after a unique lookup.

#### Supported actions

- `get_question`
- `check_answer`

#### What it does

- retrieves the next verification question,
- checks the caller's answer using hidden expected answers,
- tracks attempts and failures,
- decides whether verification succeeds or fails.

#### State it manages

In session state, it maintains:

- whether verification has started,
- whether the caller is verified,
- whether verification has failed,
- current question index,
- attempts per question,
- total failures,
- verification history.

#### Safety rules

- Ask one question at a time.
- Never reveal expected answers.
- Never help the caller guess.
- Treat the caller as verified only when the tool returns `verified: true`.

---

### 3. `helsana_intents`

This tool is the post-verification unlock point.

#### When to use it

Use it only after verification has succeeded.

#### Purpose

It retrieves the verified caller's CRM context and tells the model exactly which
intents are globally supported and which are available for this customer.

#### Globally supported intents

- `FRANCHISE_CHANGE`
- `ADDRESS_CHANGE`
- `GP_CHANGE`

#### Important rules

- Any request outside those three intents is unsupported.
- Even within those three, only intents with `available: true` may be handled.
- The returned customer context can be used to answer current-account questions
  related to those supported intents.

---

## 🪜 Recommended Voice Agent Orchestration

A typical voice agent flow looks like this:

### Step 1: identify the caller

Call `helsana_crm_lookup` with whatever identity data is available.

- If `outcome = none`, ask for more identifying information.
- If `outcome = multi`, ask for more identifying information.
- If `outcome = unique`, move to verification.

### Step 2: verify the caller

Use `helsana_verification` with `action: "get_question"`.

Ask that one question to the caller.

When the caller answers, call `helsana_verification` with:

- `action: "check_answer"`
- `answer: "..."`

Repeat until the tool returns either:

- `verified: true`, or
- `status: failed`

### Step 3: unlock account help

Once verified, call `helsana_intents` with the caller's current request.

Use the returned context to:

- answer current state questions,
- handle supported intents,
- refuse unsupported requests safely.

---

## 🧪 Example Lookup Output

A typical successful unique lookup now returns a split response like this:

```json
{
  "public": {
    "outcome": "unique",
    "lookup_strategy": "deterministic_unique",
    "hits": [
      {
        "family_id": "123456",
        "person_count": 1,
        "members": [
          {
            "family_id": "123456",
            "partnernr": "123456",
            "display_name": "Alan Blessing"
          }
        ]
      }
    ],
    "verification_required": true,
    "verification_questions": [
      {
        "tag": "FRANCHISE",
        "field": "FRANCHISE",
        "question": "What is your selected franchise?",
        "type": "NUMBER"
      },
      {
        "tag": "PRODUKT",
        "field": "PRODUKT",
        "question": "Can you name at least one of your insured products?",
        "type": "TEXT"
      }
    ]
  },
  "private": {
    "hits": ["...full private CRM hit data..."],
    "verification_questions": [
      {
        "tag": "FRANCHISE",
        "field": "FRANCHISE",
        "question": "What is your selected franchise?",
        "type": "NUMBER",
        "expected_answers": ["1500.0"]
      }
    ],
    "post_verification_summary": {
      "bot_handled_intents": [
        "FRANCHISE_CHANGE",
        "ADDRESS_CHANGE",
        "GP_CHANGE"
      ]
    },
    "lookup_strategy": "deterministic_unique"
  }
}
```

This split shape is central to the new design.

---

## 🧪 Testing with cURL

### Basic structured lookup

```bash
curl -X POST "http://localhost:8003/search" \
  -H "Content-Type: application/json" \
  -d '{"last_name":"Blessing","first_name":"Alan"}' | python3 -m json.tool
```

### Fuzzy lookup with partial data

```bash
curl -X POST "http://localhost:8003/search" \
  -H "Content-Type: application/json" \
  -d '{"last_name":"tarka"}' | python3 -m json.tool
```

### Raw text / ASR-style lookup

```bash
curl -X POST "http://localhost:8003/search" \
  -H "Content-Type: text/plain" \
  --data 'my last name is müller and my email is anna at gmail dot com' \
  | python3 -m json.tool
```

### Explicit verification question retrieval

```bash
curl -X POST "http://localhost:8003/verification-questions" \
  -H "Content-Type: application/json" \
  -d '{"family_id":"123456","partnernr":"123456","count":2}' \
  | python3 -m json.tool
```

### Verification answer check

```bash
curl -X POST "http://localhost:8003/verify-answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question":"What is your selected franchise?",
    "type":"NUMBER",
    "proposed_answer":"1500",
    "expected_answers":["1500.0"]
  }' | python3 -m json.tool
```

### Post-verification summary

```bash
curl -X POST "http://localhost:8003/post-verification-summary" \
  -H "Content-Type: application/json" \
  -d '{"family_id":"123456","partnernr":"123456"}' \
  | python3 -m json.tool
```

### LLM-ready verified intent context

```bash
curl -X POST "http://localhost:8003/llm-post-verification-context" \
  -H "Content-Type: application/json" \
  -d '{"family_id":"123456","partnernr":"123456"}' \
  | python3 -m json.tool
```

---

## 🛡 Security and Safety Notes

This solution is intentionally designed to reduce leakage risk in a
conversational setting.

### Before verification

Before successful verification, the model should only use public lookup data.
It must not reveal:

- hidden expected answers,
- detailed private CRM fields,
- post-verification customer context.

### During verification

The model must:

- ask one question at a time,
- avoid hints,
- avoid multiple-choice guidance,
- avoid confirming near misses,
- fail safely when the tool says verification failed.

### After verification

Even after verification, the model is still constrained to the supported intent
set and the customer's available service context.

---

## 📁 Project Structure

A minimal project layout could look like this:

```text
.
├── data/
│   └── helsana-crm.csv
├── helsana-crm.py
└── README.md
```

---

## 🧭 Why this design fits a VA platform

Traditional REST integrations often assume that either:

- the model sees everything, or
- the backend handles everything without conversational state.

A real voice assistant platform sits in between. It needs:

- multi-turn state,
- tool orchestration,
- safe model-visible context,
- hidden operational data,
- strict security boundaries.

This project is built exactly for that environment.

It gives the model enough information to have a natural conversation, while the
platform and backend retain control over sensitive verification material and
verified account context.

That makes it a strong pattern for:

- customer identification,
- secure caller verification,
- controlled post-verification service flows.

---

## 📖 Source of truth during development

For implementation details, use both of these references together:

1. **This README** for the architecture, flow, and VA platform integration.
2. **FastAPI built-in docs** at `/docs` and `/redoc` for the exact live API
   contract.

The README explains the why and how.
The FastAPI docs show the real running endpoint schema.

---

## 🎯 Summary

The current Helsana CRM solution is a hybrid deterministic + LLM customer lookup
and verification service tailored for voice agents.

It provides:

- fuzzy CRM identification,
- strict public/private tool data separation,
- secure multi-step verification,
- verified customer context retrieval,
- controlled post-verification intent handling.

In short: **identify, verify, unlock, then help safely**.
