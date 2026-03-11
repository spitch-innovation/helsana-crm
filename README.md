# Helsana CRM Search API and VA Tooling

This repository contains a FastAPI service and three frontend tool handlers for
use in a voice assistant or chat assistant platform that supports both public
and private tool state.

The system has three stages:

1. Customer lookup
2. Identity verification
3. Post-verification intent handling

The backend reads CRM data from `./data/helsana-crm.csv`, keeps it in memory,
and exposes endpoints for lookup, verification, and verified customer context.

## What the system does

The service is built for caller identification from imperfect data.

Lookup supports:

- exact and normalized matching for structured fields
- fallback fuzzy matching through an LLM when deterministic lookup is not
  unique
- safe public responses for the model
- private responses containing hidden verification answers and full CRM context

Verification supports:

- one question at a time
- answer checking against hidden expected answers
- deterministic fast-path checking first
- guarded LLM fallback for ASR noise and minor spelling variation
- attempt tracking in private session state

Post-verification intent handling supports only these intents:

- `FRANCHISE_CHANGE`
- `ADDRESS_CHANGE`
- `GP_CHANGE`

Anything outside those three intents is unsupported.

## FastAPI app

Main file:

```python
python3 helsana-crm.py
```

The server runs on:

```text
http://0.0.0.0:8003
```

Built-in API docs:

```text
http://localhost:8003/docs
http://localhost:8003/redoc
```

Use the built-in FastAPI docs as the live reference for endpoint schemas,
request bodies, and example calls.

## Installation

Requirements:

- Python 3.9+
- OpenAI API key
- CRM CSV at `./data/helsana-crm.csv`

Install dependencies:

```bash
pip install fastapi uvicorn pandas openai
```

Set the API key.

macOS / Linux:

```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

Windows cmd:

```cmd
set OPENAI_API_KEY="sk-your-openai-key-here"
```

Start the service:

```bash
python3 helsana-crm.py
```

## Backend behavior

### Data loading and normalization

On startup, the backend:

- loads `./data/helsana-crm.csv`
- normalizes the first column to `FAMILY_ID`
- strips whitespace from all fields
- fills missing values with empty strings
- normalizes a known trailing-space column name for
  `ZAHLUNGSMITTEL_INKASSO`
- builds an in-memory search corpus from selected searchable fields

### Lookup strategy

`/search` works in two stages.

First it tries deterministic matching using structured fields such as:

- `customer_id`
- `first_name`
- `last_name`
- `tel_portal_mobil`
- `email_portal`
- `birthday`
- `address.strasse`
- `address.hausnummer`
- `address.plz`

Normalization includes:

- lowercasing and whitespace cleanup for text
- digit-only phone comparison
- relaxed email normalization for spoken variants such as `at`, `ät`, `dot`,
  and `punkt`
- date normalization through pandas

If deterministic lookup does not return a unique person, the backend sends the
raw request text plus the searchable CRM corpus to the LLM and asks for matched
`PARTNERNR` values.

### Public and private response split

The backend returns a split response:

- `public`: safe for the model to see and use in conversation
- `private`: hidden state for later verification and post-verification steps

This is the core integration pattern for the VA platform.

Public lookup data includes:

- `outcome`
- `lookup_strategy`
- safe hit summaries
- `verification_required`
- verification questions without expected answers

Private lookup data includes:

- full matched hits
- verification questions with `expected_answers`
- post-verification summary
- lookup strategy

## Endpoints

### `POST /search`

Accepts either structured JSON or raw text. Returns public and private lookup
results.

Typical outcomes:

- `none`
- `multi`
- `unique`

If the outcome is `unique`, the backend also generates up to two verification
questions for that person.

Example request:

```bash
curl -X POST "http://localhost:8003/search" \
  -H "Content-Type: application/json" \
  -d '{"last_name":"Blessing","first_name":"Alan"}'
```

Example public response:

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
  }
}
```

The private response also contains full CRM hits and hidden
`expected_answers`.

### `POST /verification-questions`

Returns verification questions for a specific `family_id` and `partnernr`.

Request body:

```json
{
  "family_id": "123456",
  "partnernr": "123456",
  "count": 2
}
```

Use this endpoint directly only if you need explicit question retrieval outside
of the standard lookup flow.

### `POST /verify-answer`

Checks one verification answer against a provided set of expected answers.

Request body:

```json
{
  "question": "What is your selected franchise?",
  "type": "NUMBER",
  "proposed_answer": "1500",
  "expected_answers": ["1500.0"]
}
```

Example response:

```json
{
  "matched": true,
  "method": "deterministic",
  "matched_expected_answer": "1500.0"
}
```

The endpoint first tries deterministic verification. If that fails, it uses a
strict LLM fallback for minor ASR or spelling variation.

### `POST /post-verification-summary`

Returns post-verification summary data for a verified person.

Request body:

```json
{
  "family_id": "123456",
  "partnernr": "123456"
}
```

### `POST /llm-post-verification-context`

Returns the verified customer context and instruction block used by the VA
platform after successful verification.

Request body:

```json
{
  "family_id": "123456",
  "partnernr": "123456"
}
```

The public response contains:

- `verified: true`
- globally supported intents
- customer-available intents
- customer context for those intents
- an instruction block for the assistant

## VA platform usage

The intended frontend integration uses three tools:

1. `helsana_crm_lookup`
2. `helsana_verification`
3. `helsana_intents`

The platform must support:

- public tool output visible to the model
- private tool output stored in session state
- reading private state from previous tool calls

### Conversation flow

1. Run lookup with any caller identity details you have.
2. If lookup is `none` or `multi`, collect more details and run lookup again.
3. If lookup is `unique`, start verification.
4. Ask one verification question at a time.
5. Check each answer through the verification tool.
6. Only after `verified: true`, load intent context.
7. Handle only supported and available intents.

## Frontend tool definition: `helsana_crm_lookup`

### Title

```text
title: helsana_crm_lookup
```

### Instructions

```text
Use this tool to identify a Helsana caller in the CRM using whatever identity
 details the caller provides.

This tool is for customer lookup only. Collect any available identifying
information from the caller, such as customer ID, first name, last name,
mobile number, email address, date of birth, or address details, and send
those values to the CRM search service.

Use this tool when:

* the caller introduces themselves,
* the caller provides personal details,
* you need to locate the correct CRM record,
* you need to narrow down none / multiple / unique customer matches.

Important behavior:

* Send whatever identifying fields you have; you do not need every field.
* Use it again if the first search returns no match or multiple matches and
  the caller provides more details.
* A unique result means the caller is identified well enough to begin
  verification, but not yet verified.
* Do not treat the caller as verified just because the tool returns a unique
  result.
* Verification must be handled in later steps using dedicated verification
  logic.
* Do not reveal sensitive account details before verification succeeds.

Tool response behavior:

* The tool returns public lookup data to the model, such as:

  * search outcome,
  * safe hit information,
  * whether verification is required.
* The tool may also store private CRM data in session state for later
  verification steps.
* The model must rely only on the public tool response and must never assume
  access to hidden verification answers or post-verification CRM details.
```

### Params

```json
{
  "type": "object",
  "properties": {
    "address": {
      "type": "object",
      "properties": {
        "plz": {
          "type": "string",
          "description": "Postal code."
        },
        "strasse": {
          "type": "string",
          "description": "Street name."
        },
        "hausnummer": {
          "type": "string",
          "description": "House number."
        }
      },
      "description": "Caller's address details, if provided.",
      "additionalProperties": false
    },
    "birthday": {
      "type": "string",
      "description": "Caller's date of birth."
    },
    "last_name": {
      "type": "string",
      "description": "Caller's last name."
    },
    "first_name": {
      "type": "string",
      "description": "Caller's first name."
    },
    "customer_id": {
      "type": "string",
      "description": "Caller's customer ID if provided."
    },
    "email_portal": {
      "type": "string",
      "description": "Caller's email address."
    },
    "tel_portal_mobil": {
      "type": "string",
      "description": "Caller's mobile phone number."
    }
  },
  "additionalProperties": false
}
```

### Backend handler

```javascript
async (args) => {
  const FASTAPI_BASE_URL = "http://127.0.0.1:8003";

  const cleanString = (value) => {
    if (value === null || value === undefined) return "";
    return String(value).trim();
  };

  const cleanObject = (obj) => {
    const out = {};
    for (const [key, value] of Object.entries(obj || {})) {
      if (value === null || value === undefined) continue;

      if (typeof value === "string") {
        const trimmed = value.trim();
        if (trimmed !== "") out[key] = trimmed;
        continue;
      }

      if (typeof value === "object" && !Array.isArray(value)) {
        const nested = cleanObject(value);
        if (Object.keys(nested).length > 0) {
          out[key] = nested;
        }
        continue;
      }

      out[key] = value;
    }
    return out;
  };

  try {
    const payload = cleanObject({
      customer_id: cleanString(args.customer_id),
      first_name: cleanString(args.first_name),
      last_name: cleanString(args.last_name),
      tel_portal_mobil: cleanString(args.tel_portal_mobil),
      email_portal: cleanString(args.email_portal),
      birthday: cleanString(args.birthday),
      address: args.address
        ? {
          strasse: cleanString(args.address.strasse),
          hausnummer: cleanString(args.address.hausnummer),
          plz: cleanString(args.address.plz)
        }
        : undefined
    });

    if (Object.keys(payload).length === 0) {
      return JSON.stringify({
        public: {
          success: false,
          error: "At least one customer identity field must be provided."
        }
      });
    }

    const response = await fetch(`${FASTAPI_BASE_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const text = await response.text();

    if (!response.ok) {
      return JSON.stringify({
        public: {
          success: false,
          error: `CRM lookup failed with status ${response.status}`,
          details: text
        }
      });
    }

    let result;
    try {
      result = JSON.parse(text);
    } catch (parseError) {
      return JSON.stringify({
        public: {
          success: false,
          error: "CRM lookup returned invalid JSON.",
          details: text
        }
      });
    }

    return JSON.stringify({
      public: {
        success: true,
        ...(result.public || {})
      },
      private: {
        source_tool: "helsana_crm_lookup",
        search_payload: payload,
        ...(result.private || {})
      }
    });
  } catch (error) {
    return JSON.stringify({
      public: {
        success: false,
        error: error?.message || String(error)
      }
    });
  }
}
```

## Frontend tool definition: `helsana_verification`

### Name

```text
name: helsana_verification
```

### Instructions

```text
Use this tool to manage caller identity verification after a unique CRM match
has already been found.

This tool works from the caller's saved private CRM lookup state and supports
 two actions:

* get_question: return the next verification question to ask
* check_answer: verify the caller's answer against the hidden expected answers
  stored in private session state

Use this tool when:

* the CRM lookup has returned a unique customer match,
* you need to ask the next verification question,
* the caller has answered a verification question and you need to check it,
* you need to continue or fail the verification flow safely.

Important behavior:

* Use this tool only after a unique customer match has been found.
* Never assume the caller is verified until this tool explicitly returns
  verified: true.
* Ask only one verification question at a time.
* Do not reveal expected answers, hints, or hidden CRM data.
* If the caller gives a wrong answer, do not help them guess.
* If verification fails, explain that secure verification could not be
  completed and move to the fallback process.

Tool response behavior:

* The tool returns only safe public verification data to the model.
* The tool uses hidden private CRM lookup data already stored in session
  state.
* The tool manages verification progress, question order, attempts, and final
  verification result.
```

### Params

```json
{
  "type": "object",
  "required": [
    "action"
  ],
  "properties": {
    "action": {
      "enum": [
        "get_question",
        "check_answer"
      ],
      "type": "string",
      "description": "Whether to fetch the next verification question or check the caller's answer."
    },
    "answer": {
      "type": "string",
      "description": "The caller's spoken answer to the current verification question. Only use this for check_answer."
    }
  },
  "additionalProperties": false
}
```

### Backend handler

```javascript
async (args, sessionContext) => {
  const FASTAPI_BASE_URL = "http://127.0.0.1:8003";

  const getLookupState = () => {
    const state = sessionContext.getPrivateToolState
      ? sessionContext.getPrivateToolState("helsana_crm_lookup")
      : null;
    return state || null;
  };

  const getVerificationBucket = () => {
    const root = sessionContext.sessionState || {};
    if (!root.helsana_verification) {
      root.helsana_verification = {
        started: false,
        verified: false,
        failed: false,
        currentIndex: 0,
        attemptsByTag: {},
        maxAttemptsPerQuestion: 2,
        maxTotalFailures: 3,
        totalFailures: 0,
        history: []
      };
    }
    return root.helsana_verification;
  };

  const buildPublicError = (message) => ({
    public: {
      success: false,
      error: message
    }
  });

  const safeQuestionPublic = (question) => ({
    tag: question.tag,
    field: question.field,
    type: question.type,
    text: question.question
  });

  try {
    const lookupState = getLookupState();
    if (!lookupState) {
      return JSON.stringify(buildPublicError(
        "No CRM lookup state found. Please identify the caller first."
      ));
    }

    const questions = Array.isArray(lookupState.verification_questions)
      ? lookupState.verification_questions
      : [];

    if (!questions.length) {
      return JSON.stringify(buildPublicError(
        "No verification questions are available for this caller."
      ));
    }

    const verification = getVerificationBucket();

    if (verification.verified) {
      return JSON.stringify({
        public: {
          success: true,
          verified: true,
          status: "verified",
          message: "The caller has already been verified."
        },
        private: verification
      });
    }

    if (verification.failed) {
      return JSON.stringify({
        public: {
          success: true,
          verified: false,
          status: "failed",
          message: "Verification has already failed."
        },
        private: verification
      });
    }

    if (args.action === "get_question") {
      const question = questions[verification.currentIndex];

      if (!question) {
        return JSON.stringify(buildPublicError(
          "No further verification question is available."
        ));
      }

      verification.started = true;

      return JSON.stringify({
        public: {
          success: true,
          verified: false,
          status: "question",
          question: safeQuestionPublic(question)
        },
        private: verification
      });
    }

    if (args.action === "check_answer") {
      const question = questions[verification.currentIndex];
      if (!question) {
        return JSON.stringify(buildPublicError(
          "No active verification question is available."
        ));
      }

      const answer = String(args.answer || "").trim();
      if (!answer) {
        return JSON.stringify(buildPublicError(
          "A verification answer is required."
        ));
      }

      const expectedAnswers = Array.isArray(question.expected_answers)
        ? question.expected_answers.filter((x) => String(x || "").trim())
        : [];

      if (!expectedAnswers.length) {
        return JSON.stringify(buildPublicError(
          "No expected answers are available for the current verification question."
        ));
      }

      const verifyResponse = await fetch(`${FASTAPI_BASE_URL}/verify-answer`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: question.question,
          type: question.type,
          proposed_answer: answer,
          expected_answers: expectedAnswers
        })
      });

      const verifyText = await verifyResponse.text();

      if (!verifyResponse.ok) {
        return JSON.stringify({
          public: {
            success: false,
            error: `Verification check failed with status ${verifyResponse.status}`,
            details: verifyText
          }
        });
      }

      let verifyResult;
      try {
        verifyResult = JSON.parse(verifyText);
      } catch (parseError) {
        return JSON.stringify({
          public: {
            success: false,
            error: "Verification check returned invalid JSON.",
            details: verifyText
          }
        });
      }

      const matched = !!verifyResult.matched;

      verification.attemptsByTag[question.tag] =
        (verification.attemptsByTag[question.tag] || 0) + 1;
      verification.history.push({
        tag: question.tag,
        answer,
        matched,
        method: verifyResult.method || "unknown",
        timestamp: new Date().toISOString()
      });

      if (!matched) {
        verification.totalFailures += 1;

        const questionAttempts = verification.attemptsByTag[question.tag];
        const tooManyForQuestion =
          questionAttempts >= verification.maxAttemptsPerQuestion;
        const tooManyTotal =
          verification.totalFailures >= verification.maxTotalFailures;

        if (tooManyForQuestion || tooManyTotal) {
          verification.failed = true;

          return JSON.stringify({
            public: {
              success: true,
              verified: false,
              status: "failed",
              message: "I’m sorry, I’m not able to verify your identity on this call."
            },
            private: verification
          });
        }

        return JSON.stringify({
          public: {
            success: true,
            verified: false,
            status: "incorrect",
            message: "That didn’t match. Please try again.",
            question: safeQuestionPublic(question)
          },
          private: verification
        });
      }

      const nextIndex = verification.currentIndex + 1;

      if (nextIndex >= questions.length) {
        verification.verified = true;
        verification.currentIndex = nextIndex;

        return JSON.stringify({
          public: {
            success: true,
            verified: true,
            status: "verified",
            message: "Thank you. Your identity has been verified."
          },
          private: verification
        });
      }

      verification.currentIndex = nextIndex;
      const nextQuestion = questions[verification.currentIndex];

      return JSON.stringify({
        public: {
          success: true,
          verified: false,
          status: "question",
          message: "Thank you.",
          question: safeQuestionPublic(nextQuestion)
        },
        private: verification
      });
    }

    return JSON.stringify(buildPublicError(
      "Unsupported verification action."
    ));
  } catch (error) {
    return JSON.stringify({
      public: {
        success: false,
        error: error?.message || String(error)
      }
    });
  }
}
```

## Frontend tool definition: `helsana_intents`

### Name

```text
name: helsana_intents
```

### Instructions

```text
Use this tool only after caller identity has already been successfully
verified.

This tool retrieves the verified caller's post-verification CRM context and
defines exactly which intents the bot may handle in this conversation.

Use this tool when:

* verification has succeeded,
* you need the verified customer's available service context,
* you need to know which supported intents are available for this customer,
* you are about to answer or handle a request related to
  FRANCHISE_CHANGE, ADDRESS_CHANGE, or GP_CHANGE.

Important behavior:

* Use this tool only after verification succeeds.
* Do not use it before verification.
* The only globally supported intents are:

  * FRANCHISE_CHANGE
  * ADDRESS_CHANGE
  * GP_CHANGE
* Any request outside those three intents is always unsupported.
* Even among those three, only intents marked available for this customer may
  be handled.
* Use the returned customer context to answer questions like the current
  franchise or current address.
* If an intent is globally supported but unavailable for this customer,
  explain that it is not available in the current context.
* If a request is outside the three globally supported intents, refuse
  politely.

Tool response behavior:

* Returns verified post-verification context for the caller.
* Returns the globally supported intents, the customer-available intents, and
  the customer context needed to handle them.
* This tool is the unlock point for post-verification account help.
```

### Params

```json
{
  "type": "object",
  "required": [
    "user_request"
  ],
  "properties": {
    "user_request": {
      "type": "string",
      "description": "The caller's current request or question in natural language, for example: 'I want to change my franchise, but first remind me what my current one is.'"
    }
  },
  "additionalProperties": false
}
```

### Backend handler

```javascript
async (args, sessionContext) => {
  const FASTAPI_BASE_URL = "http://127.0.0.1:8003";

  const getLookupState = () => {
    const state = sessionContext.getPrivateToolState
      ? sessionContext.getPrivateToolState("helsana_crm_lookup")
      : null;
    return state || null;
  };

  const getVerificationState = () => {
    const root = sessionContext.sessionState || {};
    return root.helsana_verification || null;
  };

  const cleanString = (value) => {
    if (value === null || value === undefined) return "";
    return String(value).trim();
  };

  const buildPublicError = (message) => ({
    public: {
      success: false,
      error: message
    }
  });

  try {
    const verificationState = getVerificationState();
    if (!verificationState || verificationState.verified !== true) {
      return JSON.stringify(buildPublicError("Caller is not verified yet."));
    }

    const userRequest = cleanString(args.user_request);
    if (!userRequest) {
      return JSON.stringify(buildPublicError("user_request is required."));
    }

    const lookupState = getLookupState();
    if (!lookupState) {
      return JSON.stringify(buildPublicError(
        "No CRM lookup state found."
      ));
    }

    const hits = Array.isArray(lookupState.hits) ? lookupState.hits : [];
    const uniqueHit = hits.length === 1 ? hits[0] : null;
    const members = uniqueHit && Array.isArray(uniqueHit.members)
      ? uniqueHit.members
      : [];
    const person = members.length === 1 ? members[0] : null;

    if (!uniqueHit || !person) {
      return JSON.stringify(buildPublicError(
        "No unique verified customer context is available."
      ));
    }

    const family_id = String(
      uniqueHit.family_id || person.FAMILY_ID || person.family_id || ""
    ).trim();

    const partnernr = String(
      person.PARTNERNR || person.partnernr || ""
    ).trim();

    if (!family_id || !partnernr) {
      return JSON.stringify(buildPublicError(
        "Verified customer identifiers are missing."
      ));
    }

    const response = await fetch(
      `${FASTAPI_BASE_URL}/llm-post-verification-context`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          family_id,
          partnernr,
          user_request: userRequest
        })
      }
    );

    const text = await response.text();

    if (!response.ok) {
      return JSON.stringify({
        public: {
          success: false,
          error: `Intent context lookup failed with status ${response.status}`,
          details: text
        }
      });
    }

    let result;
    try {
      result = JSON.parse(text);
    } catch (parseError) {
      return JSON.stringify({
        public: {
          success: false,
          error: "Intent context lookup returned invalid JSON.",
          details: text
        }
      });
    }

    return JSON.stringify({
      public: {
        success: true,
        ...(result.public || {})
      },
      private: {
        source_tool: "helsana_intents",
        family_id,
        partnernr,
        user_request: userRequest,
        ...(result.private || {})
      }
    });
  } catch (error) {
    return JSON.stringify({
      public: {
        success: false,
        error: error?.message || String(error)
      }
    });
  }
}
```

## Example lookup output

Example public and private lookup response:

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
    "verification_questions": [
      {
        "tag": "FRANCHISE",
        "field": "FRANCHISE",
        "question": "What is your selected franchise?",
        "type": "NUMBER",
        "expected_answers": [
          "1500.0"
        ]
      },
      {
        "tag": "PRODUKT",
        "field": "PRODUKT",
        "question": "Can you name at least one of your insured products?",
        "type": "TEXT",
        "expected_answers": [
          "HOSPITAL ECO",
          "SANA",
          "TOP"
        ]
      }
    ],
    "lookup_strategy": "deterministic_unique"
  }
}
```

## Notes for the VA platform

- The model should only rely on `public` tool output.
- Hidden verification answers must remain in `private` state.
- Verification should never be skipped.
- Post-verification context should only be loaded after
  `helsana_verification` returns `verified: true`.
- The FastAPI docs remain the source of truth for endpoint request and
  response shapes.
