import json
import random
import re
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI

app = FastAPI(title="Customer Search API with Hybrid Exact + LLM Fuzzy Match")

DATA_FILE = "./data/helsana-crm.csv"
df = pd.read_csv(DATA_FILE)

# ---------------------------------------------------------
# Data prep
# ---------------------------------------------------------
first_col = df.columns[0]
if str(first_col).strip() == "" or "unnamed" in str(first_col).lower():
    df.rename(columns={first_col: "FAMILY_ID"}, inplace=True)
elif first_col != "FAMILY_ID":
    df.rename(columns={first_col: "FAMILY_ID"}, inplace=True)

df = df.fillna("")

for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

if "ZAHLUNGSMITTEL_INKASSO " in df.columns and "ZAHLUNGSMITTEL_INKASSO" not in df.columns:
    df.rename(columns={"ZAHLUNGSMITTEL_INKASSO ": "ZAHLUNGSMITTEL_INKASSO"}, inplace=True)

SEARCHABLE_COLUMNS = [
    "FAMILY_ID",
    "PARTNERNR",
    "last_name",
    "first_name",
    "TEL_PORTAL_MOBIL",
    "EMAIL_PORTAL",
    "GEB_D",
    "STRASSE",
    "HAUSNUMMER",
    "PLZ",
    "ORT",
]
SEARCHABLE_COLUMNS = [c for c in SEARCHABLE_COLUMNS if c in df.columns]

search_corpus_list = df[SEARCHABLE_COLUMNS].to_dict(orient="records")
SEARCH_CORPUS_JSON = json.dumps(search_corpus_list, ensure_ascii=False)

client = OpenAI()

BOT_HANDLED_INTENTS = [
    "FRANCHISE_CHANGE",
    "ADDRESS_CHANGE",
    "GP_CHANGE",
]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("ß", "ss")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_phone(value: Any) -> str:
    s = str(value or "").strip()
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    return digits


def normalize_email(value: Any) -> str:
    s = str(value or "").strip().lower()
    replacements = [
        (r"\s+at\s+", "@"),
        (r"\s+ät\s+", "@"),
        (r"\s+arroba\s+", "@"),
        (r"\s+dot\s+", "."),
        (r"\s+punkt\s+", "."),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)
    s = s.replace(" ", "")
    return s


def normalize_date(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return s


def parse_search_payload(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {"query_text": raw_text}


def contains_token(value: str, token: str) -> bool:
    parts = [p.strip().upper() for p in str(value).split(",") if p.strip()]
    return token.strip().upper() in parts


def has_kvg(value: str) -> bool:
    return contains_token(value, "KVGO") or contains_token(value, "KVG")


def has_vvg(value: str) -> bool:
    return contains_token(value, "VVG")


def count_products(value: str) -> int:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return len(parts)


def format_product_list(value: str) -> List[str]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out = []
    seen = set()
    for p in parts:
        key = p.casefold()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def extract_doctor_name(value: str) -> str:
    s = str(value).strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    chunks = [c.strip() for c in re.split(r"[;,|]", s) if c.strip()]
    if chunks:
        return chunks[0]
    return s


def is_benefit_plus_model(value: str) -> bool:
    s = str(value).casefold()
    return "benefit plus" in s


def should_ask_franchise(row: Dict[str, Any]) -> bool:
    gesetz = str(row.get("GESETZ", ""))
    franchise = str(row.get("FRANCHISE", "")).strip()
    if not has_kvg(gesetz):
        return False
    if not franchise:
        return False
    return True


def is_empty_extracted(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return len([v for v in value if str(v).strip()]) == 0
    s = str(value).strip()
    return s == ""


def distinct_non_empty_values(rows: List[Dict[str, Any]], key: str) -> List[str]:
    values: Set[str] = set()
    for row in rows:
        v = normalize_value(row.get(key, ""))
        if v:
            values.add(v)
    return sorted(values)


def merge_person_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    merged: Dict[str, Any] = {}
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    for key in all_keys:
        values = distinct_non_empty_values(rows, key)
        if len(values) == 0:
            merged[key] = ""
        elif len(values) == 1:
            merged[key] = values[0]
        else:
            merged[key] = values[0]
            merged[f"{key}__all_values"] = values

    merged["source_row_count"] = len(rows)
    merged["source_rows"] = rows
    return merged


def group_rows_by_person(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_partnernr: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        partnernr = normalize_value(row.get("PARTNERNR", ""))
        if not partnernr:
            partnernr = f"_missing_{len(by_partnernr)}"
        by_partnernr.setdefault(partnernr, []).append(row)

    persons = []
    for _, person_rows in by_partnernr.items():
        persons.append(merge_person_rows(person_rows))

    return persons


def build_family_hits_from_partnernrs(matched_partnernrs: List[str]) -> List[Dict[str, Any]]:
    if not matched_partnernrs:
        return []

    matched_df = df[df["PARTNERNR"].isin(matched_partnernrs)].copy()
    if matched_df.empty:
        return []

    hits = []
    for family_id, family_group in matched_df.groupby("FAMILY_ID", dropna=False):
        family_rows = family_group.to_dict(orient="records")
        merged_members = group_rows_by_person(family_rows)
        hits.append({
            "family_id": family_id,
            "person_count": len(merged_members),
            "members": merged_members,
        })

    return hits


def extract_unique_persons_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    persons = []
    seen = set()

    for hit in hits:
        family_id = hit["family_id"]
        for member in hit["members"]:
            partnernr = normalize_value(member.get("PARTNERNR", ""))
            key = (family_id, partnernr)
            if key not in seen:
                seen.add(key)
                persons.append({
                    "family_id": family_id,
                    "partnernr": partnernr,
                    "person": member,
                })

    return persons


def determine_outcome(hits: List[Dict[str, Any]]) -> str:
    unique_persons = extract_unique_persons_from_hits(hits)
    if len(unique_persons) == 0:
        return "none"
    if len(unique_persons) == 1:
        return "unique"
    return "multi"


# ---------------------------------------------------------
# Safe shaping helpers
# ---------------------------------------------------------
def mask_phone(value: str) -> str:
    digits = normalize_phone(value)
    if len(digits) < 4:
        return ""
    return f"***{digits[-4:]}"


def mask_email(value: str) -> str:
    email = normalize_email(value)
    if not email or "@" not in email:
        return ""
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    return f"{local[:1]}***@{domain}"


def safe_member_public(member: Dict[str, Any]) -> Dict[str, Any]:
    first_name = normalize_value(member.get("first_name", ""))
    last_name = normalize_value(member.get("last_name", ""))
    display_name = " ".join([x for x in [first_name, last_name] if x]).strip()

    return {
        "family_id": normalize_value(member.get("FAMILY_ID", "")) or normalize_value(member.get("family_id", "")),
        "partnernr": normalize_value(member.get("PARTNERNR", "")),
        "display_name": display_name,
    }


def safe_hit_public(hit: Dict[str, Any]) -> Dict[str, Any]:
    members = hit.get("members", [])
    return {
        "family_id": normalize_value(hit.get("family_id", "")),
        "person_count": hit.get("person_count", len(members)),
        "members": [safe_member_public(m) for m in members],
    }

def build_public_search_response(outcome, hits, lookup_strategy, verification_questions=None):
    return {
        "outcome": outcome,
        "lookup_strategy": lookup_strategy,
        "hits": [safe_hit_public(hit) for hit in hits],
        "verification_required": outcome == "unique",
        "verification_questions": verification_questions or [],
    }

def build_split_response(public: Dict[str, Any], private: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "public": public,
        "private": private,
    }


# ---------------------------------------------------------
# Deterministic search
# ---------------------------------------------------------
def row_matches_customer_id(row: Dict[str, Any], customer_id: str) -> bool:
    cid = normalize_value(customer_id)
    if not cid:
        return False
    return (
        normalize_value(row.get("FAMILY_ID", "")) == cid
        or normalize_value(row.get("PARTNERNR", "")) == cid
    )


def row_matches_first_name(row: Dict[str, Any], first_name: str) -> bool:
    return normalize_text(row.get("first_name", "")) == normalize_text(first_name)


def row_matches_last_name(row: Dict[str, Any], last_name: str) -> bool:
    return normalize_text(row.get("last_name", "")) == normalize_text(last_name)


def row_matches_phone(row: Dict[str, Any], phone: str) -> bool:
    q = normalize_phone(phone)
    if not q:
        return False
    return normalize_phone(row.get("TEL_PORTAL_MOBIL", "")) == q


def row_matches_email(row: Dict[str, Any], email: str) -> bool:
    q = normalize_email(email)
    if not q:
        return False
    return normalize_email(row.get("EMAIL_PORTAL", "")) == q


def row_matches_birthday(row: Dict[str, Any], birthday: str) -> bool:
    q = normalize_date(birthday)
    if not q:
        return False
    return normalize_date(row.get("GEB_D", "")) == q


def row_matches_address(row: Dict[str, Any], address: Dict[str, Any]) -> bool:
    if not address:
        return False

    checks = []
    if address.get("strasse"):
        checks.append(normalize_text(row.get("STRASSE", "")) == normalize_text(address.get("strasse", "")))
    if address.get("hausnummer"):
        checks.append(normalize_text(row.get("HAUSNUMMER", "")) == normalize_text(address.get("hausnummer", "")))
    if address.get("plz"):
        checks.append(normalize_text(row.get("PLZ", "")) == normalize_text(address.get("plz", "")))

    return len(checks) > 0 and all(checks)


def deterministic_match_partnernrs(payload: Dict[str, Any]) -> List[str]:
    customer_id = normalize_value(payload.get("customer_id", ""))
    first_name = normalize_value(payload.get("first_name", ""))
    last_name = normalize_value(payload.get("last_name", ""))
    phone = normalize_value(payload.get("tel_portal_mobil", ""))
    email = normalize_value(payload.get("email_portal", ""))
    birthday = normalize_value(payload.get("birthday", ""))
    address = payload.get("address") if isinstance(payload.get("address"), dict) else {}

    have_any_structured = any([
        customer_id,
        first_name,
        last_name,
        phone,
        email,
        birthday,
        bool(address),
    ])
    if not have_any_structured:
        return []

    matched_partnernrs: Set[str] = set()

    for row in df.to_dict(orient="records"):
        if customer_id and not row_matches_customer_id(row, customer_id):
            continue
        if first_name and not row_matches_first_name(row, first_name):
            continue
        if last_name and not row_matches_last_name(row, last_name):
            continue
        if phone and not row_matches_phone(row, phone):
            continue
        if email and not row_matches_email(row, email):
            continue
        if birthday and not row_matches_birthday(row, birthday):
            continue
        if address and not row_matches_address(row, address):
            continue

        partnernr = normalize_value(row.get("PARTNERNR", ""))
        if partnernr:
            matched_partnernrs.add(partnernr)

    return sorted(matched_partnernrs)


def deterministic_lookup(payload: Dict[str, Any]) -> Tuple[List[str], str]:
    matched = deterministic_match_partnernrs(payload)
    hits = build_family_hits_from_partnernrs(matched)
    outcome = determine_outcome(hits)

    if outcome == "unique":
        return matched, "deterministic_unique"
    if outcome == "multi":
        return matched, "deterministic_non_unique"
    return matched, "deterministic_none"


# ---------------------------------------------------------
# LLM fallback search
# ---------------------------------------------------------
def llm_fuzzy_lookup(raw_query_text: str) -> List[str]:
    system_prompt = """You are an advanced fuzzy search assistant for a customer database.
You will be provided with:
1. a JSON array of customer records
2. a raw search query from the user

The query may be valid JSON, broken JSON, a messy string, or an ASR transcript.

Your job:
- infer the likely search intent
- fuzzily identify the best matching customer records
- strictly refuse if there are NO reasonable matches, don't return 'possible'

CRITICAL MATCHING RULES:
- Perform robust FUZZY MATCHING for all fields.
- EMAIL MATCHING IS CRITICAL: The user input might spell out punctuation phonetically.
- Phone numbers in the records might have '.0' at the end or spacing differences; ignore these and match on the core digits.
- Return ONLY a JSON object.
- Output format MUST be exactly:
{"matched_partnernrs": ["12345", "67890"]}
- If no matches are found:
{"matched_partnernrs": []}
- No markdown, no explanation, no extra keys.
"""

    user_prompt = (
        f"Raw Search Input / ASR Transcript:\n{raw_query_text}\n\n"
        f"Customer Records:\n{SEARCH_CORPUS_JSON}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        llm_result = json.loads(response.choices[0].message.content)
        matched_ids = llm_result.get("matched_partnernrs", [])
        return [str(x).strip() for x in matched_ids if str(x).strip()]
    except Exception as e:
        print(f"LLM Call Failed: {e}")
        return []


# ---------------------------------------------------------
# Allowed verification questions
# ---------------------------------------------------------
def q_bank_exkasso_condition(row: Dict[str, Any]) -> bool:
    return contains_token(row.get("ZAHLUNGSMITTEL_EXKASSO", ""), "EZAG")


def q_bank_exkasso_extract(val: Any) -> str:
    return str(val).strip()


def q_hausarzt_condition(row: Dict[str, Any]) -> bool:
    return is_benefit_plus_model(row.get("GRUNDVERSICHERUNG_MODELL", ""))


def q_hausarzt_extract(val: Any) -> str:
    return extract_doctor_name(val)


def q_franchise_condition(row: Dict[str, Any]) -> bool:
    return should_ask_franchise(row)


def q_franchise_extract(val: Any) -> str:
    return str(val).strip()


def q_zahlungsmittel_inkasso_condition(row: Dict[str, Any]) -> bool:
    return True


def q_zahlungsmittel_inkasso_extract(val: Any) -> str:
    return str(val).strip()


def q_grundversicherung_modell_condition(row: Dict[str, Any]) -> bool:
    return has_kvg(row.get("GESETZ", ""))


def q_grundversicherung_modell_extract(val: Any) -> str:
    return str(val).strip()


def q_produkt_condition(row: Dict[str, Any]) -> bool:
    return count_products(row.get("PRODUKT", "")) > 1


def q_produkt_extract(val: Any) -> List[str]:
    return format_product_list(val)


def q_gesamtpraemie_condition(row: Dict[str, Any]) -> bool:
    gesetz = row.get("GESETZ", "")
    return has_kvg(gesetz) and has_vvg(gesetz)


def q_gesamtpraemie_extract(val: Any) -> str:
    return str(val).strip()


ALL_QUESTIONS = [
    {
        "col": "BANK_EXKASSO",
        "tag": "BANK_EXKASSO",
        "question": "Which bank do we use for payouts?",
        "type": "TEXT",
        "condition": q_bank_exkasso_condition,
        "extract": q_bank_exkasso_extract,
    },
    {
        "col": "HAUSARZT",
        "tag": "HAUSARZT",
        "question": "Please state the name of your general practitioner.",
        "type": "TEXT",
        "condition": q_hausarzt_condition,
        "extract": q_hausarzt_extract,
    },
    {
        "col": "FRANCHISE",
        "tag": "FRANCHISE",
        "question": "What is your selected franchise?",
        "type": "NUMBER",
        "condition": q_franchise_condition,
        "extract": q_franchise_extract,
    },
    {
        "col": "ZAHLUNGSMITTEL_INKASSO",
        "tag": "ZAHLUNGSMITTEL_INKASSO",
        "question": "Which payment method do you use to pay for your invoice?",
        "type": "TEXT",
        "condition": q_zahlungsmittel_inkasso_condition,
        "extract": q_zahlungsmittel_inkasso_extract,
    },
    {
        "col": "GRUNDVERSICHERUNG_MODELL",
        "tag": "GRUNDVERSICHERUNG_MODELL",
        "question": "Which basic insurance model do you have with us?",
        "type": "TEXT",
        "condition": q_grundversicherung_modell_condition,
        "extract": q_grundversicherung_modell_extract,
    },
    {
        "col": "PRODUKT",
        "tag": "PRODUKT",
        "question": "Can you name at least one of your insured products?",
        "type": "TEXT",
        "condition": q_produkt_condition,
        "extract": q_produkt_extract,
    },
    {
        "col": "GESAMTPRÄMIE",
        "tag": "GESAMTPRÄMIE",
        "question": "What is your total premium?",
        "type": "NUMBER",
        "condition": q_gesamtpraemie_condition,
        "extract": q_gesamtpraemie_extract,
    },
]


def build_verification_questions_for_person(person: Dict[str, Any], count: int = 2) -> List[Dict[str, Any]]:
    source_rows = person.get("source_rows", [])
    candidates = []

    for q in ALL_QUESTIONS:
        valid_answers = set()

        for row in source_rows:
            try:
                if not q["condition"](row):
                    continue

                raw_val = row.get(q["col"], "")
                extracted = q["extract"](raw_val)

                if is_empty_extracted(extracted):
                    continue

                if isinstance(extracted, list):
                    for item in extracted:
                        item_s = str(item).strip()
                        if item_s:
                            valid_answers.add(item_s)
                else:
                    ans = str(extracted).strip()
                    if ans:
                        valid_answers.add(ans)
            except Exception:
                continue

        if valid_answers:
            candidates.append({
                "tag": q["tag"],
                "field": q["col"],
                "question": q["question"],
                "type": q["type"],
                "expected_answers": sorted(valid_answers),
            })

    random.shuffle(candidates)
    return candidates[:count]


def build_post_verification_summary(person: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "bot_handled_intents": BOT_HANDLED_INTENTS,
        "intent_context": {},
    }

    gp_name = extract_doctor_name(person.get("HAUSARZT", ""))
    franchise = str(person.get("FRANCHISE", "")).strip()
    address = {
        "strasse": person.get("STRASSE", ""),
        "hausnummer": person.get("HAUSNUMMER", ""),
        "plz": person.get("PLZ", ""),
        "ort": person.get("ORT", ""),
        "land": person.get("LAND", ""),
    }

    summary["intent_context"]["FRANCHISE_CHANGE"] = {
        "current_franchise": franchise,
        "available": bool(franchise),
    }

    summary["intent_context"]["ADDRESS_CHANGE"] = {
        "current_address": address,
        "available": any(str(v).strip() for v in address.values()),
    }

    summary["intent_context"]["GP_CHANGE"] = {
        "current_general_practitioner": gp_name,
        "basic_insurance_model": person.get("GRUNDVERSICHERUNG_MODELL", ""),
        "available": bool(gp_name) or is_benefit_plus_model(person.get("GRUNDVERSICHERUNG_MODELL", "")),
    }

    return summary


def strip_expected_answers_for_public(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for q in questions:
        out.append({
            "tag": q.get("tag"),
            "field": q.get("field"),
            "question": q.get("question"),
            "type": q.get("type"),
        })
    return out


# ---------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------
@app.post("/search")
async def search_customers_fuzzy(request: Request):
    raw_body = await request.body()
    raw_text = raw_body.decode("utf-8").strip()

    if not raw_text:
        return build_split_response(
            public={
                "outcome": "none",
                "hits": [],
                "lookup_strategy": "empty_input",
                "verification_required": False,
            },
            private={
                "hits": [],
                "verification_questions": [],
                "post_verification_summary": None,
                "lookup_strategy": "empty_input",
            },
        )

    payload = parse_search_payload(raw_text)

    matched_ids, strategy = deterministic_lookup(payload)
    hits = build_family_hits_from_partnernrs(matched_ids)
    outcome = determine_outcome(hits)

    if outcome != "unique":
        llm_matched_ids = llm_fuzzy_lookup(raw_text)
        hits = build_family_hits_from_partnernrs(llm_matched_ids)
        outcome = determine_outcome(hits)
        strategy = "llm_fallback"

    verification_questions: List[Dict[str, Any]] = []
    post_verification_summary = None

    if outcome == "unique":
        unique_person = extract_unique_persons_from_hits(hits)[0]["person"]
        verification_questions = build_verification_questions_for_person(unique_person, count=2)
        post_verification_summary = build_post_verification_summary(unique_person)

    public_questions = strip_expected_answers_for_public(verification_questions)

    return build_split_response(
        public=build_public_search_response(outcome, hits, strategy, public_questions),
        private={
            "hits": hits,
            "verification_questions": verification_questions,
            "post_verification_summary": post_verification_summary,
            "lookup_strategy": strategy,
        },
    )


# ---------------------------------------------------------
# Explicit verification question endpoint
# ---------------------------------------------------------
@app.post("/verification-questions")
async def get_verification_questions(payload: Dict[str, Any]):
    family_id = normalize_value(payload.get("family_id", ""))
    partnernr = normalize_value(payload.get("partnernr", ""))
    count = payload.get("count", 2)

    try:
        count = int(count)
    except Exception:
        count = 2

    if not family_id or not partnernr:
        raise HTTPException(status_code=400, detail="family_id and partnernr are required")

    matched_df = df[
        (df["FAMILY_ID"] == family_id) &
        (df["PARTNERNR"] == partnernr)
    ].copy()

    if matched_df.empty:
        raise HTTPException(status_code=404, detail="person not found")

    merged_person = merge_person_rows(matched_df.to_dict(orient="records"))
    questions = build_verification_questions_for_person(merged_person, count=count)

    return build_split_response(
        public={
            "family_id": family_id,
            "partnernr": partnernr,
            "verification_questions": strip_expected_answers_for_public(questions),
        },
        private={
            "family_id": family_id,
            "partnernr": partnernr,
            "verification_questions": questions,
        },
    )

# ---------------------------------------------------------
# LLM-ready post-verification context endpoint
# ---------------------------------------------------------
@app.post("/llm-post-verification-context")
async def get_llm_post_verification_context(payload: Dict[str, Any]):
    family_id = normalize_value(payload.get("family_id", ""))
    partnernr = normalize_value(payload.get("partnernr", ""))

    if not family_id or not partnernr:
        raise HTTPException(status_code=400, detail="family_id and partnernr are required")

    matched_df = df[
        (df["FAMILY_ID"] == family_id) &
        (df["PARTNERNR"] == partnernr)
    ].copy()

    if matched_df.empty:
        raise HTTPException(status_code=404, detail="person not found")

    merged_person = merge_person_rows(matched_df.to_dict(orient="records"))
    summary = build_post_verification_summary(merged_person)

    bot_handled_intents = summary.get("bot_handled_intents", []) or []
    intent_context = summary.get("intent_context", {}) or {}

    llm_instruction = (
        "The caller has already been successfully verified. "
        "You may now use the verified customer context below.\n\n"

        "You are Sana, a Helsana support agent for verified existing customers. "
        "Always speak English. "
        "Respond concisely, clearly, and naturally.\n\n"

        "GLOBAL SUPPORTED INTENTS:\n"
        "The only intents this assistant can ever handle are:\n"
        "- FRANCHISE_CHANGE\n"
        "- ADDRESS_CHANGE\n"
        "- GP_CHANGE\n\n"

        "Any request outside these three intents is always unsupported and must be refused politely.\n\n"

        "CUSTOMER-SPECIFIC AVAILABILITY:\n"
        "Even among those three supported intents, not every intent is available for every customer.\n"
        "For this verified caller, check customer_context[intent].available before helping.\n"
        "An intent is actionable only if:\n"
        "1. it is one of the three globally supported intents above, and\n"
        "2. customer_context[intent].available is true.\n\n"

        "If an intent is one of the three supported intents but customer_context[intent].available is false, "
        "you must not perform that intent. "
        "Instead, explain briefly that this service is not available in the caller's current context.\n\n"

        "Use customer_context as verified account context. "
        "You may refer directly to this information when answering the caller.\n\n"

        "Behavior rules:\n"
        "- Only use information present in customer_context.\n"
        "- Do not invent missing facts, policies, prices, eligibility rules, or unsupported actions.\n"
        "- If a field is empty, say so plainly.\n"
        "- If the caller asks for current account information related to one of the three supported intents, answer directly from customer_context.\n"
        "- After answering a context question, continue helping only within intents that are both globally supported and available for this customer.\n\n"

        "Intent-specific guidance:\n"
        "- FRANCHISE_CHANGE: you may discuss the current franchise if present.\n"
        "- ADDRESS_CHANGE: you may discuss the current saved address if present.\n"
        "- GP_CHANGE: you may discuss the current general practitioner and insurance model context if present.\n\n"

        "Decision policy:\n"
        "- If the request matches one of the three globally supported intents and it is available for this customer, help with it.\n"
        "- If the request matches one of the three globally supported intents but it is unavailable for this customer, explain that it is not available in the current context.\n"
        "- If the request is outside the three globally supported intents, refuse politely.\n"
        "- Stay strictly within the three globally supported intents and the provided customer_context."
    )

    return {
        "public": {
            "verified": True,
            "bot_handled_intents": bot_handled_intents,
            "intent_context": intent_context,
            "llm_context": {
                "allowed_global_intents": [
                    "FRANCHISE_CHANGE",
                    "ADDRESS_CHANGE",
                    "GP_CHANGE",
                ],
                "customer_available_intents": [
                    intent_name
                    for intent_name, context in intent_context.items()
                    if isinstance(context, dict) and context.get("available") is True
                ],
                "customer_context": intent_context,
                "instruction": llm_instruction,
            },
        },
        "private": {
            "family_id": family_id,
            "partnernr": partnernr,
            "post_verification_summary": summary,
        },
    }
    
# ---------------------------------------------------------
# Post-verification data endpoint
# ---------------------------------------------------------
@app.post("/post-verification-summary")
async def get_post_verification_summary(payload: Dict[str, Any]):
    family_id = normalize_value(payload.get("family_id", ""))
    partnernr = normalize_value(payload.get("partnernr", ""))

    if not family_id or not partnernr:
        raise HTTPException(status_code=400, detail="family_id and partnernr are required")

    matched_df = df[
        (df["FAMILY_ID"] == family_id) &
        (df["PARTNERNR"] == partnernr)
    ].copy()

    if matched_df.empty:
        raise HTTPException(status_code=404, detail="person not found")

    merged_person = merge_person_rows(matched_df.to_dict(orient="records"))
    summary = build_post_verification_summary(merged_person)

    return build_split_response(
        public={
            "family_id": family_id,
            "partnernr": partnernr,
            "post_verification_summary": summary,
        },
        private={
            "family_id": family_id,
            "partnernr": partnernr,
            "post_verification_summary": summary,
        },
    )

# ---------------------------------------------------------
# Verification answer check (guarded fuzzy match)
# ---------------------------------------------------------
def normalize_verification_number(value: Any) -> str:
    return str(value or "").replace(",", ".").replace(" ", "").strip()


def normalize_verification_text(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("ß", "ss")
    s = re.sub(r"\s+", " ", s)
    return s


def deterministic_verify_answer(question_type: str, proposed_answer: str, expected_answers: List[str]) -> bool:
    qtype = str(question_type or "TEXT").upper()
    if qtype == "NUMBER":
        proposed = normalize_verification_number(proposed_answer)
        return any(proposed == normalize_verification_number(x) for x in expected_answers if str(x).strip())

    proposed = normalize_verification_text(proposed_answer)
    for expected in expected_answers:
        exp = normalize_verification_text(expected)
        if not exp:
            continue
        if proposed == exp:
            return True
        if proposed in exp or exp in proposed:
            return True

    return False


def llm_verify_answer(question: str, question_type: str, proposed_answer: str, expected_answers: List[str]) -> Dict[str, Any]:
    """
    Very guarded fuzzy verifier:
    - robust to ASR / pronunciation / minor spelling variation
    - not meant to be permissive to semantically different answers
    """
    system_prompt = """You are a strict verification matcher for insurance call-center identity verification.

Your task is to compare:
1. a verification question
2. a caller's proposed answer (possibly ASR-noisy)
3. a small list of allowed expected answers

You must decide whether the proposed answer should count as a valid match.

Rules:
- Be robust to ASR mistakes, minor spelling errors, accents, punctuation differences, singular/plural variation, and small phonetic confusions.
- Accept obvious ASR confusions like:
  - "echo" vs "eco"
  - "prevea unfall" vs "prevea unfall"
  - mild tokenization issues
- For NUMBER questions:
  - only accept if the numeric value is effectively the same
  - do not accept nearby numbers
- For TEXT questions:
  - accept only if the proposed answer is clearly referring to one of the expected answers
  - do not accept semantically different products, banks, doctors, insurance models, or payment methods
  - do not be generous across different entities
- If uncertain, return false.
- Never invent new expected answers.
- Return ONLY JSON in exactly this shape:
{"matched": true, "matched_expected_answer": "..."}

or

{"matched": false, "matched_expected_answer": null}
"""

    user_prompt = json.dumps(
        {
            "question": question,
            "question_type": question_type,
            "proposed_answer": proposed_answer,
            "expected_answers": expected_answers,
        },
        ensure_ascii=False,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        return {
            "matched": bool(parsed.get("matched", False)),
            "matched_expected_answer": parsed.get("matched_expected_answer"),
        }
    except Exception as e:
        print(f"LLM verification call failed: {e}")
        return {
            "matched": False,
            "matched_expected_answer": None,
        }


@app.post("/verify-answer")
async def verify_answer(payload: Dict[str, Any]):
    question = str(payload.get("question", "")).strip()
    question_type = str(payload.get("type", "TEXT")).strip().upper()
    proposed_answer = str(payload.get("proposed_answer", "")).strip()
    expected_answers = payload.get("expected_answers", [])

    if not proposed_answer:
        raise HTTPException(status_code=400, detail="proposed_answer is required")

    if not isinstance(expected_answers, list) or not expected_answers:
        raise HTTPException(status_code=400, detail="expected_answers must be a non-empty list")

    expected_answers = [str(x).strip() for x in expected_answers if str(x).strip()]
    if not expected_answers:
        raise HTTPException(status_code=400, detail="expected_answers must contain at least one non-empty value")

    # 1) deterministic fast path
    deterministic_match = deterministic_verify_answer(
        question_type=question_type,
        proposed_answer=proposed_answer,
        expected_answers=expected_answers,
    )
    if deterministic_match:
        return {
            "matched": True,
            "method": "deterministic",
            "matched_expected_answer": next(
                (
                    x for x in expected_answers
                    if (
                        normalize_verification_number(x) == normalize_verification_number(proposed_answer)
                        if question_type == "NUMBER"
                        else (
                            normalize_verification_text(x) == normalize_verification_text(proposed_answer)
                            or normalize_verification_text(x) in normalize_verification_text(proposed_answer)
                            or normalize_verification_text(proposed_answer) in normalize_verification_text(x)
                        )
                    )
                ),
                None
            ),
        }

    # 2) guarded LLM fallback
    llm_result = llm_verify_answer(
        question=question,
        question_type=question_type,
        proposed_answer=proposed_answer,
        expected_answers=expected_answers,
    )

    return {
        "matched": bool(llm_result.get("matched", False)),
        "method": "llm_fallback",
        "matched_expected_answer": llm_result.get("matched_expected_answer"),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)