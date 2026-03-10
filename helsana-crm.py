import json
import random
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI

app = FastAPI(title="Customer Search API with LLM Fuzzy Match")

DATA_FILE = "helsana-crm.csv"
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
    # preserve order, dedupe
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

    # common cleanup for values like "Dr. Max Muster, Zürich"
    s = re.sub(r"\s+", " ", s).strip()

    # if semicolon/comma-delimited, take first meaningful chunk
    chunks = [c.strip() for c in re.split(r"[;,|]", s) if c.strip()]
    if chunks:
        return chunks[0]

    return s


def is_benefit_plus_model(value: str) -> bool:
    s = str(value).casefold()
    return "benefit plus" in s


def should_ask_franchise(row: Dict[str, Any]) -> bool:
    # conservative default:
    # ask only when there is KVG/KVGO and franchise is non-empty/non-zero-ish
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
# Allowed verification questions
# ---------------------------------------------------------
def q_bank_exkasso_condition(row: Dict[str, Any]) -> bool:
    return contains_token(row.get("ZAHLUNGSMITTEL_EXKASSO", ""), "EZAG")


def q_bank_exkasso_extract(val: Any) -> str:
    return str(val).trim() if hasattr(str(val), "trim") else str(val).strip()


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
    """
    Evaluate conditions row-by-row on source_rows.
    A question is eligible if:
    - at least one source row satisfies the condition
    - extracted answer is non-empty
    - all distinct valid extracted answers are preserved
    """
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
    """
    Provide safe structured info the bot can use AFTER successful verification.
    This is not for pre-verification display.
    """
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


# ---------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------
@app.post("/search")
async def search_customers_fuzzy(request: Request):
    raw_body = await request.body()
    query_text = raw_body.decode("utf-8").strip()

    if not query_text:
        return {
            "outcome": "none",
            "hits": [],
            "verification_questions": [],
            "post_verification_summary": None,
        }

    system_prompt = """You are an advanced fuzzy search assistant for a customer database.
You will be provided with:
1. a JSON array of customer records
2. a raw search query from the user

The query may be:
- valid JSON
- broken JSON
- a messy string
- an ASR transcript with transcription mistakes

Your job:
- infer the likely search intent
- fuzzily identify the best matching customer records

Rules:
- Handle typos, phonetic email spellings, broken formatting, and noisy ASR.
- Return ONLY a JSON object.
- Output format MUST be exactly:
{"matched_partnernrs": ["12345", "67890"]}
- If no matches are found:
{"matched_partnernrs": []}
- No markdown, no explanation, no extra keys.
"""

    user_prompt = (
        f"Raw Search Input / ASR Transcript:\n{query_text}\n\n"
        f"Customer Records:\n{SEARCH_CORPUS_JSON}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        llm_result = json.loads(response.choices[0].message.content)
        matched_ids = llm_result.get("matched_partnernrs", [])
        matched_ids = [str(x).strip() for x in matched_ids if str(x).strip()]
    except Exception as e:
        print(f"LLM Call Failed: {e}")
        matched_ids = []

    hits = build_family_hits_from_partnernrs(matched_ids)
    outcome = determine_outcome(hits)

    verification_questions: List[Dict[str, Any]] = []
    post_verification_summary = None

    if outcome == "unique":
        unique_person = extract_unique_persons_from_hits(hits)[0]["person"]
        verification_questions = build_verification_questions_for_person(unique_person, count=2)

        # Intentionally included but meant for use only after verification succeeds.
        # If you want stricter separation, move this entirely to a dedicated endpoint.
        post_verification_summary = build_post_verification_summary(unique_person)

    return {
        "outcome": outcome,
        "hits": hits,
        "verification_questions": verification_questions,
        "post_verification_summary": post_verification_summary,
    }


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

    return {
        "family_id": family_id,
        "partnernr": partnernr,
        "verification_questions": questions,
    }


# ---------------------------------------------------------
# Post-verification data endpoint
# Call this only after verification has succeeded
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

    return {
        "family_id": family_id,
        "partnernr": partnernr,
        "post_verification_summary": build_post_verification_summary(merged_person),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
