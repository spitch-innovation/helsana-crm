import csv
import json
unique_names = set()

with open("helsana-crm.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        first = row.get("first_name", "").strip()
        last = row.get("last_name", "").strip()
        if first or last:
            unique_names.add(f"{first} {last}".strip())

print(json.dumps(list(unique_names), ensure_ascii=False))
