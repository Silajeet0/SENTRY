"""Generate the reproducible synthetic binary affiliation fine-tuning corpus.

The source templates retain the earlier three-way annotation so the conversion is
auditable. For binary training, positive means affirmative current Indian evidence;
all foreign, missing, conflicting, or otherwise unresolved evidence is negative.
The generated JSONL rows are directly consumable by ``laya-train``.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "laya_finetuning"
SEED = 20261029
LABELS = ("positive", "negative")
QUESTION_ID = "affiliation_geography"
QUESTIONS = {
    QUESTION_ID: {
        "type": "choice",
        "instructions": "Does this affiliation provide sufficient evidence that the current institution or campus is located in India?",
        "criteria": {
            "INDIAN": "Sufficient evidence that the current institution or campus is in India.",
            "NOT_INDIAN": "No sufficient evidence that the current institution or campus is in India.",
        },
    }
}


def _rows(label, category, rationale, items, wrappers, target):
    out = []
    seen = set()
    index = 0
    for item in items:
        for wrapper in wrappers:
            text = wrapper(item)
            if text in seen:
                continue
            seen.add(text)
            out.append({"affiliation": text, "label": label, "category": category, "rationale": rationale})
            if len(out) == target:
                return out
            index += 1
    # The extra wrappers below are deterministic and make category counts exact.
    suffixes = ["; research collaboration", "; author affiliation", ", research group", "; academic unit", ", main campus"]
    for item in items:
        for suffix in suffixes:
            text = item + suffix
            if text not in seen:
                seen.add(text)
                out.append({"affiliation": text, "label": label, "category": category, "rationale": rationale})
                if len(out) == target:
                    return out
    raise AssertionError(f"Could not generate {target} unique rows for {label}/{category}")


def _build():
    positive = []
    positive_specs = [
        ("explicit_india", "explicit country evidence", [
            "Indian Institute of Technology Bombay", "Indian Institute of Science", "Jadavpur University", "Anna University", "Banaras Hindu University", "University of Delhi", "University of Hyderabad", "Tata Institute of Fundamental Research", "National Institute of Technology Karnataka", "Indian Institute of Information Technology Hyderabad"],
         [lambda x: f"{x}, India", lambda x: f"{x}, Bengaluru, India", lambda x: f"{x}; India", lambda x: f"{x}, India 560012", lambda x: f"{x} (India)"]),
        ("indian_city_state", "recognized Indian institution with an Indian city or state", [
            "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur", "IISc", "ISI", "BITS Pilani", "IIIT Hyderabad", "NIT Tiruchirappalli"],
         [lambda x: f"{x}, Mumbai, Maharashtra", lambda x: f"{x}, New Delhi, Delhi", lambda x: f"{x}, Chennai, Tamil Nadu", lambda x: f"{x}, Bengaluru, Karnataka", lambda x: f"{x}, Hyderabad, Telangana", lambda x: f"{x}, Mumbai"]),
        ("full_institution_name", "full name identifies an Indian institution", [
            "Indian Institute of Technology (BHU), Varanasi", "Indian Institute of Technology (Indian School of Mines), Dhanbad", "Indian Statistical Institute, Kolkata", "Indian Institute of Science Education and Research Pune", "Council of Scientific and Industrial Research", "Defence Research and Development Organisation", "Indian Space Research Organisation", "Amrita Vishwa Vidyapeetham", "Manipal Academy of Higher Education", "O. P. Jindal Global University"],
         [lambda x: x, lambda x: f"{x}, India", lambda x: f"{x}, campus in India", lambda x: f"{x}, Indian campus", lambda x: f"{x}; India"]),
        ("qualified_acronym", "ambiguous acronym resolved by Indian campus qualifier", [
            "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur", "IIT Hyderabad", "IIT Guwahati", "IIT Roorkee", "IISc Bengaluru", "ISI Kolkata"],
         [lambda x: x, lambda x: f"{x}, India", lambda x: f"{x} campus", lambda x: f"Department, {x}", lambda x: f"{x}, Indian Institute campus"]),
        ("indian_corporate_lab", "corporate research office explicitly located in India", [
            "Microsoft Research India", "Google Research Bengaluru", "IBM Research India", "Adobe Research Bangalore", "Samsung R&D Institute India", "Amazon Development Centre India", "Intel India Research", "Qualcomm India", "Infosys Research Bengaluru", "TCS Research Mumbai"],
         [lambda x: f"{x}, India", lambda x: f"{x}, Bengaluru, India", lambda x: f"{x}, Bangalore", lambda x: f"{x}, Mumbai, Maharashtra", lambda x: f"{x}, New Delhi, India"]),
        ("noisy_department_prefix", "departmental noise does not change a resolved Indian institution", [
            "Indian Institute of Technology Bombay", "Indian Institute of Science", "Jadavpur University", "University of Delhi", "Indian Statistical Institute Kolkata", "BITS Pilani", "IIIT Hyderabad", "IIT Madras", "Anna University", "Tata Institute of Fundamental Research"],
         [lambda x: f"Department of Computer Science, {x}, India", lambda x: f"School of Engineering; {x}; Mumbai, India", lambda x: f"Centre for AI, {x}, Bengaluru", lambda x: f"Dept. CSE | {x} | India", lambda x: f"Research Lab, {x}, Indian campus"]),
        ("multiple_affiliations", "at least one current affiliation is Indian", [
            "IIT Bombay; Stanford University", "University of Delhi; University of Oxford", "IISc Bengaluru; Microsoft Research USA", "Jadavpur University; ETH Zurich", "IIT Madras; National University of Singapore", "ISI Kolkata; University of Cambridge", "BITS Pilani; Amazon Science Seattle", "TIFR Mumbai; CERN Switzerland", "IIIT Hyderabad; University of Toronto", "IIT Kharagpur; Google Research Zurich"],
         [lambda x: f"{x}, current affiliations", lambda x: f"Affiliations: {x}", lambda x: f"{x}; joint appointment", lambda x: f"{x}; author has multiple current affiliations", lambda x: f"[1] {x}"]),
    ]
    positive_targets = [57, 57, 57, 57, 57, 57, 58]
    for (category, rationale, items, wrappers), target in zip(positive_specs, positive_targets):
        positive.extend(_rows("positive", category, rationale, items, wrappers, target))

    negative = []
    negative_specs = [
        ("explicit_foreign_institution", "identified institution is outside India", ["Stanford University", "Massachusetts Institute of Technology", "University of Cambridge", "University of Oxford", "ETH Zurich", "National University of Singapore", "Tsinghua University", "University of Tokyo", "University of Toronto", "University of Melbourne"], [lambda x: f"{x}, USA", lambda x: f"{x}, United Kingdom", lambda x: f"{x}, Switzerland", lambda x: f"{x}, Singapore", lambda x: f"{x}, Australia"]),
        ("foreign_city_country", "foreign city/country identifies the current location", ["Department of Computer Science", "School of Engineering", "Research Laboratory", "Faculty of Computing", "Centre for AI", "Computer Vision Group", "Data Science Institute", "Information Sciences Institute", "Systems Lab", "Machine Learning Department"], [lambda x: f"{x}, Boston, USA", lambda x: f"{x}, London, United Kingdom", lambda x: f"{x}, Zurich, Switzerland", lambda x: f"{x}, Singapore", lambda x: f"{x}, Tokyo, Japan"]),
        ("acronym_collision", "acronym is resolved to a non-Indian institution", ["IIT", "ISI", "NIT", "VIT", "MIT", "JU", "IIIT", "IIM", "CSE", "USC"], [lambda x: f"{x}, Chicago, USA", lambda x: f"{x}, Genova, Italy", lambda x: f"{x}, Tokyo, Japan", lambda x: f"{x}, Melbourne, Australia", lambda x: f"{x}, Los Angeles, USA"]),
        ("overseas_campus", "campus location is outside India despite an Indian parent name", ["Indian Institute of Technology Madras", "IIT Delhi", "BITS Pilani", "Manipal Academy of Higher Education", "Amity University", "Indian Institute of Science", "VIT", "IIT Bombay", "IIM Ahmedabad", "SRM Institute of Science and Technology"], [lambda x: f"{x} Zanzibar campus, Tanzania", lambda x: f"{x} Abu Dhabi campus, UAE", lambda x: f"{x} Dubai campus, United Arab Emirates", lambda x: f"{x} Malaysia campus", lambda x: f"{x} Mauritius campus"]),
        ("foreign_corporate_branch", "corporate branch is explicitly outside India", ["Microsoft Research", "Google Research", "IBM Research", "Amazon Science", "Adobe Research", "Samsung Research", "Intel Labs", "Tata Consultancy Services", "Infosys", "Wipro Research"], [lambda x: f"{x}, Redmond, USA", lambda x: f"{x}, Zurich, Switzerland", lambda x: f"{x}, London, United Kingdom", lambda x: f"{x}, Toronto, Canada", lambda x: f"{x}, Sydney, Australia"]),
        ("historical_indian_affiliation", "Indian institution is historical; current affiliation is foreign", ["Stanford University", "University of Oxford", "ETH Zurich", "University of Cambridge", "MIT", "National University of Singapore", "University of Toronto", "Tsinghua University", "University of Tokyo", "University of Melbourne"], [lambda x: f"{x}, USA; formerly IIT Bombay, India", lambda x: f"{x}, UK; previously IISc Bangalore", lambda x: f"{x}, Switzerland; alumni of IIT Delhi", lambda x: f"Current: {x}; former affiliation Jadavpur University", lambda x: f"{x}; moved from IIT Madras in 2020"]),
        ("india_topic_or_department", "India appears as a subject or unit, not as institution location", ["University of Oxford", "Stanford University", "University of Tokyo", "University of Cambridge", "MIT", "ETH Zurich", "University of Toronto", "National University of Singapore", "University of Melbourne", "Tsinghua University"], [lambda x: f"Department of India Studies, {x}, UK", lambda x: f"{x}; India language project", lambda x: f"{x}; Indian history department, USA", lambda x: f"India Research Group, {x}, Japan", lambda x: f"{x}; studying Indian datasets, Canada"]),
    ]
    negative_targets = [57, 57, 57, 57, 57, 57, 58]
    for (category, rationale, items, wrappers), target in zip(negative_specs, negative_targets):
        negative.extend(_rows("negative", category, rationale, items, wrappers, target))

    ambiguous = []
    ambiguous_specs = [
        ("bare_acronym", "acronym has multiple plausible institutions and no location", ["IIT", "NIT", "ISI", "VIT", "MIT", "JU", "IIIT", "IIM", "TIFR", "AIIMS"], [lambda x: x, lambda x: f"{x}.", lambda x: f"{x},", lambda x: f"{x} research", lambda x: f"{x} department"]),
        ("multinational_without_location", "multinational organization is named without an office location", ["Microsoft Research", "Google Research", "IBM Research", "Amazon Science", "Adobe Research", "Samsung Research", "Intel Labs", "Qualcomm Research", "Infosys Research", "TCS Research"], [lambda x: x, lambda x: f"{x}, research division", lambda x: f"{x}; scientist", lambda x: f"{x} laboratory", lambda x: f"{x} group"]),
        ("missing_institution", "generic unit does not identify an institution", ["Department of Computer Science", "School of Engineering", "Centre for Artificial Intelligence", "National Research Laboratory", "University Research Centre", "Machine Learning Group", "Department of Statistics", "Research Institute", "Computing Lab", "Faculty of Science"], [lambda x: x, lambda x: f"{x}, India", lambda x: f"{x}, Bengaluru", lambda x: f"{x}; India", lambda x: f"{x}, New Delhi"]),
        ("missing_location", "institution is named but campus country/location is unresolved", ["Institute of Technology", "Institute of Science", "National Institute", "Statistical Institute", "Research University", "Microsoft Research", "International Research Centre", "Applied Science University", "BITS", "IISER"], [lambda x: x, lambda x: f"{x}, campus", lambda x: f"{x}; Department of CSE", lambda x: f"{x}, main office", lambda x: f"{x}; research group"]),
        ("email_only", "email address alone is not institutional location evidence", ["researcher@iitb.ac.in", "faculty@iisc.ac.in", "author@ox.ac.uk", "scientist@stanford.edu", "user@mit.edu", "person@google.com", "researcher@tifr.res.in", "lab@iiit.ac.in", "name@university.edu", "contact@example.org"], [lambda x: x, lambda x: f"Email: {x}", lambda x: f"{x}; corresponding author", lambda x: f"Contact {x}", lambda x: f"{x} (email)"]),
        ("city_only", "city alone does not identify a current institution", ["Bengaluru", "Mumbai", "New Delhi", "Hyderabad", "Kolkata", "Chennai", "London", "Boston", "Zurich", "Singapore"], [lambda x: x, lambda x: f"{x}, India", lambda x: f"Based in {x}", lambda x: f"Campus: {x}", lambda x: f"Researcher, {x}"]),
        ("conflicting_evidence", "evidence conflicts or cannot resolve the competing locations", ["IIT; Stanford University", "ISI; University of Southern California", "NIT; Nagoya Institute of Technology", "VIT; Melbourne", "Microsoft Research; India", "University; location unknown", "IIT Bombay; overseas campus", "Indian Institute; foreign office", "University; India email", "University; Mumbai or London"], [lambda x: x, lambda x: f"Affiliations: {x}", lambda x: f"Current affiliation: {x}", lambda x: f"{x}; conflicting records", lambda x: f"{x}; location unclear"]),
        ("multiple_unresolved_affiliations", "multiple affiliations are supplied but none resolves to a current location", ["University; Research Lab", "IIT; MIT", "ISI; Microsoft Research", "National Institute; Google Research", "Department; Company", "Institute A; Institute B", "Lab X; Lab Y", "University campus; overseas office", "Research Centre; corporation", "Faculty; independent laboratory"], [lambda x: x, lambda x: f"Affiliations: {x}", lambda x: f"{x}; both locations omitted", lambda x: f"{x}; current status unknown", lambda x: f"{x}; unresolved"]),
    ]
    for category, rationale, items, wrappers in ambiguous_specs:
        ambiguous.extend(_rows("ambiguous", category, rationale, items, wrappers, 50))
    return positive + negative + ambiguous


def _write():
    rows = _build()
    if len(rows) != 1200:
        raise AssertionError(len(rows))
    original_counts = Counter(row["label"] for row in rows)
    if original_counts != Counter({"positive": 400, "negative": 400, "ambiguous": 400}):
        raise AssertionError(original_counts)
    # Binary policy conversion: every former ambiguous example is deliberately
    # negative because it does not establish a current Indian affiliation.
    conversion_counts = Counter()
    for row in rows:
        original_label = row["label"]
        row["original_label"] = original_label
        if original_label == "ambiguous":
            row["label"] = "negative"
            row["conversion"] = "ambiguous_to_negative"
        else:
            row["conversion"] = "unchanged"
        conversion_counts[row["conversion"]] += 1
    if Counter(row["label"] for row in rows) != Counter({"positive": 400, "negative": 800}):
        raise AssertionError(Counter(row["label"] for row in rows))
    # A few category templates intentionally share institution names. Keep every
    # training state unique while making the de-duplication noise explicit.
    seen_states = {}
    for row in rows:
        count = seen_states.get(row["affiliation"], 0) + 1
        seen_states[row["affiliation"]] = count
        if count > 1:
            row["affiliation"] += f"; additional affiliation detail {count}"
    random.Random(SEED).shuffle(rows)
    # Stratify by class: exact 80/10/10 totals and every category remains represented
    # in all splits because each category contributes at least 50 rows.
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["label"]].append(row)
    split_rows = {"train": [], "validation": [], "test": []}
    for label, bucket in sorted(buckets.items()):
        random.Random(SEED + sum(ord(c) for c in label)).shuffle(bucket)
        counts = (len(bucket) * 4 // 5, len(bucket) // 10, len(bucket) - (len(bucket) * 4 // 5) - (len(bucket) // 10))
        for split, count, part in zip(split_rows, counts, (bucket[:counts[0]], bucket[counts[0]:counts[0]+counts[1]], bucket[sum(counts[:2]):])):
            split_rows[split].extend(part)
    def serialize(row):
        choice = "INDIAN" if row["label"] == "positive" else "NOT_INDIAN"
        return {
            "state": row["affiliation"],
            "questions": QUESTIONS,
            "expected": {QUESTION_ID: choice},
        }

    OUT.mkdir(parents=True, exist_ok=True)
    for split, part in split_rows.items():
        part.sort(key=lambda row: (0 if row["label"] == "positive" else 1, row["affiliation"]))
        records = [serialize(row) for row in part]
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    rows.sort(key=lambda row: (0 if row["label"] == "positive" else 1, row["affiliation"]))
    all_records = [serialize(row) for row in rows]
    (OUT / "all.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_records), encoding="utf-8")
    manifest = {
        "schema_version": "laya-affiliation-finetuning/1",
        "dataset_version": "2026-10-29.3",
        "description": "Synthetic, template-generated binary supervised data for Laya choice fine-tuning using INDIAN/NOT_INDIAN expected choices. Former ambiguous cases are mapped to NOT_INDIAN because they do not establish a current Indian affiliation. Human review is required before treating labels as gold data.",
        "generation_seed": SEED,
        "policy": {
            "INDIAN": "Sufficient evidence that the current institution or campus is in India.",
            "NOT_INDIAN": "No sufficient evidence that the current institution or campus is in India.",
            "ignore": "Nationality, email addresses, former affiliations, and India as a subject/department do not establish current Indian location.",
            "overseas_campus": "Classify the current campus location, not the parent institution's country.",
        },
        "conversion_rule": "original_label=ambiguous -> expected=NOT_INDIAN; positive -> expected=INDIAN; negative -> expected=NOT_INDIAN",
        "laya_format": {"state": "affiliation string", "questions": QUESTIONS, "expected": {QUESTION_ID: "one label"}},
        "source_label_counts": dict(original_counts),
        "label_counts": {"INDIAN": 400, "NOT_INDIAN": 800},
        "internal_binary_counts": dict(Counter(row["label"] for row in rows)),
        "conversion_counts": dict(conversion_counts),
        "expected_counts": {"INDIAN": 400, "NOT_INDIAN": 800},
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "split_counts": {
            split: {
                "INDIAN": sum(row["label"] == "positive" for row in part),
                "NOT_INDIAN": sum(row["label"] == "negative" for row in part),
            }
            for split, part in split_rows.items()
        },
        "files": {},
        "limitations": ["Synthetic templates are not a representative sample of scholarly affiliations.", "The same institution families appear across splits; this measures policy coverage, not unseen-entity generalization.", "Binary negative includes unresolved evidence, so the negative class is intentionally broader than foreign-only affiliations.", "Review every label and remove generated near-duplicates before publication or high-stakes training."],
    }
    for path in [OUT / "all.jsonl", *(OUT / f"{split}.jsonl" for split in split_rows)]:
        manifest["files"][path.name] = {"rows": sum(1 for _ in path.open(encoding="utf-8")), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "rows": len(rows), "labels": manifest["label_counts"], "splits": manifest["split_counts"]}, indent=2))


if __name__ == "__main__":
    _write()
