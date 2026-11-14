"""Cheap, rule-based query understanding used by retrieval.

Deliberately simple: keyword stems (English + common Hindi/Hinglish) map a question to
memory categories, so "What should I focus on for my career?" prioritises career
memories. Semantic search then covers whatever the keywords miss.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9ऀ-ॿ']+")

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "career": (
        "job", "jobs", "career", "work", "promotion", "promoted", "salary", "office", "boss",
        "interview", "resume", "profession", "switch", "business", "company", "hike", "resign",
        "employer", "startup", "colleague", "naukri", "kaam", "नौकरी", "करियर", "व्यवसाय", "कारोबार",
    ),
    "finance": (
        "money", "finance", "financial", "invest", "investment", "saving", "savings", "loan",
        "debt", "income", "wealth", "paisa", "paise", "stock", "property", "dhan", "पैसा", "धन",
    ),
    "relationships": (
        "marriage", "marry", "married", "love", "partner", "relationship", "spouse", "wife",
        "husband", "girlfriend", "boyfriend", "dating", "shaadi", "vivah", "pyaar", "divorce",
        "family", "शादी", "विवाह", "प्यार",
    ),
    "health": (
        "health", "sick", "illness", "disease", "fitness", "sehat", "stress", "diet", "wellness",
        "सेहत", "स्वास्थ्य",
    ),
    "interests": ("hobby", "hobbies", "interest", "interests", "passion", "enjoy", "free time"),
    "preferences": ("language", "prefer", "preference", "preferences"),
    "personal": ("birth", "born", "birthday", "janam", "kundli", "city", "location", "जन्म", "कुंडली"),
}

_RECALL_RE = re.compile(
    r"\b(remember|recall|know about me|told you|i told|yaad|last time|previous(?:ly)?|"
    r"earlier|talked about|discussed|mentioned)\b|what do you know",
    re.IGNORECASE,
)

_FOLLOWUP_RE = re.compile(
    r"^\s*(why|how so|what about|and|also|then|so|but|really|kyun|kyu|kaise|aur|explain|"
    r"elaborate|tell me more|what do you mean|can you explain|how come)\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def detect_categories(text: str) -> list[str]:
    """Categories whose keywords appear in the text (word match, or prefix for stems >= 4 chars)."""
    tokens = tokenize(text)
    found: list[str] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if " " in kw:
                if kw in text.lower():
                    found.append(category)
                    break
                continue
            if any(t == kw or (len(kw) >= 4 and t.startswith(kw)) for t in tokens):
                found.append(category)
                break
    return found


def has_recall_intent(text: str) -> bool:
    return bool(_RECALL_RE.search(text))


def is_followup(text: str) -> bool:
    """Short or connective messages ('Why do you say that?') depend on the previous turn."""
    return bool(_FOLLOWUP_RE.match(text)) or len(tokenize(text)) <= 4
