"""The closed set of things we remember about a user.

A fixed slot vocabulary matters because small local LLMs invent inconsistent key
names ("job_plan", "career_plans", ...) when left unconstrained. With a closed set,
a correction is simply an upsert on the same key.

Kinds:
  slot  one value per key; a new value replaces (and archives) the old one
  list  values are merged and de-duplicated (e.g. interests)
  fact  free-text notes; key is ``fact_<hash>`` and near-duplicates are merged
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

FACT_KEY_RE = re.compile(r"^fact_[0-9a-f]{8}$")
NEW_FACT_KEY = "other_fact"


@dataclass(frozen=True)
class SlotSpec:
    key: str
    category: str
    kind: str  # slot | list | fact
    label: str
    description: str


_SPECS = [
    SlotSpec("language_preference", "preferences", "slot", "preferred language",
             "Language the user wants replies in (e.g. Hindi, English)."),
    SlotSpec("career_goal", "career", "slot", "career goal",
             "Career plans or goals (switching jobs, promotion, starting a business)."),
    SlotSpec("current_job", "career", "slot", "current job",
             "The user's present occupation, employer or study."),
    SlotSpec("financial_goal", "finance", "slot", "financial goal",
             "Money goals (buying a house, saving, investing, clearing debt)."),
    SlotSpec("relationship_status", "relationships", "slot", "relationship status",
             "Single, married, engaged, etc."),
    SlotSpec("relationship_goal", "relationships", "slot", "relationship goal",
             "Plans about marriage, partner, family."),
    SlotSpec("health_focus", "health", "slot", "health focus",
             "Health concern or wellness goal."),
    SlotSpec("location", "personal", "slot", "location",
             "Where the user lives."),
    SlotSpec("birth_details", "personal", "slot", "birth details",
             "Date, time or place of birth."),
    SlotSpec("interests", "interests", "list", "interests",
             "Hobbies and things the user enjoys."),
    SlotSpec(NEW_FACT_KEY, "personal", "fact", "note",
             "Any other important personal fact not covered by another key."),
]

SLOT_SPECS: dict[str, SlotSpec] = {s.key: s for s in _SPECS}

# Slots that are injected into every prompt when present (standing preferences).
ALWAYS_INCLUDE_KEYS: tuple[str, ...] = ("language_preference",)

CATEGORIES: tuple[str, ...] = tuple(sorted({s.category for s in _SPECS}))


def spec_for(key: str) -> Optional[SlotSpec]:
    """Return the spec for a key, or None if the key is not part of the vocabulary."""
    if key in SLOT_SPECS:
        return SLOT_SPECS[key]
    if FACT_KEY_RE.match(key):
        return SLOT_SPECS[NEW_FACT_KEY]
    return None


def valid_key(key: str) -> bool:
    return spec_for(key) is not None
