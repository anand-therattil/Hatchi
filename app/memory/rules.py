"""Deterministic, high-precision memory extraction.

Runs before the LLM. A 1.5-3B model is good at *finding* odd facts but unreliable at
consistent formatting, so obvious patterns (language preference, job-switch plans, ...)
are handled here and the LLM only fills the gaps.

Also provides the "gate": only declarative, first-person-ish sentences are even
considered, so questions like "What should I focus on for my career?" never become memories.
"""
from __future__ import annotations

import re

from app.models import MemoryOp

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r",?\s+(?:and|but|also)\s+(?=i\b|i'|my\b|please\b)", re.IGNORECASE)

# questions and imperative requests ("Tell me about Saturn") are never statements about the user
_QUESTION_START = re.compile(
    r"^\s*(what|why|how|when|where|who|which|can|could|should|will|would|do|does|did|is|are|am|"
    r"kya|kab|kaise|kyun|kyu|kaun|kaha|"
    r"tell|show|give|explain|predict|suggest|help|check|read|describe|list|say)\b",
    re.IGNORECASE,
)
_SELF_CUE = re.compile(
    r"\b(i|i'm|im|i've|i'd|i'll|my|mine|we|our|mujhe|mera|meri|mere|main|hum|please|always|"
    r"from now on)\b",
    re.IGNORECASE,
)

LANGS = (
    "hindi|english|hinglish|tamil|telugu|marathi|bengali|gujarati|kannada|malayalam|punjabi|urdu|odia"
)
_LANG_IN = re.compile(rf"\b(?:in|using)\s+({LANGS})\b", re.IGNORECASE)
_LANG_MEIN = re.compile(rf"\b({LANGS})\s+(?:mein|me|language|please|only)\b", re.IGNORECASE)
_LANG_VERB = re.compile(
    r"\b(prefer\w*|want|wish|like|reply|respond|answer|advice|advise|speak|talk|chat|explain|tell|"
    r"write|guidance|suggestions?|comfortable|language|batao|bolo)\b",
    re.IGNORECASE,
)
_NEGATION_BEFORE = re.compile(r"(?:\bnot|n't|\bno)\s+(?:\w+\s+){0,2}$", re.IGNORECASE)

_NEGATION = re.compile(
    r"\b(no longer|not anymore|any ?more|changed my mind|dropped|abandon\w*|give up|gave up|"
    r"not planning|don't want|dont want|not going|no plans)\b",
    re.IGNORECASE,
)

_CAREER_SWITCH = re.compile(
    r"\b(?:plan(?:ning|s|ned)?|want(?:s|ing)?|hop(?:e|es|ing)|think(?:ing)?|look(?:ing)?|aim(?:ing)?|"
    r"try(?:ing)?|going|decid(?:ed|ing)|intend(?:ing)?)\s+(?:to\s+)?(?:switch|change|quit|leave|move)\b"
    r".*?\b(?:job|jobs|career|company|companies|organi[sz]ation|field|role|profession)\b[^.;!?]*",
    re.IGNORECASE,
)
_CAREER_BUSINESS = re.compile(
    r"\b(?:plan|want|hope|aim|try|look|going|intend)\w*\s+(?:to\s+)?(?:start|launch|begin|open)\s+"
    r"(?:my\s+own|a|an|my)\s+(?:own\s+)?(?:business|startup|company|firm)\b[^.;!?]*",
    re.IGNORECASE,
)
_CAREER_PROMO = re.compile(
    r"\b(?:plan|want|hope|aim|try|look|going)\w*\s+(?:to\s+)?(?:get\s+)?(?:a\s+)?promot\w+[^.;!?]*",
    re.IGNORECASE,
)

_JOB_WORK = re.compile(r"\b((?:work|working|employed)\s+(?:as|at|in|with|for)\s+[^.,;!?]+)", re.IGNORECASE)
_JOB_ROLE = re.compile(
    r"\bi(?:'m| am)\s+an?\s+([a-z ]{0,30}?(?:engineer|developer|manager|doctor|teacher|designer|"
    r"analyst|student|lawyer|accountant|consultant|nurse|founder|entrepreneur|scientist))\b",
    re.IGNORECASE,
)
_LOCATION = re.compile(
    r"\b(?:i live in|i'm from|i am from|i'm based in|i am based in|i stay in|i reside in|living in)\s+"
    r"([A-Za-z][A-Za-z ]{1,30}?)(?=\s+(?:and|but)\b|[.,;!?]|$)",
    re.IGNORECASE,
)
_BIRTH = re.compile(
    r"\b(?:i was )?(born\b[^.;!?]*|my (?:date of birth|dob|birth ?date|birth ?time|birth ?place)\b[^.;!?]*)",
    re.IGNORECASE,
)
_REL_STATUS = re.compile(
    r"\bi(?:'m| am)\s+(single|married|engaged|divorced|widowed|separated|in a relationship)\b",
    re.IGNORECASE,
)
_REL_GOAL = re.compile(
    r"\b(?:plan|want|hope|look|think)\w*\s+(?:to\s+)?(?:get married|marry|have a baby|settle down|"
    r"get engaged)\b[^.;!?]*",
    re.IGNORECASE,
)
_FIN_GOAL = re.compile(
    r"\b(?:plan|want|aim|hope|try|sav)\w*\s+(?:to\s+)?(?:buy|purchase|invest|save|build|clear|repay)\b"
    r"[^.;!?]*\b(?:house|home|flat|apartment|car|property|land|shares|stocks|gold|mutual funds?|"
    r"savings|crore|lakh|loan|debt)\b[^.;!?]*",
    re.IGNORECASE,
)
_INTERESTS = re.compile(
    r"\b(?:i (?:really )?(?:love|enjoy|adore)|i(?:'m| am) (?:really )?(?:interested in|into|passionate about)|"
    r"my hobb(?:y|ies) (?:is|are|include))\s+([^.;!?]+)",
    re.IGNORECASE,
)


def candidate_statements(message: str) -> list[str]:
    """Sentences worth extracting from: declarative and self-referential (the 'gate')."""
    out = []
    for sentence in _SENTENCE_SPLIT.split(message.strip()):
        s = sentence.strip()
        if not s or s.endswith("?") or _QUESTION_START.match(s):
            continue
        if _SELF_CUE.search(s):
            out.append(s)
    return out


def _clauses(statement: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(statement) if c.strip()]


def _clean_value(text: str) -> str:
    text = re.sub(r"^(?:and|but|also|so|actually)\s+", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^(?:i am|i'm|im|i)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" ,.;!").strip()


def _language(statement: str) -> str | None:
    if not _LANG_VERB.search(statement):
        return None
    for pattern in (_LANG_IN, _LANG_MEIN):
        for m in pattern.finditer(statement):
            if _NEGATION_BEFORE.search(statement[: m.start()][-18:]):
                continue
            return m.group(1).capitalize()
    return None


def rule_extract(statements: list[str]) -> list[MemoryOp]:
    """Return memory ops for the obvious patterns found in the given statements."""
    ops: dict[str, MemoryOp] = {}

    def put(key: str, value: str, action: str = "save") -> None:
        if value or action == "delete":
            ops[key] = MemoryOp(action=action, key=key, value=value, explicit=True)

    for statement in statements:
        lang = _language(statement)
        if lang:
            put("language_preference", lang)

        for clause in _clauses(statement):
            career = _CAREER_SWITCH.search(clause) or _CAREER_BUSINESS.search(clause) or _CAREER_PROMO.search(clause)
            if career:
                if _NEGATION.search(clause):
                    put("career_goal", "", "delete")
                else:
                    put("career_goal", _clean_value(career.group(0)))

            if m := (_JOB_WORK.search(clause) or _JOB_ROLE.search(clause)):
                put("current_job", _clean_value(m.group(1)))
            if m := _LOCATION.search(clause):
                put("location", m.group(1).strip())
            if m := _BIRTH.search(clause):
                put("birth_details", _clean_value(m.group(1)))
            if m := _REL_STATUS.search(clause):
                put("relationship_status", m.group(1).lower())
            if m := _REL_GOAL.search(clause):
                put("relationship_goal", _clean_value(m.group(0)))
            if m := _FIN_GOAL.search(clause):
                put("financial_goal", _clean_value(m.group(0)))
            if m := _INTERESTS.search(clause):
                part = re.split(r"\s+(?:and|but)\s+(?:i|my)\b", m.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
                items = [i.strip(" .") for i in re.split(r",\s*|\s+and\s+", part) if i.strip(" .")]
                put("interests", ", ".join(items))

    return list(ops.values())
