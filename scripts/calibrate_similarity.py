"""Measure how your embedding model separates relevant from unrelated memories / session summaries.

    python -m scripts.calibrate_similarity            # uses .env (EMBEDDING_BACKEND / EMBEDDING_MODEL)
    python -m scripts.calibrate_similarity --margin 0.1

Cosine scores from real embedding models are compressed (e.g. relevant ~0.90, unrelated ~0.79), so the
retrieval thresholds must come from data. This prints every (question, memory) score with its label, then
sweeps the absolute threshold and shows precision/recall for the rule the retriever uses:

    include  if  sim >= MIN_SIMILARITY  and  sim >= best_sim_for_this_question - SEMANTIC_MARGIN

Edit the labelled cases below to match your real users' phrasing before trusting the numbers: the shipped set is
small (a sanity check, not a benchmark). Category keyword matching still rescues obvious cases at runtime, so a
slightly conservative semantic threshold costs little.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from app.config import Settings
from app.container import build_embedder

# text exactly as it is embedded in production: "<label>: <value>"
MEMORIES = {
    "career_goal": "career goal: planning to switch jobs next year",
    "interests": "interests: cricket, cooking",
    "language_preference": "preferred language: Hindi",
    "location": "location: Pune",
    "financial_goal": "financial goal: plan to buy a house in two years",
    "relationship_status": "relationship status: single",
    "health_focus": "health focus: chronic back pain",
}

# (question, keys of the memories that ARE relevant to it)
MEMORY_QUESTIONS = [
    ("What should I focus on for my career?", {"career_goal"}),
    ("Should I change my job this year?", {"career_goal"}),
    ("मेरे करियर के बारे में क्या सलाह है?", {"career_goal"}),
    ("meri naukri ke baare mein batao", {"career_goal"}),
    ("Which hobbies suit me best?", {"interests"}),
    ("Will I be able to buy a house soon?", {"financial_goal"}),
    ("When will I get married?", {"relationship_status"}),
    ("How can I improve my health?", {"health_focus"}),
    # paraphrases with NO category keyword: only the semantic path can find these
    ("Is now the right moment to move to a new organisation?", {"career_goal"}),
    ("Will I be able to afford my own place soon?", {"financial_goal"}),
    ("Why does my back keep hurting?", {"health_focus"}),
    ("What should I do on weekends to relax?", {"interests"}),
    ("Tell me about Saturn", set()),
    ("What is my lucky number?", set()),
]

SUMMARIES = {
    "career_talk": "User is planning to switch jobs next year and is nervous about the timing. Topics: career.",
    "money_talk": "User asked whether buying a house in two years is a good idea. Topics: finance.",
}
SUMMARY_QUESTIONS = [
    ("What did we talk about regarding my job change?", {"career_talk"}),
    ("Remind me what you said about changing careers", {"career_talk"}),
    ("मैंने नौकरी बदलने के बारे में क्या पूछा था?", {"career_talk"}),
    ("What did you say about buying a house?", {"money_talk"}),
    ("Did I ask you about moving to a new company?", {"career_talk"}),
    ("Tell me about Saturn", set()),
    ("Will I get married soon?", set()),
]


@dataclass
class Row:
    question: str
    key: str
    sim: float
    relevant: bool
    best: float  # best similarity for this question (the margin is relative to it)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))  # embeddings are L2-normalised


def score(embedder, docs: dict[str, str], questions) -> list[Row]:
    keys = list(docs)
    vectors = dict(zip(keys, embedder.embed_documents([docs[k] for k in keys])))
    rows = []
    for question, relevant in questions:
        qv = embedder.embed_query(question)
        sims = {k: _dot(qv, v) for k, v in vectors.items()}
        best = max(sims.values())
        rows += [Row(question, k, s, k in relevant, best) for k, s in sims.items()]
    return rows


def evaluate(rows: list[Row], threshold: float, margin: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for r in rows:
        included = r.sim >= threshold and r.sim >= r.best - margin
        tp += included and r.relevant
        fp += included and not r.relevant
        fn += (not included) and r.relevant
    return tp, fp, fn


def report(title: str, rows: list[Row], margin: float) -> dict:
    print(f"\n=== {title} " + "=" * (66 - len(title)))
    question = None
    for r in rows:
        if r.question != question:
            question = r.question
            print(f"\n  {question}")
        flag = "RELEVANT " if r.relevant else "         "
        print(f"    {r.sim:.3f}  {flag}{r.key}")

    rel = [r.sim for r in rows if r.relevant]
    irr = [r.sim for r in rows if not r.relevant]
    print(f"\n  relevant   : min {min(rel):.3f}  mean {sum(rel) / len(rel):.3f}  (n={len(rel)})")
    print(f"  unrelated  : max {max(irr):.3f}  mean {sum(irr) / len(irr):.3f}  (n={len(irr)})")
    gap = min(rel) - max(irr)
    print(f"  gap (lowest relevant - highest unrelated): {gap:+.3f}"
          + ("   <- separable by a single threshold" if gap > 0 else "   <- overlap: the relative margin matters"))

    print(f"\n  sweep (margin={margin}):  threshold  precision  recall   F1")
    best = (-1.0, 0.0)
    for i in range(50, 100, 2):
        t = i / 100
        tp, fp, fn = evaluate(rows, t, margin)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 >= best[0]:
            best = (f1, t)
        print(f"                          {t:.2f}       {precision:.2f}      {recall:.2f}    {f1:.2f}")
    print(f"\n  best F1 {best[0]:.2f} at threshold {best[1]:.2f}  (F1 weighs a leak and a miss equally)")

    rec = recommend(rows, margin)
    if rec is None:
        print("  -> no threshold gives >=95% precision with any recall: check the embedder / prefixes before tuning")
        return {"threshold": best[1], "max_unrelated": max(irr)}
    low, mid, high, recall = rec
    print(f"  -> PRECISION-FIRST: thresholds {low:.3f}-{high:.3f} keep >=95% precision at recall {recall:.2f}; "
          f"recommend {mid:.3f} (middle of that range)")
    return {"threshold": mid, "max_unrelated": max(irr)}


def recommend(rows: list[Row], margin: float, min_precision: float = 0.95):
    """Highest-recall thresholds that still keep precision >= min_precision; returns (low, middle, high, recall).

    A wrong memory in the prompt is costlier than a missed one (category keywords rescue many misses), and the
    middle of the working range is safer than its edge.
    """
    best_recall, ok = 0.0, []
    for i in range(500, 991, 5):
        t = i / 1000
        tp, fp, fn = evaluate(rows, t, margin)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision < min_precision or recall == 0.0:
            continue
        if recall > best_recall + 1e-9:
            best_recall, ok = recall, [t]
        elif abs(recall - best_recall) <= 1e-9:
            ok.append(t)
    return (ok[0], ok[len(ok) // 2], ok[-1], best_recall) if ok else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings = Settings()
    parser.add_argument("--margin", type=float, default=settings.semantic_margin)
    args = parser.parse_args()

    embedder = build_embedder(settings)
    print(f"embedding backend: {settings.embedding_backend} ({settings.embedding_model})")
    mem = report("MEMORIES", score(embedder, MEMORIES, MEMORY_QUESTIONS), args.margin)
    summ = report("SESSION SUMMARIES", score(embedder, SUMMARIES, SUMMARY_QUESTIONS), args.margin)

    print("\nSuggested .env (re-run after editing the cases above to match real phrasing):")
    print(f"  MEMORY_MIN_SIMILARITY={mem['threshold']:.2f}")
    print(f"  SUMMARY_MIN_SIMILARITY={summ['threshold']:.2f}")
    print(f"  SUMMARY_RECALL_MIN_SIMILARITY={summ['max_unrelated'] + 0.015:.2f}   "
          "(just above the highest unrelated summary score seen)")
    print(f"  SEMANTIC_MARGIN={args.margin}")


if __name__ == "__main__":
    main()
