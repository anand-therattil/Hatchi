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


def report(title: str, rows: list[Row], margin: float) -> float:
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
    if best[0] == 0.0:
        print("\n  -> no threshold retrieves anything relevant: check the embedder / prefixes before tuning")
    else:
        print(f"\n  -> best F1 {best[0]:.2f} at threshold {best[1]:.2f}")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings = Settings()
    parser.add_argument("--margin", type=float, default=settings.semantic_margin)
    args = parser.parse_args()

    embedder = build_embedder(settings)
    print(f"embedding backend: {settings.embedding_backend} ({settings.embedding_model})")
    mem_t = report("MEMORIES", score(embedder, MEMORIES, MEMORY_QUESTIONS), args.margin)
    sum_t = report("SESSION SUMMARIES", score(embedder, SUMMARIES, SUMMARY_QUESTIONS), args.margin)

    print("\nSuggested .env (re-run after editing the cases above to match real phrasing):")
    print(f"  MEMORY_MIN_SIMILARITY={mem_t:.2f}")
    print(f"  SUMMARY_MIN_SIMILARITY={sum_t:.2f}")
    print(f"  SEMANTIC_MARGIN={args.margin}")
    print("  (SUMMARY_RECALL_MIN_SIMILARITY: set a few points below SUMMARY_MIN_SIMILARITY)")


if __name__ == "__main__":
    main()
