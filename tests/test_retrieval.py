"""Retriever: which memories and past sessions reach the LLM, and which are kept out."""
from __future__ import annotations

from app.models import Memory, MemoryOp, Message, SessionSummary


def save(key: str, value: str) -> MemoryOp:
    return MemoryOp(action="save", key=key, value=value)


async def retrieve(container, message: str, recent=None, user="u", session="current"):
    return await container.chat._retriever.retrieve(user, session, message, recent or [])


async def seed(container, *ops: MemoryOp, user="u"):
    await container.memory_manager.apply(user, "seed", list(ops))


async def test_unrelated_question_pulls_in_no_memories(container):
    await seed(container, save("interests", "cricket, cooking"), save("career_goal", "switch jobs"))
    result = await retrieve(container, "Tell me about Saturn")
    assert result.memories == [] and result.context_used == []
    assert sorted(result.debug["excluded"]) == ["career_goal", "interests"]   # considered, then rejected


async def test_standing_language_preference_is_always_included(container):
    await seed(container, save("language_preference", "Hindi"), save("interests", "cricket"))
    result = await retrieve(container, "Tell me about Saturn")
    assert result.context_used == ["language_preference"]
    assert result.memories[0].reasons == ["always_on"]


async def test_category_match_beats_everything_else_for_a_career_question(container):
    await seed(container, save("career_goal", "switch jobs"), save("interests", "cricket"), save("location", "Pune"))
    result = await retrieve(container, "What should I focus on for my career?")
    assert result.context_used == ["career_goal"]
    assert sorted(result.debug["excluded"]) == ["interests", "location"]      # semantic candidates, rejected


async def test_semantic_match_alone_can_qualify_a_memory(make_container):
    container = make_container(memory_min_similarity=0.3)                    # fake embedder: word overlap only
    await seed(container, save("interests", "astronomy telescope stargazing"))
    result = await retrieve(container, "Which telescope should I buy?")      # no category keyword in the question
    assert result.context_used == ["interests"]
    assert result.memories[0].reasons == ["semantic"]


async def test_short_followup_inherits_the_previous_questions_topic(container):
    await seed(container, save("career_goal", "switch jobs"), save("location", "Pune"))
    recent = [Message(role="user", content="What should I focus on for my career?"),
              Message(role="assistant", content="Think about timing.")]

    followup = await retrieve(container, "Why do you say that?", recent)
    assert followup.debug["followup"] is True
    assert followup.context_used == ["career_goal"]

    alone = await retrieve(container, "Why do you say that?")                 # no history: nothing to inherit
    assert alone.context_used == []


async def test_what_do_you_remember_returns_everything_up_to_the_cap(make_container):
    container = make_container(memory_recall_max=2)
    await seed(container, save("language_preference", "Hindi"), save("career_goal", "switch jobs"),
               save("interests", "cricket"), save("location", "Pune"))

    result = await retrieve(container, "What do you remember about me?")
    assert result.debug["recall_intent"] is True
    assert len(result.memories) == 3                                          # always-on + the 2 best of the rest
    assert result.memories[0].memory.key == "language_preference"
    assert len(result.debug["excluded"]) == 1


async def test_recall_question_with_a_category_stays_on_topic(container):
    await seed(container, save("career_goal", "switch jobs"), save("interests", "cricket"))
    result = await retrieve(container, "What do you remember about my career plans?")
    assert result.context_used == ["career_goal"]


async def test_vector_hit_without_a_mongo_document_is_skipped(container):
    # a stale index entry: present in Chroma, gone from MongoDB
    await container.vectors.upsert_memory(
        Memory(user_id="u", key="location", value="Pune city", category="personal", kind="slot"))
    result = await retrieve(container, "Pune city location")
    assert result.memories == []


async def test_memories_of_other_users_are_never_candidates(container):
    await seed(container, save("career_goal", "switch jobs"), user="alice")
    result = await retrieve(container, "What should I focus on for my career?", user="bob")
    assert result.memories == []


async def _summary(container, session_id: str, text: str, user="u"):
    await container.vectors.upsert_summary(SessionSummary(user_id=user, session_id=session_id, summary=text))


async def test_past_session_needs_to_be_similar_enough_unless_the_user_asks_to_recall(container):
    await _summary(container, "old", "career timing and astrology guidance")

    unrelated = await retrieve(container, "Tell me about Saturn")
    assert unrelated.past_sessions == []

    related = await retrieve(container, "career timing question")
    assert [p.session_id for p in related.past_sessions] == ["old"]

    recall = await retrieve(container, "What did we discuss earlier about guidance?")   # lower bar when recalling
    assert [p.session_id for p in recall.past_sessions] == ["old"]


async def test_current_session_and_other_users_summaries_are_excluded(container):
    await _summary(container, "current", "career timing and astrology guidance")
    await _summary(container, "theirs", "career timing and astrology guidance", user="someone-else")
    result = await retrieve(container, "career timing question", session="current")
    assert result.past_sessions == []


# ---- relative margin: real embedding scores are compressed, so "above the floor" is not enough ----------------------
async def test_weaker_semantic_hit_is_dropped_when_a_much_better_one_exists(make_container):
    loose = dict(memory_min_similarity=0.2)                                    # both clear the absolute floor ...
    strict = make_container(semantic_margin=0.1, **loose)
    off = make_container(semantic_margin=1.0, **loose)                         # ... margin disabled
    for container in (strict, off):
        await seed(container, save("interests", "astronomy telescope stargazing"), save("location", "telescope shop downtown"))

    question = "telescope stargazing astronomy"                                # no category keyword
    assert (await retrieve(strict, question)).context_used == ["interests"]
    assert sorted((await retrieve(off, question)).context_used) == ["interests", "location"]
    assert (await retrieve(strict, question)).debug["top_similarity"] > 0.8


async def test_weaker_past_session_is_dropped_when_a_much_better_one_exists(container):
    await _summary(container, "close", "career timing astrology guidance")
    await _summary(container, "weak", "guidance about gardening")
    result = await retrieve(container, "career timing astrology guidance question")
    assert [p.session_id for p in result.past_sessions] == ["close"]
