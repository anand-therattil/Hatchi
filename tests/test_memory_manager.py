"""MemoryManager: update semantics, confidence policy, de-duplication and index recovery."""
from __future__ import annotations

from app.models import MemoryOp


def save(key: str, value: str, explicit: bool = True) -> MemoryOp:
    return MemoryOp(action="save", key=key, value=value, explicit=explicit)


async def test_list_memories_merge_without_duplicates(container):
    mm, store = container.memory_manager, container.memory_store
    await mm.apply("u", "s", [save("interests", "cricket")])
    await mm.apply("u", "s", [save("interests", "Cooking, cricket")])

    memory = await store.get("u", "interests")
    assert memory.value == "cricket, Cooking" and memory.version == 2

    assert await mm.apply("u", "s", [save("interests", "COOKING")]) == []      # nothing new: no-op
    assert (await store.get("u", "interests")).version == 2


async def test_inferred_value_never_overwrites_an_explicit_one(container):
    mm, store = container.memory_manager, container.memory_store
    await mm.apply("u", "s", [save("language_preference", "Hindi", explicit=True)])

    assert await mm.apply("u", "s", [save("language_preference", "English", explicit=False)]) == []
    assert (await store.get("u", "language_preference")).value == "Hindi"

    changes = await mm.apply("u", "s", [save("language_preference", "English", explicit=True)])
    assert [c.action for c in changes] == ["updated"] and changes[0].previous == "Hindi"


async def test_history_is_capped_and_versions_keep_counting(container):
    mm, store = container.memory_manager, container.memory_store
    for i in range(12):
        await mm.apply("u", "s", [save("career_goal", f"goal number {i}")])

    memory = await store.get("u", "career_goal")
    assert memory.version == 12 and len(memory.history) == 10
    assert memory.history[-1]["value"] == "goal number 10"                    # newest archived value last


async def test_near_duplicate_fact_updates_the_existing_note(container):
    mm, store = container.memory_manager, container.memory_store
    await mm.apply("u", "s", [save("other_fact", "has a younger sister named Meera")])
    await mm.apply("u", "s", [save("other_fact", "has a younger sister named Meera.")])   # same note, new wording
    await mm.apply("u", "s", [save("other_fact", "works night shifts at a hospital")])    # genuinely different

    facts = [m for m in await store.list("u") if m.kind == "fact"]
    assert len(facts) == 2
    assert sorted(m.version for m in facts) == [1, 2]
    assert all(m.key.startswith("fact_") for m in facts)


async def test_invalid_operations_are_ignored(container):
    mm, store = container.memory_manager, container.memory_store
    changes = await mm.apply("u", "s", [
        save("made_up_key", "anything"),
        save("career_goal", "   "),
        MemoryOp(action="delete", key="location"),                            # nothing to delete
    ])
    assert changes == [] and await store.list("u") == []


async def test_one_failing_operation_does_not_lose_the_others(container, monkeypatch):
    mm, store = container.memory_manager, container.memory_store
    real_save = store.save

    async def flaky_save(memory):
        if memory.key == "location":
            raise RuntimeError("db hiccup")
        await real_save(memory)

    monkeypatch.setattr(store, "save", flaky_save)
    changes = await mm.apply("u", "s", [save("location", "Pune"), save("career_goal", "switch jobs")])
    assert [c.key for c in changes] == ["career_goal"]


async def test_failed_index_write_is_recovered_by_reindex(container, monkeypatch):
    mm, store, vectors = container.memory_manager, container.memory_store, container.vectors
    real_upsert = vectors.upsert_memory

    async def broken(memory):
        raise ConnectionError("chroma is down")

    with monkeypatch.context() as patch:
        patch.setattr(vectors, "upsert_memory", broken)
        await mm.apply("u", "s", [save("career_goal", "switch jobs next year")])

    memory = await store.get("u", "career_goal")
    assert memory is not None and memory.indexed is False                    # Mongo has it, the index does not
    assert await vectors.search_memories("u", "switch jobs", k=3) == []

    assert await mm.reindex_pending() == 1
    assert (await store.get("u", "career_goal")).indexed is True
    assert [h.key for h in await vectors.search_memories("u", "switch jobs", k=3)] == ["career_goal"]


async def test_delete_removes_the_memory_from_store_and_index(container):
    mm, store, vectors = container.memory_manager, container.memory_store, container.vectors
    await mm.apply("u", "s", [save("career_goal", "switch jobs")])

    assert await mm.delete("u", "career_goal") is True
    assert await mm.delete("u", "career_goal") is False
    assert await store.get("u", "career_goal") is None
    assert await vectors.search_memories("u", "switch jobs", k=3) == []
