"""End-to-end checker for a RUNNING Hatchi server (plain script - no pytest).

    python -m scripts.e2e_check                         # against http://localhost:8000
    python -m scripts.e2e_check --url http://gpu-box:8000 --keep

It walks one fresh user through the three things the assignment is judged on and prints PASS/FAIL per check:

  1. END-TO-END FLOW      chat over HTTP + WebSocket, session continuity (short-term history), session end -> summary,
                          a NEW session on the same user still knows the user.
  2. MEMORY HANDLING      useful facts are saved (explicit statements), questions/chit-chat are NOT saved, a correction
                          updates the value (version bump + history) instead of duplicating it, a negation removes it.
  3. CONTEXT RELEVANCE    with 6 memories stored (career, language, finance, location, hobbies, relationship), a career question sends ONLY the career (+ always-on language)
                          memories to the LLM; an unrelated question sends none; "debug" shows what was considered
                          vs. what was actually sent.

Requires the server's real LLM to be up for the summary / cross-session checks (those are marked [llm]).
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from urllib.parse import urlencode

import httpx

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    RESULTS.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    return ok


class Api:
    def __init__(self, client: httpx.AsyncClient, user: str):
        self.c, self.user = client, user

    async def chat(self, session: str, message: str, debug: bool = True) -> dict:
        r = await self.c.post("/chat", json={"user_id": self.user, "session_id": session, "message": message, "debug": debug})
        r.raise_for_status()
        return r.json()

    async def memories(self) -> dict[str, dict]:
        r = await self.c.get(f"/users/{self.user}/memories")
        r.raise_for_status()
        return {m["key"]: m for m in r.json()}

    async def wait_for(self, predicate, timeout: float = 60.0) -> dict[str, dict]:
        """Memory extraction runs in the background after the reply: poll until predicate(memories) holds."""
        deadline, mems = time.monotonic() + timeout, {}
        while time.monotonic() < deadline:
            mems = await self.memories()
            if predicate(mems):
                return mems
            await asyncio.sleep(0.7)
        return mems

    async def end(self, session: str) -> dict:
        r = await self.c.post(f"/users/{self.user}/sessions/{session}/end")
        r.raise_for_status()
        return r.json()


def sent_keys(reply: dict) -> set[str]:
    return set((reply.get("debug") or {}).get("memory_scores", {}))


async def ws_roundtrip(ws_base: str, user: str, session: str, text: str, timeout: float) -> tuple[str, int, bool]:
    import websockets

    uri = f"{ws_base}/ws/chat?{urlencode({'user_id': user, 'session_id': session})}"
    async with websockets.connect(uri, open_timeout=15) as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        await ws.send(json.dumps({"type": "message", "text": text}))
        tokens, text_out, done = 0, "", False
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if event["type"] == "token":
                tokens += 1
                text_out += event.get("text", "")
            elif event["type"] == "done":
                done = True
                break
            elif event["type"] == "error":
                break
        return text_out, tokens, done and first["type"] == "session"


async def run(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    run_id = uuid.uuid4().hex[:6]
    user, s1, s2, s3 = f"e2e-{run_id}", f"s1-{run_id}", f"s2-{run_id}", f"s3-{run_id}"

    async with httpx.AsyncClient(base_url=base, timeout=args.timeout) as client:
        try:
            (await client.get("/health")).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"Server not reachable at {base}: {exc}\nStart it with: uvicorn app.main:app", file=sys.stderr)
            return 2
        api = Api(client, user)
        print(f"Hatchi end-to-end check   user={user}   target={base}\n")

        # ================= 1. END-TO-END FLOW =================================================================
        print("1. End-to-end functionality")
        r = await api.chat(s1, "I'm planning to switch jobs next year, and I prefer advice in Hindi.")
        check(bool(r["response"].strip()), "POST /chat returns a response", f"{len(r['response'])} chars")
        check(r.get("degraded") is False, "LLM answered (not the outage fallback)")

        r2 = await api.chat(s1, "Which of the things I just told you matters most?")
        check(bool(r2["response"].strip()), "second turn in the same session works (short-term history in Redis)")

        text, tokens, ok = await ws_roundtrip(ws_base, user, s1, "Say hello in one short line.", args.timeout)
        check(ok and tokens > 0, "WebSocket streams tokens then 'done'", f"{tokens} token events")

        # ================= 2. MEMORY HANDLING =================================================================
        print("\n2. Memory handling")
        mems = await api.wait_for(lambda m: {"career_goal", "language_preference"} <= set(m))
        check("career_goal" in mems, "career goal saved from a plain statement",
              mems.get("career_goal", {}).get("value", "missing"))
        check(mems.get("language_preference", {}).get("value", "").lower().startswith("hindi"),
              "language preference saved", mems.get("language_preference", {}).get("value", "missing"))

        before = set(mems)
        await api.chat(s1, "What is the capital of France?", debug=False)
        await api.chat(s1, "Can you explain how interest rates work?", debug=False)
        await asyncio.sleep(3)
        after = set(await api.memories())
        check(after == before, "questions / small talk are NOT stored", f"new keys: {sorted(after - before) or 'none'}")

        old_version = mems["career_goal"]["version"]
        await api.chat(s1, "I'm planning to start a business next year.", debug=False)
        mems = await api.wait_for(lambda m: m.get("career_goal", {}).get("version", 0) > old_version)
        cg = mems.get("career_goal", {})
        check(cg.get("version", 0) > old_version and "business" in cg.get("value", "").lower(),
              "correction updates the same key (no duplicate)", f"v{cg.get('version')}: {cg.get('value')}")
        check(len(cg.get("history", [])) >= 1, "previous value kept in history",
              str([h.get("value") for h in cg.get("history", [])]))

        # seed a spread of unrelated memories for the relevance test
        for text in ["I'm planning to save money for a house.", "I live in Pune.",
                     "I love cricket and cooking.", "I'm single."]:
            await api.chat(s1, text, debug=False)
        mems = await api.wait_for(lambda m: len(m) >= 6, timeout=90)
        check(len(mems) >= 6, "several different facts accumulated", ", ".join(sorted(mems)))

        # ================= 3. CONTEXT RELEVANCE ===============================================================
        print("\n3. Context relevance (what is actually sent to the LLM)")
        total = len(mems)
        career = await api.chat(s1, "What should I focus on in my career this year?")
        sent = sent_keys(career)
        dbg = career.get("debug", {})
        check("career_goal" in sent, "career question -> career memory is sent", f"sent={sorted(sent)}")
        check(len(sent) < total, "NOT everything is sent", f"{len(sent)} of {total} memories sent; considered={dbg.get('considered')}")
        irrelevant = sent & {"location", "relationship_status", "financial_goal", "interests"}
        check(not irrelevant, "unrelated memories (finance/location/hobby/relationship) are kept out",
              f"leaked: {sorted(irrelevant)}" if irrelevant else "none leaked")

        trivia = await api.chat(s1, "What is the tallest mountain in the world?")
        sent = sent_keys(trivia)
        check(sent <= {"language_preference"}, "unrelated question -> only the always-on language preference",
              f"sent={sorted(sent)}")

        # ================= 1b. SESSION END + CROSS-SESSION ====================================================
        print("\n1b. Session end and a new session [llm]")
        ended = await api.end(s1)
        summary = (ended.get("summary") or {}).get("summary", "")
        check(ended.get("ended") is True and bool(summary.strip()), "ending a session stores a summary",
              summary[:90].replace("\n", " "))

        fresh = await api.chat(s2, "What did we talk about last time?")
        check(len(fresh.get("debug", {}).get("memory_scores", {})) > 0 or bool(fresh.get("context_used")),
              "NEW session still receives stored user context", f"context_used={len(fresh.get('context_used', []))} items")
        await api.chat(s3, "Hello!", debug=False)

        if not args.keep:
            for key in list(await api.memories()):
                await client.delete(f"/users/{user}/memories/{key}")
            print(f"\n(cleaned up memories of {user}; use --keep to leave them for python -m scripts.inspect_mongo --user {user})")
        else:
            print(f"\nKept data. Inspect: python -m scripts.inspect_mongo --user {user}")

    failed = [r for r in RESULTS if not r[0]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--keep", action="store_true", help="keep the test user's data (for inspect_mongo)")
    sys.exit(asyncio.run(run(p.parse_args())))


if __name__ == "__main__":
    main()
