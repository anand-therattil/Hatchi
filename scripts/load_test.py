"""Concurrency / load test for a RUNNING Hatchi server (plain script - no pytest).

    python -m scripts.load_test                                  # 10 users in parallel, 3 turns each, over HTTP
    python -m scripts.load_test --users 50 --messages 4 --concurrency 10     # at most 10 requests in flight
    python -m scripts.load_test --mode ws --users 20             # WebSocket; also reports time-to-first-token
    python -m scripts.load_test --same-user --users 12           # race test: ONE user, 12 parallel messages
    python -m scripts.load_test --url http://gpu-box:8000 --users 100 --concurrency 32 --cleanup

Each simulated user runs a short script (a statement with two facts, a career question, a hobby statement, a
follow-up, ...) sequentially, while users run in parallel. The report shows throughput, latency percentiles, errors
and "degraded" (LLM-outage fallback) replies, then CHECKS THE DATA: that every user's memories were really written
(background extraction under load) - or, with --same-user, that no memory update was lost to a race.

Exit code is 1 if any request failed or any memory check failed.

Side effects: memories are written under ids "<prefix><run>-<n>" (see --user-prefix). --cleanup deletes them. Their
Redis sessions end by themselves later (idle sweeper / WebSocket grace period), which writes a summary per session;
list them with  python -m scripts.inspect_mongo --list --prefix load-
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx

JOB_MSG = "I'm planning to switch jobs next year, and I prefer advice in Hindi."
HOBBY_MSG = "I love cricket and cooking."
SCRIPT = [
    JOB_MSG,
    "What should I focus on for my career?",
    HOBBY_MSG,
    "Why do you say that?",
    "Will my finances improve this year?",
]
EXPECTED_KEYS = {JOB_MSG: {"career_goal", "language_preference"}, HOBBY_MSG: {"interests"}}
HOBBIES = ["cricket", "cooking", "chess", "painting", "hiking", "reading", "singing", "cycling", "gardening",
           "yoga", "football", "photography", "dancing", "fishing", "knitting", "baking"]


# ---- statistics -----------------------------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))]


def describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return exc.__class__.__name__ + (f": {str(exc)[:60]}" if str(exc) else "")


@dataclass
class Stats:
    latencies: list[float] = field(default_factory=list)
    first_token: list[float] = field(default_factory=list)
    errors: Counter = field(default_factory=Counter)
    degraded: int = 0

    def ok(self, seconds: float, degraded: bool = False, first_token: float | None = None) -> None:
        self.latencies.append(seconds)
        self.degraded += bool(degraded)
        if first_token is not None:
            self.first_token.append(first_token)

    def fail(self, reason: str) -> None:
        self.errors[reason] += 1

    @property
    def total(self) -> int:
        return len(self.latencies) + sum(self.errors.values())


def line(label: str, values: list[float]) -> str:
    if not values:
        return f"  {label:<22} (no data)"
    return (f"  {label:<22} min {min(values):6.2f}  mean {sum(values) / len(values):6.2f}  p50 {percentile(values, 50):6.2f}"
            f"  p95 {percentile(values, 95):6.2f}  p99 {percentile(values, 99):6.2f}  max {max(values):6.2f}")


# ---- traffic generators ------------------------------------------------------------------------------------------
Plan = tuple[str, str, list[str]]  # (user_id, session_id, messages)


async def run_http(client: httpx.AsyncClient, sem: asyncio.Semaphore, plan: Plan, stats: Stats) -> None:
    user_id, session_id, messages = plan
    for text in messages:
        async with sem:
            started = time.perf_counter()
            try:
                response = await client.post("/chat", json={"user_id": user_id, "session_id": session_id, "message": text})
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001 - every failure mode is a data point
                stats.fail(describe(exc))
                continue
        stats.ok(time.perf_counter() - started, degraded=data.get("degraded", False))


async def run_ws(ws_base: str, timeout: float, sem: asyncio.Semaphore, plan: Plan, stats: Stats) -> None:
    import websockets

    user_id, session_id, messages = plan
    uri = f"{ws_base}/ws/chat?{urlencode({'user_id': user_id, 'session_id': session_id})}"
    try:
        async with websockets.connect(uri, open_timeout=15, max_size=None) as ws:
            await asyncio.wait_for(ws.recv(), timeout)  # {"type": "session"}
            for text in messages:
                async with sem:
                    started, first = time.perf_counter(), None
                    try:
                        await ws.send(json.dumps({"type": "message", "text": text}))
                        while True:
                            event = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                            if event["type"] == "token" and first is None:
                                first = time.perf_counter() - started
                            elif event["type"] == "done":
                                stats.ok(time.perf_counter() - started, event.get("degraded", False), first)
                                break
                            elif event["type"] == "error":
                                raise RuntimeError(event.get("detail", "error event"))
                    except Exception as exc:  # noqa: BLE001
                        stats.fail(describe(exc))
    except Exception as exc:  # noqa: BLE001
        stats.fail("connect: " + describe(exc))


# ---- post-run data checks ----------------------------------------------------------------------------------------
async def wait_until(client: httpx.AsyncClient, user_id: str, done, timeout: float) -> tuple[bool, float, dict]:
    """Poll the user's memories until done(memories_by_key) is true. Returns (ok, seconds_waited, last_memories)."""
    started, memories = time.perf_counter(), {}
    while True:
        try:
            response = await client.get(f"/users/{user_id}/memories")
            memories = {m["key"]: m for m in response.json()} if response.status_code == 200 else {}
        except Exception:  # noqa: BLE001
            memories = {}
        if done(memories):
            return True, time.perf_counter() - started, memories
        if time.perf_counter() - started > timeout:
            return False, time.perf_counter() - started, memories
        await asyncio.sleep(1.0)


async def cleanup(client: httpx.AsyncClient, user_ids: list[str]) -> int:
    removed = 0
    for user_id in user_ids:
        try:
            for memory in (await client.get(f"/users/{user_id}/memories")).json():
                removed += (await client.delete(f"/users/{user_id}/memories/{memory['key']}")).status_code == 200
        except Exception:  # noqa: BLE001
            pass
    return removed


# ---- orchestration ----------------------------------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    run_id = uuid.uuid4().hex[:6]
    concurrency = args.concurrency or args.users
    limits = httpx.Limits(max_connections=max(100, concurrency * 2))

    async with httpx.AsyncClient(base_url=base, timeout=args.timeout, limits=limits) as client:
        try:
            (await client.get("/health")).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"Server not reachable at {base}: {describe(exc)}\nStart it with: uvicorn app.main:app", file=sys.stderr)
            return 2

        # ---- build the plan --------------------------------------------------------------------------------------
        if args.same_user:
            user = f"{args.user_prefix}{run_id}-same"
            hobbies = [h if i < len(HOBBIES) else f"{HOBBIES[i % len(HOBBIES)]}{i}" for i, h in
                       enumerate((HOBBIES * (args.users // len(HOBBIES) + 1))[: args.users])]
            hobbies = [f"{h}{i // len(HOBBIES) or ''}" for i, h in enumerate(hobbies)]
            plans: list[Plan] = [(user, f"s-{user}", [f"I love {h}."]) for h in hobbies]
            user_ids = [user]
        else:
            user_ids = [f"{args.user_prefix}{run_id}-{i}" for i in range(args.users)]
            messages = [SCRIPT[k % len(SCRIPT)] for k in range(args.messages)]
            plans = [(u, f"s-{u}", messages) for u in user_ids]

        sem = asyncio.Semaphore(concurrency)
        stats = Stats()
        total = sum(len(p[2]) for p in plans)
        print(f"Hatchi load test  run={run_id}  mode={args.mode}  target={base}")
        print(f"  {len(user_ids)} user(s), {total} request(s), max {concurrency} in flight"
              + ("   [same-user race test]" if args.same_user else ""))

        started = time.perf_counter()
        if args.mode == "ws":
            await asyncio.gather(*(run_ws(ws_base, args.timeout, sem, p, stats) for p in plans))
        else:
            await asyncio.gather(*(run_http(client, sem, p, stats) for p in plans))
        wall = time.perf_counter() - started

        # ---- traffic report ------------------------------------------------------------------------------------------
        ok = len(stats.latencies)
        print(f"\nRequests : {stats.total} sent, {ok} ok, {sum(stats.errors.values())} failed, {stats.degraded} degraded"
              " (LLM fallback)")
        for reason, count in stats.errors.most_common():
            print(f"             x{count}  {reason}")
        print(f"Wall time: {wall:.1f}s   throughput: {ok / wall:.2f} req/s")
        print("Latency (seconds):")
        print(line("full response", stats.latencies))
        if stats.first_token:
            print(line("time to first token", stats.first_token))

        # ---- data checks: did the background work keep up and stay correct? ----------------------------------------------
        failures = 0
        if not args.no_verify:
            print(f"\nChecking stored memories (waiting up to {args.verify_wait:.0f}s for background extraction) ...")
            if args.same_user:
                expected = {h.lower() for h in (plans_item[2][0].removeprefix("I love ").rstrip(".") for plans_item in plans)}

                def have_all(mems: dict, want=expected) -> bool:
                    items = {i.strip().lower() for i in mems.get("interests", {}).get("value", "").split(",")}
                    return want <= items

                good, waited, mems = await wait_until(client, user_ids[0], have_all, args.verify_wait)
                got = {i.strip().lower() for i in mems.get("interests", {}).get("value", "").split(",") if i.strip()}
                missing = sorted(expected - got)
                print(f"  {len(expected) - len(missing)}/{len(expected)} parallel updates present in 'interests'"
                      f" (settled in {waited:.1f}s)" + (f"; LOST: {', '.join(missing)}" if missing else ""))
                if not good:
                    print("  FAIL: updates were lost -> the per-user write lock is not holding")
                    failures += 1
            else:
                needed = set().union(*(EXPECTED_KEYS.get(m, set()) for m in plans[0][2]))
                results = await asyncio.gather(*(
                    wait_until(client, u, lambda mems, n=needed: n <= set(mems), args.verify_wait) for u in user_ids))
                good = [r for r in results if r[0]]
                waits = [r[1] for r in good]
                print(f"  expected keys per user: {', '.join(sorted(needed)) or '(none)'}")
                print(f"  {len(good)}/{len(user_ids)} users fully written"
                      + (f"   (settled after traffic ended: p50 {percentile(waits, 50):.1f}s, max {max(waits):.1f}s)" if waits else ""))
                bad = [(u, sorted(needed - set(r[2]))) for u, r in zip(user_ids, results) if not r[0]]
                for u, missing in bad[:5]:
                    print(f"    {u}: missing {', '.join(missing)}")
                failures += bool(bad)

        if args.cleanup:
            print(f"\nCleanup: removed {await cleanup(client, user_ids)} memories")

    verdict_ok = not stats.errors and not failures
    print("\nRESULT:", "PASS" if verdict_ok else "FAIL")
    return 0 if verdict_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the running server")
    parser.add_argument("--mode", choices=["http", "ws"], default="http", help="POST /chat or WebSocket /ws/chat")
    parser.add_argument("--users", type=int, default=10, help="simulated users (or parallel messages with --same-user)")
    parser.add_argument("--messages", type=int, default=3, help="turns per user (scripted conversation, cycles after 5)")
    parser.add_argument("--concurrency", type=int, default=0, help="max requests in flight (default: = users)")
    parser.add_argument("--same-user", action="store_true", help="all requests share ONE user: tests lost memory updates")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout in seconds")
    parser.add_argument("--verify-wait", type=float, default=90.0, help="seconds to wait for background memory writes")
    parser.add_argument("--no-verify", action="store_true", help="skip the stored-memory checks")
    parser.add_argument("--cleanup", action="store_true", help="delete the test users' memories afterwards")
    parser.add_argument("--user-prefix", default="load-", help="prefix for generated user ids")
    args = parser.parse_args()
    if args.users < 1 or args.messages < 1:
        parser.error("--users and --messages must be >= 1")
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
