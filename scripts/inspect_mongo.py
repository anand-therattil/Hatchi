"""Read-only viewer for what Hatchi stored in MongoDB, per user (plain script).

    python -m scripts.inspect_mongo --list                       # all users with memory / summary counts
    python -m scripts.inspect_mongo --list --prefix load-        # only users whose id starts with 'load-'
    python -m scripts.inspect_mongo --user alice                 # alice's memories + session summaries
    python -m scripts.inspect_mongo --user alice --history       # also show previous values of each memory
    python -m scripts.inspect_mongo --user alice --json          # raw documents as JSON
    python -m scripts.inspect_mongo --user alice --mongo-url mongodb://host:27017 --db hatchi

Defaults come from .env (MONGO_URL / MONGO_DB) like the app. Nothing is ever modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime

from pymongo import MongoClient


def fmt_ts(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, datetime) else str(value or "-")


def list_users(db, prefix: str) -> int:
    regex = {"$regex": "^" + prefix.replace("\\", "\\\\").replace(".", r"\.")} if prefix else {"$exists": True}
    counts: dict[str, list] = defaultdict(lambda: [0, 0, None])
    for doc in db.user_memories.find({"user_id": regex}, {"user_id": 1, "updated_at": 1}):
        row = counts[doc["user_id"]]
        row[0] += 1
        row[2] = max(filter(None, [row[2], doc.get("updated_at")]), default=None)
    for doc in db.session_summaries.find({"user_id": regex}, {"user_id": 1}):
        counts[doc["user_id"]][1] += 1
    if not counts:
        print("No users found.")
        return 0
    print(f"{'user_id':<40} {'memories':>8} {'sessions':>8}  last memory update")
    for user_id, (memories, sessions, last) in sorted(counts.items()):
        print(f"{user_id:<40} {memories:>8} {sessions:>8}  {fmt_ts(last)}")
    print(f"\n{len(counts)} user(s)")
    return 0


def show_user(db, user_id: str, history: bool, as_json: bool) -> int:
    memories = list(db.user_memories.find({"user_id": user_id}).sort([("category", 1), ("key", 1)]))
    summaries = list(db.session_summaries.find({"user_id": user_id}).sort("created_at", 1))
    if as_json:
        print(json.dumps({"user_id": user_id, "memories": memories, "session_summaries": summaries},
                         indent=2, default=str, ensure_ascii=False))
        return 0 if (memories or summaries) else 1
    if not memories and not summaries:
        print(f"No data stored for user '{user_id}'. (Try --list to see existing users.)")
        return 1

    print(f"USER {user_id}")
    print(f"\nMEMORIES ({len(memories)})")
    for m in memories:
        indexed = "" if m.get("indexed", True) else "  [not yet in vector index]"
        print(f"  {m['key']:<20} = {m['value']}")
        print(f"  {'':<20}   {m.get('category', '-')}/{m.get('kind', '-')}  conf {m.get('confidence', 0):.2f}  "
              f"v{m.get('version', 1)}  updated {fmt_ts(m.get('updated_at'))}  session {m.get('source_session') or '-'}{indexed}")
        if history:
            for h in m.get("history", []):
                print(f"  {'':<20}   previously: {h.get('value')!r}  (v{h.get('version', '?')}, {h.get('replaced_at') or h.get('ts') or ''})")
    if not memories:
        print("  (none)")

    print(f"\nSESSION SUMMARIES ({len(summaries)})")
    for s in summaries:
        print(f"  session {s.get('session_id')}  ({s.get('message_count', '?')} messages, {fmt_ts(s.get('created_at'))})")
        print(f"    topics : {', '.join(s.get('topics', [])) or '-'}")
        print(f"    summary: {s.get('summary', '').strip()}")
    if not summaries:
        print("  (none - a summary is written when a session ends)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", help="user_id to show")
    parser.add_argument("--list", action="store_true", help="list all users")
    parser.add_argument("--prefix", default="", help="with --list: only user ids starting with this")
    parser.add_argument("--history", action="store_true", help="show previous values of updated memories")
    parser.add_argument("--json", action="store_true", help="print raw documents as JSON")
    parser.add_argument("--mongo-url", help="override MONGO_URL")
    parser.add_argument("--db", help="override MONGO_DB")
    args = parser.parse_args()
    if not args.user and not args.list:
        parser.error("give --user <id> or --list")

    url, name = args.mongo_url, args.db
    if not (url and name):
        try:
            from app.config import get_settings

            settings = get_settings()
            url, name = url or settings.mongo_url, name or settings.mongo_db
        except Exception:  # noqa: BLE001 - run standalone without the app installed
            url, name = url or "mongodb://localhost:27017", name or "hatchi"

    client = MongoClient(url, serverSelectionTimeoutMS=4000, tz_aware=True)
    try:
        client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot reach MongoDB at {url}: {exc.__class__.__name__}\nIs `docker compose up -d mongo` running?", file=sys.stderr)
        sys.exit(2)
    db = client[name]
    sys.exit(list_users(db, args.prefix) if args.list else show_user(db, args.user, args.history, args.json))


if __name__ == "__main__":
    main()
