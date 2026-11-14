"""Terminal chat client for the WebSocket API.

    python cli.py --user user-123 [--session session-456] [--url ws://localhost:8000]

Type a message and press Enter. Commands: /end (finish the session and print its summary), /quit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from urllib.parse import urlencode

import websockets


async def run(url: str, user: str, session: str | None) -> None:
    query = {"user_id": user, **({"session_id": session} if session else {})}
    async with websockets.connect(f"{url}/ws/chat?{urlencode(query)}") as ws:
        hello = json.loads(await ws.recv())
        print(f"[connected] session_id={hello['session_id']}  (/end to finish, /quit to leave)")

        while True:
            try:
                text = (await asyncio.to_thread(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                continue
            if text == "/quit":
                return
            await ws.send(json.dumps({"type": "end"} if text == "/end" else {"type": "message", "text": text}))

            print("naksh> ", end="", flush=True)
            while True:
                event = json.loads(await ws.recv())
                kind = event["type"]
                if kind == "token":
                    print(event["text"], end="", flush=True)
                elif kind == "done":
                    print(f"\n  [context: {', '.join(event['context_used']) or 'none'}]")
                    break
                elif kind == "ended":
                    summary = event["summary"]
                    print(f"\n  [session ended] {summary['summary'] if summary else '(empty session)'}")
                    return
                elif kind == "error":
                    print(f"\n  [error] {event['detail']}")
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Hatchi terminal chat")
    parser.add_argument("--user", required=True)
    parser.add_argument("--session")
    parser.add_argument("--url", default="ws://localhost:8000")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url, args.user, args.session))
    except (ConnectionRefusedError, OSError) as exc:
        sys.exit(f"Could not connect to {args.url}: {exc}")


if __name__ == "__main__":
    main()
