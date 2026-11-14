"""Redis-backed short-term session state.

Keys (all per user+session; ids are validated to never contain ':'):
  hatchi:session:{user}:{session}:messages   LIST of Message JSON, oldest first
  hatchi:session:{user}:{session}:summary    STRING rolling summary of folded-away messages
  hatchi:session:{user}:{session}:meta       HASH  started_at, last_active, message_count
  hatchi:sessions:active                     ZSET  member=[user, session] score=last_active
"""
from __future__ import annotations

import json
import time

from app.models import Message

ACTIVE_ZSET = "hatchi:sessions:active"


class RedisSessionStore:
    def __init__(self, client, ttl_seconds: int = 86400):
        # `client` is a redis.asyncio.Redis created with decode_responses=True
        self._r = client
        self._ttl = ttl_seconds

    @staticmethod
    def _key(user_id: str, session_id: str, part: str) -> str:
        return f"hatchi:session:{user_id}:{session_id}:{part}"

    @staticmethod
    def _member(user_id: str, session_id: str) -> str:
        return json.dumps([user_id, session_id])

    async def append(self, user_id: str, session_id: str, messages: list[Message]) -> int:
        msgs_key = self._key(user_id, session_id, "messages")
        meta_key = self._key(user_id, session_id, "meta")
        now = time.time()
        async with self._r.pipeline(transaction=True) as pipe:
            for m in messages:
                pipe.rpush(msgs_key, m.model_dump_json())
            pipe.expire(msgs_key, self._ttl)
            pipe.hsetnx(meta_key, "started_at", now)
            pipe.hset(meta_key, "last_active", now)
            pipe.hincrby(meta_key, "message_count", len(messages))
            pipe.expire(meta_key, self._ttl)
            pipe.zadd(ACTIVE_ZSET, {self._member(user_id, session_id): now})
            pipe.llen(msgs_key)
            results = await pipe.execute()
        return int(results[-1])

    async def recent(self, user_id: str, session_id: str) -> list[Message]:
        raw = await self._r.lrange(self._key(user_id, session_id, "messages"), 0, -1)
        return [Message.model_validate_json(item) for item in raw]

    async def drop_oldest(self, user_id: str, session_id: str, count: int) -> None:
        if count > 0:
            await self._r.ltrim(self._key(user_id, session_id, "messages"), count, -1)

    async def get_summary(self, user_id: str, session_id: str) -> str:
        return await self._r.get(self._key(user_id, session_id, "summary")) or ""

    async def set_summary(self, user_id: str, session_id: str, summary: str) -> None:
        await self._r.set(self._key(user_id, session_id, "summary"), summary, ex=self._ttl)

    async def get_meta(self, user_id: str, session_id: str) -> dict:
        raw = await self._r.hgetall(self._key(user_id, session_id, "meta"))
        return {
            "started_at": float(raw.get("started_at", 0) or 0),
            "last_active": float(raw.get("last_active", 0) or 0),
            "message_count": int(raw.get("message_count", 0) or 0),
        }

    async def idle_sessions(self, last_active_before: float) -> list[tuple[str, str]]:
        members = await self._r.zrangebyscore(ACTIVE_ZSET, 0, last_active_before)
        pairs = []
        for member in members:
            user_id, session_id = json.loads(member)
            pairs.append((user_id, session_id))
        return pairs

    async def delete(self, user_id: str, session_id: str) -> None:
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.delete(
                self._key(user_id, session_id, "messages"),
                self._key(user_id, session_id, "summary"),
                self._key(user_id, session_id, "meta"),
            )
            pipe.zrem(ACTIVE_ZSET, self._member(user_id, session_id))
            await pipe.execute()
