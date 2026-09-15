import redis
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=0,
    decode_responses=True,
    socket_connect_timeout=0.2,
    socket_timeout=0.5,
    health_check_interval=30,
)


def create_cache_key(prompt: str, model: str) -> str:
    raw_key = f"{model}:{prompt}"
    hash_key = hashlib.sha256(raw_key.encode()).hexdigest()
    return f"llm:response:{hash_key}"


def cache_response(prompt: str, model: str, response: str):
    key = create_cache_key(prompt=prompt, model=model)

    value = {
        "model": model,
        "response": response,
    }

    try:
        redis_client.set(key, json.dumps(value), ex=60)
    except redis.exceptions.RedisError:
        logger.warning("Redis cache write failed; continuing without cache")


def get_cached_response(prompt: str, model: str):
    key = create_cache_key(prompt=prompt, model=model)

    try:
        cached = redis_client.get(key)
    except redis.exceptions.RedisError:
        logger.warning("Redis cache lookup failed; continuing without cache")
        return None

    if cached:
        try:
            logger.warning("cache hit")
            return json.loads(cached)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed Redis cache entry")
            return None

    return None

