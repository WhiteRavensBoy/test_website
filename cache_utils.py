import redis
import hashlib
import json

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
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
        redis_client.set(key, json.dumps(value), ex=30)
    except redis.exceptions.ConnectionError:
        print("Redis is not available. Cache write skipped.")


def get_cached_response(prompt: str, model: str):
    key = create_cache_key(prompt=prompt, model=model)

    try:
        cached = redis_client.get(key)
    except redis.exceptions.ConnectionError:
        print("Redis is not available. Cache lookup skipped.")
        return None

    if cached:
        print("CACHE HIT..")
        return json.loads(cached)
    print("CACHE MISS...")

    return None

