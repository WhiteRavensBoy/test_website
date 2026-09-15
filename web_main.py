import asyncio
import functools
import logging
import os
import time
from typing import Annotated

import ollama
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from agent import ask_agent
from cache_utils import cache_response, get_cached_response

#from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def environment_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(int(value), minimum)
    except ValueError:
        logger.warning("Invalid integer for %s; using default", name)
        return default


OLLAMA_HOST = os.getenv("OLLAMA_HOST")
if not OLLAMA_HOST:
    if os.path.exists("/.dockerenv"):
        OLLAMA_HOST = "http://host.docker.internal:11434"
    else:
        OLLAMA_HOST = "http://127.0.0.1:11434"

client = ollama.Client(host=OLLAMA_HOST)

concurrent_llm_limit = environment_int("LLM_CONCURRENCY", 50)
llm_timeout_seconds = environment_int("LLM_TIMEOUT_SECONDS", 120)
agent_timeout_seconds = environment_int("AGENT_TIMEOUT_SECONDS", 120)
max_query_length = environment_int("MAX_QUERY_LENGTH", 2_000)
max_response_length = environment_int("MAX_RESPONSE_LENGTH", 20_000)
lock = asyncio.Semaphore(concurrent_llm_limit)
inflight_requests = {}
inflight_lock = asyncio.Lock()

app = FastAPI()

model = os.getenv("OLLAMA_MODEL", "llava:7b")
count = 0

#llm = LLM(model=model, trust_remote_code= True)
#sampling_params = SamplingParams(temparature=0.7, top_p=0.95, max_tokens=100)

def calculate_time(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"function completed in :{end_time-start_time} seconds.")
        return result
    return wrapper

async def chat_ai(query: str):
    async with lock:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.chat,
                model=model,
                messages=[{"role": "user", "content": query}],
            ),
            timeout=llm_timeout_seconds,
        )
        content = response.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The model returned an empty response")
        return content[:max_response_length]


async def get_cached_or_generate(query: str) -> str:
    cached_response = get_cached_response(query, model=model)
    if cached_response:
        return cached_response["response"]

    async with inflight_lock:
        cached_response = get_cached_response(query, model=model)
        if cached_response:
            return cached_response["response"]

        if query not in inflight_requests:
            inflight_requests[query] = asyncio.create_task(chat_ai(query=query))

        task = inflight_requests[query]

    try:
        response = await task
    except Exception:
        async with inflight_lock:
            if inflight_requests.get(query) is task:
                inflight_requests.pop(query, None)
        raise

    async with inflight_lock:
        if inflight_requests.get(query) is task:
            inflight_requests.pop(query, None)

    cache_response(query, model, response)
    return response


@app.get("/")
async def greeting():
    return {"message": "Hi, arun...."}


@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(request: Request, exc: asyncio.TimeoutError):
    logger.warning("Request timed out: %s", request.url.path)
    return JSONResponse(status_code=504, content={"detail": "The AI service timed out"})


def validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if len(normalized) > max_query_length:
        raise HTTPException(
            status_code=413,
            detail=f"Query exceeds the {max_query_length} character limit",
        )
    if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
        raise HTTPException(status_code=400, detail="Query contains unsupported control characters")
    return normalized


def calculate_user_count(func):
    async def wrapper(query: str):
        global count
        count += 1
        print(count)
        return await func(query)

    return wrapper


@app.get("/ask/{query}")
@calculate_user_count
@calculate_time
async def ask_ai(query: Annotated[str, Path(min_length=1)]):
    query = validate_query(query)
    try:
        response = await get_cached_or_generate(query)
    except asyncio.TimeoutError:
        raise
    except Exception:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=503, detail="The AI service is unavailable") from None
    return "User : " + query + "\n" + response


@app.get("/ask_agent/{query}")
@calculate_time
async def ask_ai_agent(query: Annotated[str, Path(min_length=1)]):
    query = validate_query(query)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(ask_agent, query=query),
            timeout=agent_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise
    except Exception:
        logger.exception("Agent request failed")
        raise HTTPException(status_code=503, detail="The agent service is unavailable") from None


"""
@app.get("/ask_vllm/{query}")
@calculate_user_count
@calculate_time
async def ask_ai_with_vllm(query: str):
    response = generate_with_vllm(query=query)
    print(response)
    return response
    
async def generate_with_vllm(prompt: str):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: llm.generate(prompt, sampling_params=sampling_params)
    )
    """