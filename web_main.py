from fastapi import FastAPI
import asyncio
import os
import ollama
import sys
from agent import ask_agent
import functools
import time

#from vllm import LLM, SamplingParams

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
if OLLAMA_HOST is None:
    if os.path.exists("/.dockerenv"):
        OLLAMA_HOST = "http://host.docker.internal:11434"
    else:
        OLLAMA_HOST = "http://127.0.0.1:11434"

client = ollama.Client(host=OLLAMA_HOST)

concurrent_llm_limit = 50
lock = asyncio.Semaphore(concurrent_llm_limit)

app = FastAPI()

model = "llava:7b"
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
        response = await asyncio.to_thread(
            client.chat,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"{query}",
                }
            ],
        )
        return response["message"]["content"]

async def generate_with_vllm(prompt: str):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: llm.generate(prompt, sampling_params=sampling_params)
    )


@app.get("/")
async def greeting():
    return {"message": "Hi, arun...."}


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
async def ask_ai(query: str):
    ans = "User : " + str(query)
    response = await chat_ai(query=query)
    ans += "\n" + response
    print(count)
    print(ans)
    return ans

@app.get("/ask_agent/{query}")
@calculate_time
async def ask_ai_agent(query: str):
    response = ask_agent(query=query)
    print(response)
    return response
"""
@app.get("/ask_vllm/{query}")
@calculate_user_count
@calculate_time
async def ask_ai_with_vllm(query: str):
    response = generate_with_vllm(query=query)
    print(response)
    return response"""