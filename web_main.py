from fastapi import FastAPI
import asyncio
import os
import ollama
import sys


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
async def ask_ai(query: str):
    ans = "User : " + str(query)
    response = await chat_ai(query=query)
    ans += "\n" + response
    print(count)
    print(ans)
    return ans