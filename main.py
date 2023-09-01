import copy
import os
import gc
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from starlette.routing import Route
import time
import numpy as np

from model import InputData
from controller import get_model, handle_llama

app = FastAPI()

@app.post("/api/generate")
async def handle(data: InputData):
    model = await get_model()
    response = ""
    start_time = time.perf_counter()
    total_tokens = 0
    async for token in handle_llama(data.system_prompt, data.message, data.conversation, model):
        response += token
        total_tokens += 1
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print(f"## Time taken (seconds): {time_taken} ##")
    print(f"## Tokens generated: {total_tokens} ##")
    print(f"## Tokens per second: {total_tokens / time_taken} ##")

    # TODO: return time_taken and total_tokens

    return StreamingResponse(iter([response]), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9998)