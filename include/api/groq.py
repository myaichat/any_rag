import os, time
import asyncio
import groq
from groq import AsyncGroq, Groq

from pprint import pprint as pp

import include.config.init_config as init_config 

apc = init_config.apc


class AsyncClient(AsyncGroq):   
    def __init__(self, api_key):
        super().__init__(api_key=api_key)





async def call_llm(cot_model, messages: list,
                   temperature: float = 0.7,
                   max_tokens: int = 8000) -> str:
    """Call the Groq API."""
    api=cot_model['api']
    model=cot_model['name']
    client=apc.get_client(api)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content



