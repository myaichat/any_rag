import os, sys, time
import asyncio
import groq
from huggingface_hub import AsyncInferenceClient

from pprint import pprint as pp
e=sys.exit
import include.config.init_config as init_config 

apc = init_config.apc


class AsyncClient(AsyncInferenceClient):
    def __init__(self, api_key):
        
        super().__init__( token=api_key)    

    async def chat(self, model, messages, temperature, max_tokens):
        response = await self.chat_completion(messages,model=model, temperature=temperature, max_tokens=max_tokens)
        
        return response
            
    async def close(self):

        pass

async def call_llm(cot_model, messages: list,
                   temperature: float = 0.7,
                   max_tokens: int = 8000) -> str:

    api=cot_model['api']
    model=cot_model['name']    
    print(f'\t:hugging_face: call_llm:', model)
    apc.prompt_log['rag_models'][model]={}
    apc.prompt_log['rag_models'][model]



    client=apc.get_client(api)
    response = await client.chat(
        model=model,
        messages=messages, 
        temperature=temperature,
        max_tokens=max_tokens,
    )
    #print(f"\t\t{layer}:     openai: Sleep: {sleep_time}: Model: ", model)
    
    
    
    content = response.choices[0].message.content
    
    assert content
    

    
    print(f'\t hugging_face'.rjust(10,' '),f':{model}:Content:', len(content))
    apc.prompt_log['rag_models'][model]['response']=content
    return content




