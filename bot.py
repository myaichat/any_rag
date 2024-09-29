import os, sys, time
from typing import List, Tuple
from os.path import join
import asyncio
import yaml 
import click
from pprint import pprint as pp
from os.path import join
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from typing import List, Tuple

#from  include.api.deepinfra import AsyncClient, get_final_stream
import include.config.init_config as init_config 

init_config.init(**{})
apc = init_config.apc
apc.models={}


#import include.api.deepinfra as deepinfra
import include.api.groq as groq
#import include.api.together as together
#import include.api.openai as openai 
#import include.api.mistral as mistral
#import include.api.nvidia as nvidia
#import include.api.deepseek as deepseek
import include.api.hugging_face as hugging_face
#import include.api.anthropic as anthropic
#import include.api.gemini as gemini
#import include.api.cohere as cohere
#import include.api.palm2 as palm2
from include.common import get_aggregator 
e=sys.exit

apc.clients={}

DOCS = ['text.txt']
DOCS = ['code.txt']
DOCS = ['storm_text.txt', 'medium_text.txt']
DOCS = ['code_2.txt']
DOCS = ['lesson_2.txt']
DOCS = [join('lessons','L3_extended.txt')]
console = Console()
       
async def close_clients():

    for client in apc.clients.values():
        await client.close()

def save_models(rag_models):   
    apc.rag_models=rag_models 
    for _, model in rag_models.items():
        if type(model) is not dict:
            assert type(model) is list
            for m in model:
                api=m['api']
                apc.apis[api]=globals()[api]
        else:
            api=model['api']
            apc.apis[api]=globals()[api]

def save_prompt(rag_prompt):   
    apc.rag_prompt=rag_prompt 


def get_prompt(model_prompt, user_prompt):
    parsed_string = model_prompt.format(user_prompt=user_prompt)
   
    return parsed_string




async def generalist_llm(query: str) -> Tuple[bool, str]:
    """Call the generalist LLM to determine if the query is knowledge-intensive."""
    rag_model=apc.rag_models['generalist']
    generalist_model    =rag_model['name']       
    generalist_system_prompt=apc.rag_prompt['generalist']  
    api=rag_model['api']

    messages = [{
        "role": "system",
        "content": generalist_system_prompt
    }, {
        "role": "user",
        "content": query
    }]
    
    response = await apc.apis[api].call_llm(rag_model,
                              messages,
                              temperature=0.5,
                              max_tokens=50)
   
    console.print(Panel(response, title="generalist_llm", title_align="left", border_style="Yellow", style="white"))
    is_complex = response.lower().strip().startswith('yes')
    return is_complex, f"Generalist ({generalist_model}) decision: {'Knowledge-intensive' if is_complex else 'Simple'}"


async def drafter_llm(mid, did, rag_model, query: str, document: str) -> Tuple[str, str, str]:
    """Call the specialist LLM to generate a draft and rationale."""

    
    specialist_model    =rag_model['name']       
    specialist_system_prompt=apc.rag_prompt['specialist']  
    api=rag_model['api']
    
    messages = [{
        "role": "system",
        "content": specialist_system_prompt
    }, {
        "role": "user",
        "content": f"Query: {query}\n\nDocument:\n{document}"
    }]

    response = await apc.apis[api].call_llm(rag_model,
                              messages,
                              temperature=0.8,
                              max_tokens=2048)

    # Split response into draft and rationale
    parts = response.split("Rationale:", 1)
    draft = parts[0].strip()
    console.print(Panel(draft, title=f"Draft/doc: {did}/{mid}:{specialist_model}", title_align="left", border_style="yellow", style="white"))
    rationale = parts[1].strip() if len(
        parts) > 1 else "No explicit rationale provided."
    console.print(Panel(rationale, title=f"Rationale/doc: {did}/{mid}: {specialist_model}", title_align="left", border_style="yellow", style="white"))
    log = f"Specialist ({specialist_model}) generated a draft."
    return draft, rationale, log


async def evaluator_llm(
        query: str,
        drafts_and_rationales: List[Tuple[str, str]]) -> Tuple[int, str, str]:
    """Call the generalist LLM to evaluate and select the best draft."""
    drafts_text = "\n\n".join([
        f"Draft {i+1}:\n{draft}\nRationale:\n{rationale}"
        for i, (draft, rationale) in enumerate(drafts_and_rationales)
    ])

    rag_model=apc.rag_models['evaluator']
    evaluator_model    =rag_model['name']       
    evaluator_system_prompt=apc.rag_prompt['evaluator']  
    api=rag_model['api']
    

    messages = [{
        "role": "system",
        "content": evaluator_system_prompt
    }, {
        "role":
        "user",
        "content":
        f"Query: {query}\n\nDrafts and Rationales:\n{drafts_text}"
    }]
    response = await apc.apis[api].call_llm(rag_model,
                              messages,
                              temperature=0.3,
                              max_tokens=512)

    # Parse the response to extract the best draft number and rationale
    lines = response.split('\n')
    try:
        best_draft_num = int(
            lines[0].split(':')[1].strip()[0]) - 1  # Convert to 0-based index
    except:
        console.print(Panel(response, title="evaluator_llm", title_align="left", border_style="red", style="white"))
        return None, 'No explicit rationale provided.', 'No explicit rationale provided.'
    rationale = '\n'.join(lines[1:]).strip()

    log = f"Evaluator ({evaluator_model}) selected Draft {best_draft_num + 1} as the best."
    return best_draft_num, rationale, log


async def final_response_llm(query: str, best_draft: str,
                             rationale: str) -> Tuple[str, str]:
    """Call the generalist LLM to craft the final response."""

    rag_model=apc.rag_models['final']
          
    final_response_prompt=apc.rag_prompt['final']  
    api=rag_model['api']

    messages = [{
        "role":
        "system",
        "content":
        final_response_prompt.format(query=query,
                                     best_draft=best_draft,
                                     rationale=rationale)
    }, {
        "role": "user",
        "content": "Please provide the final response."
    }]
    #pp(messages)
    #print('--------------------------------------final_response_llm')    
    response = await apc.apis[api].call_llm(rag_model,
                              messages,
                              temperature=0.7,
                              max_tokens=2048)
    log = f"Final response crafted using the best draft and evaluator's rationale."
    return response, log


async def process_document(mid,did, rag_model, query: str, document: str) -> Tuple[str, str]:
    """Process a single document with the specialist LLM."""
    draft, rationale, _ = await drafter_llm(mid, did, rag_model, query, document)
    return draft, rationale


async def speculative_rag(query: str, documents: List[str]) -> Tuple[str, str]:
    """Implement the Speculative RAG process."""
    process_log = []
    rag_model=apc.rag_models['generalist']
    
    num_of_drafts=apc.pipeline.get('num_of_drafts', 1)
    
  
    console.print(Panel(str(num_of_drafts), title="num_of_drafts", title_align="left", border_style="white", style="white"))
    if 'is_complex' not in apc.pipeline:
        # Step 1: Determine if the query is knowledge-intensive
        is_complex, gen_log = await generalist_llm(query)
        process_log.append(gen_log)
    else:
        is_complex=  apc.pipeline  ['is_complex'] 

    if is_complex:
        console.print(Panel('Complex', title="speculative_rag", title_align="left", border_style="Yellow", style="white"))
        # Step 2: Generate drafts using the specialist LLM for each document

        tasks = [process_document(mid,did, model, query, doc) for did,doc in enumerate(documents) for mid, model in enumerate(apc.rag_models['specialist'])]

        drafts_and_rationales = await asyncio.gather(*tasks)

        # Step 3: Evaluate and select the best draft
        best_draft_num, eval_rationale, eval_log = await evaluator_llm(
            query, drafts_and_rationales)
        process_log.append(eval_log)
        
        # Step 4: Craft final response using the best draft
        if best_draft_num is not None:
            try:
                best_draft = drafts_and_rationales[best_draft_num][0]
            except:
                console.print(Panel(str(best_draft_num), title="best_draft_num", title_align="left", border_style="red", style="white")) 
                pp(drafts_and_rationales)
                final_response = 'No best draft found'
                return final_response, "\n".join(process_log)
            console.print(Panel(str(best_draft_num), title="best_draft_num", title_align="left", border_style="Green", style="white")) 
            final_response, final_log = await final_response_llm(
                query, best_draft, eval_rationale)
            process_log.append(final_log)
        else:
            final_response = 'No best draft found'
            return final_response, "\n".join(process_log)
    else:
        console.print(Panel('Simple', title="speculative_rag", title_align="left", border_style="Yellow", style="white"))
        # For simple queries, use the generalist LLM to generate a response
        documents = [doc for doc in documents if doc]
        docs=''
        content=query
        if documents:
            docs=os.linesep.join([f'Doc {i}: {doc}' for i, doc in enumerate(documents)])
        
            content= f"Query: {query}\n\nDocuments: \n\n{docs}"

        
        generalist_model    =rag_model['name']       
        
        api=rag_model['api']

        messages = [{
            "role":
            "system",
            "content":
            "You are a helpful assistant. Please answer the following query concisely"
        }, {
            "role": "user",
            "content": content
        }]
        final_response = await apc.apis[api].call_llm(rag_model,
                                        messages,
                                        temperature=0.7,
                                        max_tokens=512)
        process_log.append(
            f"Simple query: Generalist ({generalist_model}) provided the response."
        )

    return final_response, "\n".join(process_log)


def read_markdown_file(file_obj) -> str:
    """Read the contents of an uploaded Markdown file."""
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj  # If it's already a string, return it directly
    if hasattr(file_obj, 'name'):
        # If it's a file-like object with a 'name' attribute, read the file
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            return f.read()
    # If we can't handle the input, return an empty string
    return ""



@click.command()
@click.argument('yaml_file_path', type=click.Path(exists=True))


def main(yaml_file_path):
    async def async_main():
        """Run the main loop of the MOA process."""
        with open(yaml_file_path, 'r') as file:
            apc.pipeline = data = yaml.safe_load(file)
            apc.prompt_log['pipeline']=apc.pipeline
            
        apc.prompt_log['rag_models']={}    
        #pp(data)
        if rag_models := data.get('rag_models', None  ):
            save_models(rag_models)
        else:
            raise Exception('No rag_models found')
        if rag_prompt := data.get('rag_prompt', None  ):
            save_prompt(rag_prompt)
        else:
            raise Exception('No rag_prompt found')            


        apc.prompt_log['pipeline']={'models':rag_models}
        print("Running main loop...")


        try:
            while True:

                default_prompt="Justify importance of number 42"
                user_prompt = input(f"Enter your prompt ({default_prompt}): ")
                if not user_prompt:
                    user_prompt = default_prompt


                apc.prompt_log['pipeline']['user_prompt']=user_prompt
                rag_model_prompt=apc.rag_models['generalist'].get('user_prompt', None)
                

                
                if rag_model_prompt:
                      
                    parsed_user_prompt = get_prompt(rag_model_prompt, user_prompt)
                else:
                    parsed_user_prompt=user_prompt

                apc.prompt_log['pipeline']['parsed_user_prompt']=parsed_user_prompt

                #console.print(parsed_user_prompt, style="bold yellow")
                console.print(Panel(parsed_user_prompt, title="User prompt", title_align="left", border_style="white", style="bold yellow"))
                



                """Wrapper to run the full CoT reasoning and display results."""
                
                documents=[read_markdown_file(open(doc,'r') ) for doc in DOCS]

                out, log = await speculative_rag(parsed_user_prompt, documents)


                print()
                apc.prompt_log['final_stream']={}
                apc.prompt_log['final_stream']['result']=out
                apc.prompt_log['result'] = ' '.join(out)
                console.print(Panel(out, title="Output", title_align="left", border_style="bright_blue", style="white"))
        finally:
            await close_clients()
    asyncio.run(async_main())

    

if __name__ == "__main__":
    main()

