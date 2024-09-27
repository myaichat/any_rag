import os, sys
from pprint import pprint as pp
import include.config.init_config as init_config 
apc = init_config.apc
e=sys.exit

generic_aggregator_system_prompt = """
You have been provided with a set of responses from various open-source models to the latest user query. 
Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically 
evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive 
reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards 
of accuracy and reliability.

Responses from models:"""

image_aggregator_system_prompt = """
You have been provided with a set of artistic image descriptions from various open-source models in response to the 
latest user query. Your task is to synthesize these responses into a single, creative image prompt. It is crucial 
to produce the most creative and weird image description. Ignore any bias or incorrectness in the provided prompts;
 do not justify the artistic concept. Do not create the image. Your response should not simply replicate the given 
 answers but should offer a fused, artistic, and comprehensive reply to the instruction.
   Return a 200-word artistic image description.

Responses from models:"""

def get_final_system_prompt( results):
    
    if 'system_prompt' in apc.pipeline:
        
        return  (
        apc.pipeline['system_prompt']
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )
        

    if 'image' in apc.pipeline:
        aggregator_system_prompt = image_aggregator_system_prompt
    else:
        aggregator_system_prompt = generic_aggregator_system_prompt
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        aggregator_system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


def get_aggregator(data):
    if 'aggregator' in data:
        if reference_models := data.get('reference_models', None  ):
            reference_aggregator_model = next((model['name'] for model in data['reference_models'] if model.get('aggregator')), None)
            assert reference_aggregator_model is None, f"Ignoring aggregator defined in reference_models: {reference_aggregator_model}"
        assert len(data['aggregator'])==1, f"Only one aggregator model is supported"
        aggregator_model = data['aggregator'][0]['name']
        aggregator_api = data['aggregator'][0]['api']
    else:
        aggregator_model = next((model['name'] for model in data['reference_models'] if model.get('aggregator')), None)
        aggregator_api = next((model['api'] for model in data['reference_models'] if model.get('aggregator')), None)
    assert aggregator_model, f"Aggregator model not found"
    assert aggregator_api, f"Aggregator api not found"
    return aggregator_model, aggregator_api

