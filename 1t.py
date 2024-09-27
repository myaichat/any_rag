import os, sys
import asyncio
from pprint import pprint as pp
from groq import AsyncGroq
import time
e=sys.exit

# Initialize Groq client
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# Define model
model = "llama-3.1-70b-versatile"

# Initial system prompt (regular Chain of Thought)
initial_system_prompt = """You are an AI assistant capable of detailed, step-by-step thinking. When presented with a question or problem, break down your thought process into clear, logical steps. For each step, explain your reasoning. Conclude with a final answer. Use the following markdown structure:

## Reasoning
1. [First step]
   **Explanation:** [Detailed explanation of this step]
2. [Second step]
   **Explanation:** [Detailed explanation of this step]
...

## Answer
[Final answer]

Be comprehensive and show your reasoning clearly."""

# Followup system prompt
followup_system_prompt = """You are an AI assistant tasked with analyzing and improving upon previous problem-solving steps. Review the original query and the previous turns of reasoning, then provide a new perspective or deeper analysis. Use the following markdown structure:

## Critique
[Provide a brief critique of the previous reasoning, highlighting its strengths and potential weaknesses]

## New Reasoning
1. [First step of new or refined approach]
   **Explanation:** [Detailed explanation of this step, referencing the previous reasoning if relevant]
2. [Second step of new or refined approach]
   **Explanation:** [Explanation of how this step builds upon or differs from the previous thinking]
...

## Updated Answer
[Updated answer based on this new analysis]

Be critical yet constructive, and strive to provide new insights or improvements."""

# Synthesis prompt
synthesis_prompt = """You are an AI assistant tasked with synthesizing multiple turns of reasoning into a final, comprehensive answer. You will be presented with three different turns of reasoning for solving a problem. Your task is to:

1. Analyze each turn, considering its strengths and weaknesses.
2. Compare and contrast the different methods.
3. Synthesize the insights from all turns into a final, well-reasoned answer.
4. Provide a concise, clear final answer that a general audience can understand.

Use the following markdown structure:

## Analysis of Turns
[Provide a brief analysis of each turn of reasoning]

## Comparison
[Compare and contrast the turns, highlighting key differences and similarities]

## Final Reasoning
[Provide a final, synthesized reasoning process that combines the best insights from all turns]

## Comprehensive Final Answer
[Comprehensive final answer]

## Concise Answer
[A brief, clear, and easily understandable version of the final answer, suitable for a general audience. This should be no more than 2-3 sentences.]

Be thorough in your analysis and clear in your reasoning process."""


async def call_llm(messages: list,
                   temperature: float = 0.7,
                   max_tokens: int = 8000) -> str:
    """Call the Groq API."""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def generate_turn(query: str, previous_turns: list = None) -> str:
    """Generate a single turn of reasoning, considering previous turns if available."""
    is_first_turn = previous_turns is None or len(previous_turns) == 0
    if is_first_turn:
        messages = [{
            "role": "system",
            "content": initial_system_prompt
        }, {
            "role": "user",
            "content": query
        }]
    else:
        previous_content = "\n\n".join(previous_turns)
        messages = [{
            "role": "system",
            "content": followup_system_prompt
        }, {
            "role":
            "user",
            "content":
            f"Original Query: {query}\n\nPrevious Turns:\n{previous_content}\n\nProvide the next turn of reasoning."
        }]

    return await call_llm(messages)


async def synthesize_turns(query: str, turns: list) -> str:
    """Synthesize multiple turns of reasoning into a final answer."""
    turns_text = "\n\n".join(
        [f"Turn {i+1}:\n{turn}" for i, turn in enumerate(turns)])
    messages = [{
        "role": "system",
        "content": synthesis_prompt
    }, {
        "role":
        "user",
        "content":
        f"Original Query: {query}\n\nTurns of Reasoning:\n{turns_text}"
    }]
    return await call_llm(messages)


async def full_cot_reasoning(query: str) -> tuple:
    """Perform full Chain of Thought reasoning with multiple turns."""
    start_time = time.time()
    turns = []
    turn_times = []
    full_output = f"# Chain of Thought Reasoning\n\n## Original Query\n{query}\n\n"

    for i in range(3):  # Generate 3 turns of reasoning
        turn_start = time.time()
        turn = await generate_turn(query, turns)
        turns.append(turn)
        turn_times.append(time.time() - turn_start)
        full_output += f"## Turn {i+1}\n{turn}\n\n"

    mid_time = time.time()
    synthesis = await synthesize_turns(query, turns)
    full_output += f"## Synthesis\n{synthesis}\n\n"
    end_time = time.time()

    timing = {
        'turn_times': turn_times,
        'total_turns_time': mid_time - start_time,
        'synthesis_time': end_time - mid_time,
        'total_time': end_time - start_time
    }

    full_output += f"## Timing Information\n"
    full_output += f"- Turn 1 Time: {timing['turn_times'][0]:.2f}s\n"
    full_output += f"- Turn 2 Time: {timing['turn_times'][1]:.2f}s\n"
    full_output += f"- Turn 3 Time: {timing['turn_times'][2]:.2f}s\n"
    full_output += f"- Total Turns Time: {timing['total_turns_time']:.2f}s\n"
    full_output += f"- Synthesis Time: {timing['synthesis_time']:.2f}s\n"
    full_output += f"- Total Time: {timing['total_time']:.2f}s\n"

    return full_output


def main(query):
    """Wrapper to run the full CoT reasoning and display results."""
    output = asyncio.run(full_cot_reasoning(query))
    return output

if __name__ == "__main__":
    query = """
incorporate vladimir putin into this image prompt, make him look grotesque and miserable.
return a detailed description of updated  image prompt in <fused_image> tags.
Input Image prompt:
balenciaga new 'blackonblack' ad, in the style of surrealistic biomechanics, a russian man, unsettling beauty, baby blue eyes, white skin, he has Lime green tattoos on his neck and chest, light pink hair style, eye-catching, androgynous, his neck and upper chin are covered with a sleek black steel robotic device, wearing a blue bomber jacket covered with white anime characters, asymmetrical balance, emotional scene, Akira Toriyama, wearing a clear skull mask to his face made of glass, skull motifs, hiphop aesthetics, made of glass, in style of the movie Akira Toriyama anime, dragon ball , osgemeos, exacting precision, hedi xandt, Timothy Hogan, pure white background, high detail, hyper quality, high texture details, high definition, sharp, iconic, magazine cover, frontal, blink and you’ll miss, frontal, tense gaze, scanner photography
"""

    query = """
add pinup essence to this image prompt, make it look like a vintage pinup poster.
return a detailed description of updated  image prompt in <fused_image> tags.
Input Image prompt:
art-deco space-force recruiting propaganda pin-ups promoting women space nurses
"""


    query = """
incorporate ukrainian flag  this image prompt, make it look like a vintage  poster.
return a detailed description of updated  image prompt in <fused_image> tags.
Input Image prompt:
soft anime-style digital illustration, young girl with hair bun sitting on brick wall (hex #B15E3D) under large leafy tree, girl wearing oversized striped sweater and knee-high socks, five cute cats of different colors surrounding her, pastel blue sky (hex #E6F3F7) with fluffy white clouds, vibrant green foliage (hex #8DC63F) with dappled sunlight effect, warm summer day atmosphere, watercolor texture throughout, soft shadows and highlights, no outlines, dreamy and peaceful mood, inspired by Studio Ghibli art style, highly detailed cat expressions and fur textures, subtle brick texture on wall, gentle color gradients in sky and foliage, balanced composition with girl as focal point
"""

    query = """
add ukrainian essence to this image prompt preserving its artistic essence
return a detailed description of updated  image prompt in <fused_image> tags.
Input Image prompt:

A woman with long dark hair and bangs wears an orange mask made of horizontal stripes on her face, in the style of John Holcroft, collage art print in the style of halftone print, minimalism, with purple, blue, pink, and orange tones, portrait close-up.
"""
    print(main(query))