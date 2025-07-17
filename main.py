import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

load_dotenv()

client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY")
)

# Please make sure to set `model="google/gemini-2.0-flash-001"` when using the client.
# Example:
#
# await client.chat.completions.create(
#   model="google/gemini-2.0-flash-001",
#   messages=[
#     {"role": "user", "content": "Hello, world!"}
#   ]
# )


async def judge_completions(prompt: str, completions: list[str]) -> int:
    """
    Use `google/gemini-2.0-flash-001` to judge the completions.

    Return the index of the better completion, e.g. `2` for the third completion.

    Make sure to always return an integer.
    """
    # We need to construct a detailed prompt for the judging LLM.
    # This prompt provides context, the original user request, the completions to evaluate, and specific instructions for the desired output format.

    # Setting up the Role and Evaluation criteria for our Judge.
    judging_prompt_header = """
    You are an impartial and expert AI judge. Your task is to carefully evaluate a series of completions for a given prompt and determine which one is the best.

    **Evaluation Criteria:**
    1.  **Adherence to Instructions:** The completion must strictly follow all constraints from the original prompt (e.g., word count, tone, character traits, word repetition).
    2.  **Quality of Writing:** The text should be well-written, coherent, engaging, and stylistically appropriate.
    3.  **Fulfillment of Intent:** The completion should best capture the essence and intent of the user's request.

    **Original Prompt:**
    """

    # The original request from user is inserted here.
    judging_prompt_body = f'"{prompt}"\n\n**Completions to Judge:**\n'

    # We can see that we have 0 to 3 as chosen_index in data
    # Each completion is appended with its index for judge's clarity.
    for i, completion in enumerate(completions):
        judging_prompt_body += f'--- Completion {i} ---\n"{completion}"\n'


    # Specific instruction for desired output goes here.
    # For the initial test we go by simple Zero shot Prompting to have the initial baseline for comparison of accuracy, completion time and maybe cost if we are in an actual test environment
    judging_prompt_footer = """
    ---
    **Your Task:**
    Review the completions based on the evaluation criteria and the original prompt. Your response must be a single integer representing the index of the best completion. Do not provide any explanation, reasoning, or other text. For example, if Completion 2 is the best, your output must be only '2'.
    """

    # Combining all parts into the final prompt.
    final_judging_prompt = judging_prompt_header + judging_prompt_body + judging_prompt_footer

    try:
        chat_completion = await client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                # adding is to fix the inspection warning, not strictly needed otherwise
                ChatCompletionUserMessageParam(role="user", content=final_judging_prompt)
            ],
            temperature=0.0,  # Setting temp to 0 for deterministic evaluation
            max_tokens=5,
        )

        # Extracting the content from the response.
        judged_index = chat_completion.choices[0].message.content.strip()

        # As we can see that we have 0 to 3 as chosen_index in data
        # Parsing the integer directly.
        try:
            return int(judged_index)
        except ValueError:
            # using regex to find a single digit from 0 to 3 in the string in case direct parsing fails for us.
            match = re.search(r'[0-3]', judged_index)
            if match:
                return int(match.group(0))
            else:
                # Fallback to the first completion if we do not find any number.
                return 0

    except Exception as e:
        # If any API error occurs, print the error and default to the first completion.
        print(f"An error occurred: {e}")
        return 0
