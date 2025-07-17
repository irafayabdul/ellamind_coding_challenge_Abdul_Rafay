import os
from dotenv import load_dotenv
from openai import AsyncOpenAI


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
    # return judged_index
    pass
