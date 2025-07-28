import os
import re
import collections

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
    You are an impartial and meticulous AI judge. Your task is to find the single best completion out of four completions for the given user prompt by following a strict, step-by-step evaluation based on the criteria below, listed in order of importance.


    **Evaluation Criteria:**
    1.  **Correctness & Factual Accuracy (Highest Priority):** The answer must be factually correct. Any calculations, code, or logic must be sound. An incorrect answer is an immediate failure.
    2.  **Fulfillment of Intent:** The completion must fully address the core question and satisfy the user's primary goal.
    3.  **Adherence to Constraints:** The completion must strictly follow all constraints from the original prompt (e.g., word count, tone, character traits, word repetition). A failure on a **major** constraint (e.g. core task) is more severe than a failure on a **minor** one (e.g., word count).
    4.  **Quality of Writing:** The text should be well-written, clear, concise, coherent, and stylistically appropriate for the prompt.
    5.  **Compliance & Formatting:** The response must be safe and use formatting (like markdown or code blocks) correctly and only when requested.

    **Note:** that the answer can only be the index of the best completion ranging from 0 to 3.

    **Original Prompt:**
    """

    # The original request from user is inserted here.
    judging_prompt_body = f'"{prompt}"\n\n**Completions to Judge:**\n'

    # We can see that we have 0 to 3 as chosen_index in data
    # Each completion is appended with its index for judge's clarity.
    for i, completion in enumerate(completions):
        judging_prompt_body += f'--- Completion {i} ---\n"{completion}"\n'


    ## Specific instruction for desired output goes here.

    #### For the initial test we go by simple Zero shot Prompting to have the initial baseline for comparison of accuracy, completion time and maybe cost if we are in an actual test environment
    # judging_prompt_footer = """
    # ---
    # **Your Task:**
    # Review the completions based on the evaluation criteria and the original prompt. Your response must be a single integer representing the index of the best completion. Do not provide any explanation, reasoning, or other text. For example, if Completion 2 is the best, your output must be only '2'.
    # """

    #### This is the Chain-of-Thought prompt. adding Lets think step-by-step instruction automatically triggers zero shot COT in LLM as showed in this paper "https://arxiv.org/pdf/2205.11916" Large Language Models are Zero-Shot Reasoners
    judging_prompt_footer = """
    ---
    ## Your Task
    **Step 1: Individual Evaluation**
    Inside a <step_1_evaluation> block, create a scorecard for **each** completion. For each one, briefly evaluate it against the 5 criteria.

    **Step 2: Comparative Analysis**
    Inside a <step_2_comparison> block, identify the top 2-3 contenders from Step 1. Write a direct, head-to-head comparison explaining the strengths and weaknesses of these top contenders against each other. Explicitly state which one is superior and why.

    **Step 3: Final Judgment**
    Finally, after the comparison, state the single integer index of the best completion inside `<final_answer>` tags.

    ## Example of your response format:
    <step_1_evaluation>
    **Analysis of Completion 0:**
    1. Correctness: [Your analysis]
    2. Intent: [Your analysis]
    3. Constraints: [Your analysis]
    4. Quality: [Your analysis]
    5. Compliance: [Your analysis]

    **Analysis of Completion 1:**
    ... and so on for all completions.
    </step_1_evaluation>

    <step_2_comparison>
    The top contenders are Completion 2 and Completion 3.

    Completion 2 fully adheres to all constraints, while Completion 3 misses the negative constraint on word repetition. Although Completion 3 has slightly more vivid language, the adherence to constraints in Completion 2 makes it the superior choice.
    </step_2_comparison>

    <final_answer>2</final_answer>
    """

    # Combining all parts into the final prompt.
    final_judging_prompt = judging_prompt_header + judging_prompt_body + judging_prompt_footer

    judgments = []

    # Run the judgment process 5 times to get a more reliable answer.
    for _ in range(5):
        try:
            chat_completion = await client.chat.completions.create(
                model="google/gemini-2.0-flash-001",  # "google/gemini-2.0-flash-exp:free"
                messages=[
                    # adding is to fix the inspection warning, not strictly needed otherwise
                    ChatCompletionUserMessageParam(role="user", content=final_judging_prompt)
                ],
                temperature=0.3,  # Setting temp to 0 for deterministic evaluation
                max_tokens=4096,  # Increased max_tokens to allow for reasoning
                # stop=["</final_answer>"],
                # seed=12345,
                top_p=0.5,
                # frequency_penalty=0.2,
                # presence_penalty=0.2,
                # top_logprobs=10
            )

            # Extracting the content from the response.
            judged_index = chat_completion.choices[0].message.content.strip()

            #### This is direct parsing for Zero shot prompt
            # As we can see that we have 0 to 3 as chosen_index in data
            # Parsing the integer directly.
            # try:
            #     return int(judged_index)
            # except ValueError:
            #     # using regex to find a single digit from 0 to 3 in the string in case direct parsing fails for us.
            #     match = re.search(r'[0-3]', judged_index)
            #     if match:
            #         return int(match.group(0))
            #     else:
            #         # Fallback to the first completion if we do not find any number.
            #         return 0

            #### Parse the COT answer from within the <final_answer> tags.
            match = re.search(r'<final_answer>([0-3])</final_answer>', judged_index)
            if match:
                judgments.append(int(match.group(1)))
            else:
                # Fallback if the model fails to use the specified format.
                judgments.append(0)


        except Exception as e:
            # If any API error occurs, print the error and default to the first completion.
            print(f"An error occurred: {e}")
            judgments.append(0)

    # Find the most common judgment (majority vote).
    if not judgments:
        return 0

    most_common_judgment = collections.Counter(judgments).most_common(1)[0][0]
    return most_common_judgment