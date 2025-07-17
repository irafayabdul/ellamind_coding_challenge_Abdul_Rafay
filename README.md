# ellamind Coding Challenge

Implement an LLM-as-a-judge. Its task is to judge the outputs of other LLMs.

Your judge will be presented with four completions to a given prompt. It needs to judge which one is the best.

Use Gemini 2.0 Flash and try to achieve a high accuracy on the provided dataset.

## Dataset

[data/dev.jsonl](data/dev.jsonl)

### Example

```json
{
  "prompt": "Write a 60 word energetic and stylized bio for a pirate character: Khawlah, a brave, fearless, stoic, focused, and immensely talented Muslim female pirate warrior skilled in swordfighting and has encyclopedic tactical knowledge. The response should not repeat any word more than 2 times.",
  "completions": [ 
    "Cutting through waves and prejudice alike, Khawlah commands respect across seven seas. This legendary Muslim warrior-captain masterfully wields both scimitar and strategy, turning battles into art. Her stoic presence strikes fear into enemies while inspiring fierce loyalty among crew. Tactical brilliance and unwavering faith guide her pursuit of justice on treacherous waters.",
    "Born to the seas, Khawlah commands respect through calculated precision and unwavering faith. This legendary Muslim warrior-captain strikes fear into enemies while inspiring fierce loyalty among her crew. Her tactical brilliance and masterful swordsmanship have become folklore across treacherous waters. Silent yet deadly, she navigates both Islamic principles and pirate life with graceful determination.",
    "Khawlah stalks the seas with unmatched precision, her scimitar gleaming beneath Islamic prayer beads. This legendary warrior-captain strikes fear into merchant vessels while commanding absolute loyalty from her crew. Through tactical brilliance and fierce swordplay, she's claimed countless victories. Silent yet deadly, the stoic pirate navigates both combat and strategy with masterful grace.",
    "Cutting through waves like destiny's blade, Khawlah commands her fleet with unwavering resolve. This legendary Muslim warrior-captain strikes fear into enemies while earning fierce loyalty from her crew. A tactical genius and master swordswoman, she navigates both treacherous waters and complex battle strategies with stoic precision, leaving tales of her fearless exploits echoing across the seas."
  ],
  "chosen_index": 2
}
```

## Requirements

1. Install dependencies: `pip install -r requirements.txt` (tested with Python 3.10).
2. Copy `.env_example` to `.env` and fill in the API key we provided.
3. Implement your full solution in `main.py` only. Do not add any other files. They will be ignored.
4. Before submitting, run `python run_submission.py --debug` to check if your solution is gradable and bug-free.
    - If you want to check the accuracy on the full dataset, omit the `--debug` flag.
    - Make sure that your solution runs in under 20 minutes on the full dataset.
    - If you run into rate limits you can lower `CONCURRENCY` in `run_submission.py`.

Please leave the `requirements.txt` unchanged and don't add any other dependencies.
You can request usage information for the provided API key from [OpenRouter](https://openrouter.ai/docs/api-reference/api-keys/get-current-api-key). The provided API key will be deactivated after the challenge.

## Submission

- Create a private GitHub repository.
- If you're Github repo is public you will spoil the challenge for others. We will automatically reject your submission.
- Please name the repository `ellamind_coding_challenge_John_Doe` where `John_Doe` is your name.
- Invite `ellamind-admin` as collaborator to your repository.
- Ensure your solution is present on the main branch.
- No further changes will be accepted after inviting `ellamind-admin`!
- If you don't follow these instructions the grading will fail.

## Additional Notes

- Grading will use a private test-set with the same structure.
- Time investment: one to three hours
- We would expect the result within a week of receiving the challenge.
- Perfect results aren't expected.
