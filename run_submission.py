from dotenv import load_dotenv
from time import perf_counter
import pandas as pd
import argparse
from datasets import Dataset, load_dataset
import asyncio
from tqdm.asyncio import tqdm


load_dotenv()

CONCURRENCY = 10


async def async_process_dataset(ds: Dataset) -> Dataset:
    """Do not change this function. It is used to process the dataset and return the judged indices."""
    from main import judge_completions

    semaphore = asyncio.Semaphore(CONCURRENCY)
    async def process_example(example: dict) -> int:
        async with semaphore:
            return await judge_completions(example["prompt"], example["completions"])

    tasks = [process_example(example) for example in ds]
    judged_indices = await tqdm.gather(*tasks)
    
    return ds.add_column("judged_index", judged_indices)

def calculate_accuracy(output_path: str):
    print(f"Calculating accuracy from {output_path}")
    df = pd.read_json(output_path, lines=True)
    df["is_correct"] = df["judged_index"] == df["chosen_index"]
    ratio_valid = df["judged_index"] != -1
    return df["is_correct"].mean(), ratio_valid.mean()

def main(dataset_path: str, output_path: str, debug: bool = False):
    print(f"Running on {dataset_path} and saving to {output_path}")
    ds = load_dataset("json", data_files=dataset_path, split="train")
    if debug:
        ds = ds.select(range(2))

    start_time = perf_counter()
    submission_ds = asyncio.run(async_process_dataset(ds))
    end_time = perf_counter()

    minutes = int((end_time - start_time) // 60)
    seconds = int((end_time - start_time) % 60)
    print(f"Time taken: {minutes}:{seconds:02d}")

    submission_ds.to_json(output_path, orient="records", lines=True)

    accuracy, ratio_valid = calculate_accuracy(output_path)
    print(f"Accuracy: {accuracy}")
    print(f"Ratio valid judgements: {ratio_valid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dataset_path", type=str, required=False, default="data/dev.jsonl")
    parser.add_argument("--output_path", type=str, required=False, default="./dev_ds.jsonl")
    args = parser.parse_args()

    main(dataset_path=args.dataset_path, output_path=args.output_path, debug=args.debug)
