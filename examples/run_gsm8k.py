import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
import re
from pathlib import Path
import pandas as pd
from typing import List, Set
import json
from datetime import datetime
import asyncio
import threading
import argparse
from mmengine import DictAction

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.logger import logger
from src.config import config
from src.models import model_manager
from src.agent import create_agent, prepare_response
from src.registry import DATASET

append_answer_lock = threading.Lock()


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"
    print("Answer exported to file:", jsonl_file.resolve())


def get_tasks_to_run(answers_file: str, dataset) -> List[dict]:
    data = dataset.data

    logger.info(f"Loading answers from {answers_file}...")
    try:
        if os.path.exists(answers_file):
            answer_df = pd.read_json(answers_file, lines=True)
            if "task_id" not in answer_df.columns:
                logger.warning(
                    f"Answers file {answers_file} does not contain 'task_id' column. Please check the file format."
                )
                done_questions = []
            else:
                done_questions = answer_df["task_id"].tolist()
                logger.info(f"Found {len(done_questions)} previous results!")
        else:
            done_questions = []
    except Exception as e:
        logger.warning("Error when loading records: %s", e)
        logger.warning("No usable records! ▶️ Starting new.")
        done_questions = []

    return [
        line for line in data.to_dict(orient="records")
        if line["task_id"] not in done_questions
    ]


def extract_final_answer_from_text(text: str) -> str:
    if not text:
        return ""

    content = str(text)
    match = re.search(r"####\s*([^\n\r]+)", content)
    if match:
        return match.group(1).strip()

    match = re.search(r"\\boxed\{([^}]*)\}", content)
    if match:
        return match.group(1).strip()

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", content)
    if numbers:
        return numbers[-1].strip()

    return content.strip()


def normalize_answer_text(answer: str) -> str:
    if answer is None:
        return ""
    normalized = str(answer).strip().lower()
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".。")
    return normalized


def is_prediction_correct(prediction: str, true_answer: str) -> bool:
    pred_answer = normalize_answer_text(extract_final_answer_from_text(prediction))
    gt_answer = normalize_answer_text(extract_final_answer_from_text(true_answer))
    return bool(pred_answer) and pred_answer == gt_answer


def print_run_accuracy(answers_file: str, run_task_ids: Set[str]) -> None:
    if not run_task_ids:
        print("Accuracy: 0/0 = N/A (no tasks run)")
        return

    if not os.path.exists(answers_file):
        print(f"Accuracy: 0/0 = N/A (answers file not found: {answers_file})")
        return

    try:
        answer_df = pd.read_json(answers_file, lines=True)
    except Exception as e:
        logger.warning("Failed to read answers file for accuracy: %s", e)
        print("Accuracy: 0/0 = N/A (failed to parse answers file)")
        return

    if answer_df.empty or "task_id" not in answer_df.columns:
        print("Accuracy: 0/0 = N/A (no runnable records found)")
        return

    answer_df["task_id"] = answer_df["task_id"].astype(str)
    run_df = answer_df[answer_df["task_id"].isin(run_task_ids)]
    if run_df.empty:
        print("Accuracy: 0/0 = N/A (no records for current run)")
        return

    run_df = run_df.drop_duplicates(subset=["task_id"], keep="last")
    total = len(run_df)
    correct = sum(
        is_prediction_correct(row.get("prediction"), row.get("true_answer"))
        for _, row in run_df.iterrows()
    )
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")


async def answer_single_question(config, example):
    try:
        agent = await create_agent(config)
        logger.visualize_agent_tree(agent)

        logger.info(f"Task Id: {example['task_id']}, Final Answer: {example['true_answer']}")

        augmented_question = example["question"]

        if example.get("file_name"):
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            file_description = f" - Attached file: {example['file_name']}"
            prompt_use_files += file_description
            augmented_question += prompt_use_files

        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        final_result = await agent.run(task=augmented_question)
        agent_memory = await agent.write_memory_to_messages(summary_mode=True)
        final_result = await prepare_response(
            augmented_question,
            agent_memory,
            reformulation_model=model_manager.registed_models["qwen-plus"],
        )

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = [str(step) for step in agent.memory.steps]

        parsing_error = any("AgentParsingError" in step for step in intermediate_steps)
        iteration_limit_exceeded = "Agent stopped due to iteration limit or time limit." in output
        raised_exception = False
        exception = None

    except Exception as e:
        logger.info("Error on %s: %s", example.get("question", ""), e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": config.agent_config.name,
        "question": example["question"],
        "augmented_question": augmented_question if "augmented_question" in locals() else example["question"],
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time if "start_time" in locals() else None,
        "end_time": end_time,
        "task": example.get("task", "gsm8k"),
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
    }
    append_answer(annotated_example, config.save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run agent on GSM8K dataset")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "config_gsm8k.py"),
        help="config file path",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config settings, e.g. dataset.split=train concurrency=4",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    config.init_config(args.config, args)

    logger.init_logger(log_path=config.log_path)
    logger.info(f"| Logger initialized at: {config.log_path}")
    logger.info(f"| Config:\n{config.pretty_text}")

    model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info("| Registed models: %s", ", ".join(model_manager.registed_models.keys()))

    dataset = DATASET.build(config.dataset)
    logger.info(f"| Loaded dataset: {len(dataset)} examples.")

    tasks_to_run = get_tasks_to_run(config.save_path, dataset)[:10]
    run_task_ids = {str(task["task_id"]) for task in tasks_to_run}
    logger.info(f"| Loaded {len(tasks_to_run)} tasks to run.")

    batch_size = getattr(config, "concurrency", 1)
    for i in range(0, len(tasks_to_run), batch_size):
        batch = tasks_to_run[i:min(i + batch_size, len(tasks_to_run))]
        await asyncio.gather(*[answer_single_question(config, task) for task in batch])
        logger.info(f"| Batch {i // batch_size + 1} done.")

    print_run_accuracy(config.save_path, run_task_ids)


if __name__ == "__main__":
    asyncio.run(main())
