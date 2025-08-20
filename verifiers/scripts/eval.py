import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

import verifiers as vf
from verifiers.utils.report_utils import (
    ReportMeta,
    get_env_version,
    write_html_report,
)


def eval_environment(
    env: str,
    env_args: dict,
    endpoints_path: str,
    model: str,
    api_key_var: str,
    api_base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent_requests: int,
    max_tokens: int,
    temperature: float,
    verbose: bool,
    write_report: bool,
    save_dataset: bool,
    save_path: str,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: str,
):
    try:
        endpoints_path_obj = Path(endpoints_path)
        if endpoints_path_obj.is_dir():
            endpoints_file = endpoints_path_obj / "endpoints.py"
        else:
            endpoints_file = endpoints_path_obj

        if endpoints_file.exists():
            spec = importlib.util.spec_from_file_location("endpoints", endpoints_file)
            assert spec and spec.loader
            endpoints_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(endpoints_module)
            ENDPOINTS = endpoints_module.ENDPOINTS
        else:
            raise ImportError(f"endpoints.py not found at {endpoints_file}")
    except (ImportError, AttributeError):
        print(
            f"No local endpoint registry found at {endpoints_path}. \
Please specify the model name (-m), API host base URL (-b), and API key variable name (-k)."
        )
        ENDPOINTS = {}

    if model in ENDPOINTS:
        api_key_var = ENDPOINTS[model]["key"]
        api_base_url = ENDPOINTS[model]["url"]
        model = ENDPOINTS[model]["model"]

    client = OpenAI(api_key=os.getenv(api_key_var, "EMPTY"), base_url=api_base_url)
    vf_env = vf.load_environment(env_id=env, **env_args)
    sampling_args = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    results = vf_env.evaluate(
        client=client,
        model=model,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent_requests=max_concurrent_requests,
    )
    print("--- Evaluation ---")
    print(f"Environment: {env}")
    print(f"Model: {model}")
    print(f"Provider: {api_base_url}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")

    print("--- Example ---")
    vf.print_prompt_completions_sample(
        results.prompt, results.completion, results.reward, step=0
    )
    print("--- All ---")
    print("Rewards:")
    print(
        f"reward: avg - {sum(results.reward) / len(results.reward):.3f}, std - {np.std(results.reward):.3f}"
    )
    n = num_examples
    r = rollouts_per_example
    if verbose:
        for i in range(len(results.prompt)):
            print(f"Prompt: {results.prompt[i]}")
            print(f"Completion: {results.completion[i]}")
            print(f"Reward: {results.reward[i]}")
            print(f"Answer: {results.answer[i]}")
            print(f"Info: {results.info[i]}")
            print(f"Task: {results.task[i]}")
    if n < 0:
        n = len(results.reward) // r
    for i in range(r):
        # rounded to 3 decimal places
        trials = [round(results.reward[(i * n) + j], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)
    for k in results.metrics:
        v = results.metrics[k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            # rounded to 3 decimal places
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)
    if save_dataset:
        dataset = vf_env.make_dataset(results)
        dataset.save_to_disk(save_path)
        print(f"Saved dataset to {save_path}")
    if save_to_hf_hub:
        if hf_hub_dataset_name == "":
            dataset_name = f"{env}--{model}--results"
        else:
            dataset_name = hf_hub_dataset_name
        vf_env.make_dataset(results, push_to_hub=True, hub_name=dataset_name)

    if write_report:
        try:
            # Determine environment directory under ./environments if present; otherwise fall back to cwd
            # module file path
            module_name = env.replace("-", "_")
            local_env_dir = Path("./environments") / module_name
            if local_env_dir.exists():
                report_dir = local_env_dir / "reports"
            else:
                report_dir = Path("./reports")

            meta = ReportMeta(
                env_id=env,
                env_version=get_env_version(module_name),
                model=model,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                api_base_url=api_base_url,
                sampling_args=sampling_args,
                env_args=env_args,
            )
            out_path = write_html_report(
                report_dir=report_dir, meta=meta, results=results
            )
            print(f"Saved HTML report to {out_path}")
        except Exception as e:
            print(f"Failed to write HTML report: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env", type=str, default="gsm8k", help="Environment module name"
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default={},
        help='Environment module arguments as JSON object (e.g., \'{"key": "value", "num": 42}\')',
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default="./configs/endpoints.py",
        help="Path to API endpoints registry",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4.1-mini",
        help="Name of model to evaluate",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for API",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=5,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=3,
        help="Number of rollouts per example",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        "-c",
        type=int,
        default=32,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", "-T", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--write-report",
        "-w",
        default=False,
        action="store_true",
        help="Write HTML report",
    )
    parser.add_argument(
        "--save-dataset",
        "-s",
        default=False,
        action="store_true",
        help="Save dataset to disk",
    )
    parser.add_argument(
        "--save-path",
        "-p",
        type=str,
        default="results",
        help="Path to save dataset",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        default=False,
        action="store_true",
        help="Save dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default="",
        help="Name of dataset to save to Hugging Face Hub",
    )
    args = parser.parse_args()

    eval_environment(
        env=args.env,
        env_args=args.env_args,
        endpoints_path=args.endpoints_path,
        model=args.model,
        api_key_var=args.api_key_var,
        api_base_url=args.api_base_url,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent_requests=args.max_concurrent_requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
        write_report=args.write_report,
        save_dataset=args.save_dataset,
        save_path=args.save_path,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name,
    )


if __name__ == "__main__":
    main()
