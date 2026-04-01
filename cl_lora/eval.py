import lm_eval
from lm_eval.models.huggingface import HFLM
import json
import os
from pathlib import Path

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_PATH = "/mnt/C-SSD/ramiro/adapters/llama-3.2-3b-instruct-lora"

# Paper's general evaluation tasks (GP metric, zero-shot)
EVAL_TASKS = ["hellaswag"]

def main():
    model = HFLM(
        pretrained=MODEL,
        peft=LORA_PATH,
        device="cuda",
        dtype="bfloat16",
    )

    # Zero-shot = GP metric from the paper
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=EVAL_TASKS,
        num_fewshot=0,
        batch_size=8,
    )

    # 5-shot = IP metric from the paper
    model_args = dict(
        pretrained=MODEL,
        peft=LORA_PATH,
        device="cuda",
        dtype="bfloat16",
    )
    results_icl = lm_eval.simple_evaluate(
        model=model,
        tasks=EVAL_TASKS,
        num_fewshot=5,
        batch_size=8,
        model_args=model_args
    )

    print("=== GP (zero-shot general performance) ===")
    for task, res in results["results"].items():
        print(f"  {task}: {res}")

    print("\n=== IP (5-shot in-context performance) ===")
    for task, res in results_icl["results"].items():
        print(f"  {task}: {res}")

    # Save results to JSON files
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "gp_results.json", "w") as f:
        json.dump(results["results"], f, indent=2)

    with open(results_dir / "ip_results.json", "w") as f:
        json.dump(results_icl["results"], f, indent=2)

if __name__ == "__main__":
    main()