import argparse

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer  # type: ignore

import verifiers as vf

"""
accelerate launch --config-file configs/zero3.yaml --num-processes 8 examples/sft.py
"""


def main(args):
    # convenience function for FA2 initialization
    model, tokenizer = vf.get_model_and_tokenizer(args.model, use_liger=False)
    dataset = load_dataset(args.dataset, split="train")

    tok_counts = []
    for row in dataset:
        # count tokens in (prompt, completion)
        messages = row["prompt"] + row["completion"]  # type: ignore
        toks = tokenizer.apply_chat_template(messages, tokenize=True)
        tok_counts.append(len(toks))

    # tok count stats
    print(f"Dataset size: {len(tok_counts)}")
    print(f"Min tokens: {min(tok_counts)}")
    print(f"Max tokens: {max(tok_counts)}")
    print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
    print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

    args = SFTConfig(
        max_length=args.max_length,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1,
        save_only_model=True,
        log_on_each_node=True,
        push_to_hub=True,
        hub_model_id=args.name_to_save,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,  # type: ignore
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="willcb/Qwen3-1.7B")
    parser.add_argument("--dataset", "-d", type=str, default="willcb/V3-wordle")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs")
    parser.add_argument("--name-to-save", "-n", type=str, default="Qwen3-1.7B-Wordle")
    parser.add_argument("--max-length", "-l", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", "-b", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", "-g", type=int, default=1)
    parser.add_argument("--learning-rate", "-r", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", "-e", type=int, default=3)
    parser.add_argument("--weight-decay", "-w", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", "-g", type=float, default=0.1)
    parser.add_argument("--push-to-hub", "-p", type=bool, default=True)
    args = parser.parse_args()
    main(args)
