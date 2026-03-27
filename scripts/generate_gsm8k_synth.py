"""Generate synthetic GSM8K-style math problems using the synth builder.

Usage:
    uv run python scripts/generate_gsm8k_synth.py
    uv run python scripts/generate_gsm8k_synth.py --num-seeds 20 --samples-per-subtopic 5
    uv run python scripts/generate_gsm8k_synth.py --filter-mode icl_calibrated --filter-ceiling 0.2
"""

import argparse
import asyncio
import logging

from environments.gsm8k.gsm8k import load_environment
from verifiers.synth import SynthDataBuilder

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic GSM8K samples")

    p.add_argument("--num-seeds", type=int, default=10,
                    help="Number of training examples to use as seeds (default: 10)")
    p.add_argument("--generator-model", default="openai/gpt-4.1",
                    help="Model for planning & generation (default: openai/gpt-4.1)")
    p.add_argument("--filter-model", default="openai/gpt-4.1",
                    help="Model for verification filtering (default: openai/gpt-4.1)")
    p.add_argument("--samples-per-subtopic", type=int, default=3,
                    help="Samples to generate per subtopic (default: 3)")
    p.add_argument("--subtopic-branches", type=int, default=2,
                    help="Subtopic branches per seed (default: 2)")
    p.add_argument("--filter-mode", choices=["standard", "icl_calibrated"],
                    default="standard",
                    help="Verification strategy (default: standard)")
    p.add_argument("--filter-threshold", type=float, default=0.8,
                    help="Min score to keep a sample (default: 0.8)")
    p.add_argument("--filter-ceiling", type=float, default=0.2,
                    help="Max without-context score for icl_calibrated mode (default: 0.2)")
    p.add_argument("--output-dir", default="./synth_output/gsm8k",
                    help="Output directory (default: ./synth_output/gsm8k)")

    return p.parse_args()


async def main():
    args = parse_args()

    env = load_environment(num_train_examples=args.num_seeds)

    builder = SynthDataBuilder(
        env=env,
        generator_model=args.generator_model,
        filter_model=args.filter_model,
    )

    result = await builder.build(
        samples_per_subtopic=args.samples_per_subtopic,
        subtopic_branches=args.subtopic_branches,
        filter_mode=args.filter_mode,
        filter_threshold=args.filter_threshold,
        filter_ceiling=args.filter_ceiling,
    )

    result.save(args.output_dir)

    print(f"\nGenerated {result.stats['total_generated']} samples")
    print(f"Kept {result.stats['total_filtered']} after filtering")
    print(f"Pass rate: {result.stats['pass_rate']:.1%}")
    print(f"Output saved to {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
