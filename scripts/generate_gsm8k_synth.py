"""Generate synthetic GSM8K-style math problems using the synth builder.

Usage:
    uv run python scripts/generate_gsm8k_synth.py
    uv run python scripts/generate_gsm8k_synth.py --num-seeds 20 --samples-per-subtopic 5
    uv run python scripts/generate_gsm8k_synth.py --filter-ceiling 0.2
"""

import argparse
import asyncio
import logging

from environments.gsm8k.gsm8k import load_environment
from verifiers.synth import SynthDataBuilder

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic GSM8K samples")

    p.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of dataset rows to use as planning seeds (default: 3)",
    )
    p.add_argument(
        "--generator-model",
        default="openai/gpt-5.4-mini",
        help="Model for planning & generation (default: openai/gpt-5.4-mini)",
    )
    p.add_argument(
        "--filter-model",
        default="openai/gpt-5.4-mini",
        help="Model for verification filtering (default: openai/gpt-5.4-mini)",
    )
    p.add_argument(
        "--samples-per-subtopic",
        type=int,
        default=3,
        help="Samples to generate per subtopic (default: 3)",
    )
    p.add_argument(
        "--max-subtopics",
        type=int,
        default=None,
        help="Max subtopics per seed; None lets the LLM decide (default: None)",
    )
    p.add_argument(
        "--filter-threshold",
        type=float,
        default=0.8,
        help="Min learnability score to keep a sample (default: 0.8)",
    )
    p.add_argument(
        "--filter-ceiling",
        type=float,
        default=None,
        help="Max without-context score for novelty check (default: None)",
    )
    p.add_argument(
        "--coverage-quality",
        type=float,
        default=0.8,
        help="Min per-subtopic learnability rate (default: 0.8)",
    )
    p.add_argument(
        "--output-dir",
        default="./synth_output/gsm8k",
        help="Output directory (default: ./synth_output/gsm8k)",
    )

    return p.parse_args()


async def main():
    args = parse_args()

    env = load_environment()

    builder = SynthDataBuilder(
        env=env,
        generator_model=args.generator_model,
        filter_model=args.filter_model,
    )

    result = await builder.build(
        max_seed_examples=args.num_seeds,
        samples_per_subtopic=args.samples_per_subtopic,
        max_subtopics=args.max_subtopics,
        filter_threshold=args.filter_threshold,
        filter_ceiling=args.filter_ceiling,
        coverage_quality=args.coverage_quality,
    )

    result.save(args.output_dir)

    print(f"\nGenerated {result.stats['total_generated']} samples")
    print(f"Kept {result.stats['total_filtered']} after filtering")
    print(f"Pass rate: {result.stats['pass_rate']:.1%}")

    failures = result.coverage_failures
    if failures:
        print(f"\nCoverage failures ({len(failures)} subtopics):")
        for name in failures:
            info = result.stats["coverage"][name]
            print(f"  - {name}: {info['rate']:.0%} learnability")

    print(f"\nOutput saved to {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
