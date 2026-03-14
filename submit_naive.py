"""
Submit naive baseline experiment only.

Paper result: Baseline performance floor for all comparisons.
"""
import argparse
from src.submit import (
    ExperimentSpec, add_common_submit_args, submit_experiment_batch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Submit naive baseline experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_submit_args(parser)
    parser.set_defaults(result_dir="results_naive")
    args = parser.parse_args()

    # Naive-only: submit_experiment_batch with an empty spec list + include_naive=True
    submit_experiment_batch(
        specs=[],
        base_dir=args.result_dir,
        total_chunks=args.total_chunks,
        include_naive=True,
    )


if __name__ == "__main__":
    main()
