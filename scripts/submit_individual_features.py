"""
Submit per-feature marginal value experiments within a subgroup.

Paper result: Appendix table showing each feature's individual contribution.
Replaces the old submit_moments.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.cli.backends import get_backend
from src.cli.submit import (
    ExperimentSpec,
    add_common_submit_args,
    build_extra_args,
    submit_experiment_batch,
)
from src.config import DEFAULT_RESULTS_DIR
from src.feature_groups import SUBGROUPS


def main():
    parser = argparse.ArgumentParser(
        description="Submit individual feature experiments within a subgroup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_submit_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        help="Model to use.",
    )
    parser.add_argument(
        "--subgroup",
        type=str,
        default="moments",
        help=f"Subgroup whose features to test individually. Choices: {list(SUBGROUPS.keys())}.",
    )
    args = parser.parse_args()

    if args.result_dir == DEFAULT_RESULTS_DIR:
        args.result_dir = f"results/individual_{args.subgroup}"

    features = SUBGROUPS[args.subgroup]
    if not features:
        print(f"Subgroup '{args.subgroup}' has no features to test individually.")
        return

    feature_type = "har"
    extra_args = build_extra_args(feature_type, args)

    specs = [
        ExperimentSpec(
            exp_id=i + 1,
            exp_name=f"{args.subgroup}_{feature}",
            model_type=args.model,
            feature_type=feature_type,
            variables=[feature],
            extra_args=extra_args,
        )
        for i, feature in enumerate(features)
    ]

    backend = get_backend(args.backend)
    submit_experiment_batch(
        specs=specs,
        base_dir=args.result_dir,
        total_chunks=args.total_chunks,
        include_naive=not args.no_naive,
        backend=backend,
    )


if __name__ == "__main__":
    main()
